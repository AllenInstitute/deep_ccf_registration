import monai.networks.nets
import torch
import torch.nn as nn


class CoordConv(nn.Module):
    """from An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, Liu et al"""

    def __init__(self):
        super().__init__()
        self.register_buffer('coords', None, persistent=False)

    def forward(self, batch_size: int, H: int, W: int, device=None):
        if self.coords is None or self.coords.shape[-2:] != (H, W):
            if device is None:
                device = next(self.parameters(), torch.tensor(0)).device

            coords = torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'),
                dim=0
            ).float()  # (2, H, W)
            coords[0] /= (H - 1)
            coords[1] /= (W - 1)

            self.coords = coords.to(device)

        return self.coords.unsqueeze(0).expand(batch_size, -1, -1, -1)


class UNetWithRegressionHeads(nn.Module):
    """
    UNet backbone with separate regression heads for coordinates and classification head for tissue mask.

    This architecture uses UNet for feature extraction and then separate task-specific heads:
    - Coordinate regression head: predicts 3D template coordinates
    - Tissue mask classification head: predicts tissue presence (binary)

    Optional positional encoding can be added to provide explicit spatial location information
    to the regression heads, helping the network learn coordinate mappings more efficiently.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        feature_channels: int = 64,
        dropout: float = 0.0,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        strides: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        out_coords: int = 3,
        include_tissue_mask: bool = False,
        use_positional_encoding: bool = False,
        pos_encoding_channels: int = 16,
    ):
        """
        Parameters
        ----------
        spatial_dims: Number of spatial dimensions (2 for 2D slices)
        in_channels: Number of input channels (1 for grayscale images)
        feature_channels: Number of feature channels output by UNet backbone
        dropout: Dropout rate for UNet
        channels: List of channel sizes for UNet encoder/decoder
        strides: List of stride sizes for UNet encoder/decoder
        out_coords: Number of output coordinate channels (3 for ML, DV, AP)
        include_tissue_mask: Whether to include tissue mask head
        head_size: Size of regression head - "small", "medium", or "large"
        use_positional_encoding: Whether to use learned positional encoding
        pos_encoding_channels: Number of channels for positional encoding
        image_height: Height of input images (required if use_positional_encoding=True)
        image_width: Width of input images (required if use_positional_encoding=True)
        """
        super().__init__()

        self.out_coords = out_coords
        self.include_tissue_mask = include_tissue_mask
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_channels = pos_encoding_channels

        coord_feature_dim = feature_channels
        feature_channels = feature_channels + 1 if include_tissue_mask else feature_channels

        if self.use_positional_encoding:
            self.positional_encoding = CoordConv()
            in_channels += 2

        # UNet backbone for feature extraction
        self.unet_backbone = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_channels,
            dropout=dropout,
            channels=channels,
            strides=strides,
        )

        self.coord_head = nn.Sequential(
            nn.Conv2d(coord_feature_dim, coord_feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_feature_dim, coord_feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_feature_dim, out_coords, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x: Input image tensor of shape (B, 1, H, W)

        Returns
        -------
        Output tensor of shape (B, 4, H, W) where:
            - [:, :3, :, :] are the predicted coordinates (ML, AP, DV)
            - [:, 3:, :, :] is the tissue mask logits
        """
        B, C, H, W = x.shape

        # Concatenate with positional encoding if enabled
        if self.use_positional_encoding:
            pos_encoding = self.positional_encoding(batch_size=B, H=H, W=W)
            x = torch.cat([x, pos_encoding], dim=1)

        # Extract features from UNet backbone
        features = self.unet_backbone(x)

        if self.include_tissue_mask:
            coord_features, mask_logits = features[:, :-1], features[:, -1].unsqueeze(1)
        else:
            coord_features, mask_logits = features, None

        # Predict coordinates using regression head
        coords = self.coord_head(coord_features)

        # Predict tissue mask using classification head
        if self.include_tissue_mask:
            output = torch.cat([coords, mask_logits], dim=1)
        else:
            output = coords

        return output
