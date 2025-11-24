import torch
import torch.nn as nn
from monai.networks.nets import UNet
from typing import Optional


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
        include_tissue_mask: bool = True,
        head_size: str = "small",
        use_positional_encoding: bool = False,
        pos_encoding_channels: int = 16,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
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

        # UNet backbone for feature extraction
        self.unet_backbone = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_channels,
            dropout=dropout,
            channels=channels,
            strides=strides,
        )

        # Positional encoding (learned embeddings for each spatial location)
        if use_positional_encoding:
            if image_height is None or image_width is None:
                raise ValueError(
                    "image_height and image_width must be specified when use_positional_encoding=True"
                )
            self.pos_embedding = nn.Parameter(
                torch.randn(1, pos_encoding_channels, image_height, image_width)
            )
            input_channels = feature_channels + pos_encoding_channels
        else:
            input_channels = feature_channels

        # Coordinate regression head
        if head_size == "small":
            self.coord_head = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, out_coords, kernel_size=1),
            )
        elif head_size == "medium":
            self.coord_head = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 48, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_coords, kernel_size=1),
            )
        elif head_size == "large":
            self.coord_head = nn.Sequential(
                nn.Conv2d(input_channels, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 96, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_coords, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown head_size: {head_size}")

        # Tissue mask classification head
        if include_tissue_mask:
            self.mask_head = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
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

        # Extract features from UNet backbone
        features = self.unet_backbone(x)

        # Concatenate with positional encoding if enabled
        if self.use_positional_encoding:
            pos_encoding = self.pos_embedding.expand(B, -1, -1, -1)
            features = torch.cat([features, pos_encoding], dim=1)

        # Predict coordinates using regression head
        coords = self.coord_head(features)

        # Predict tissue mask using classification head
        if self.include_tissue_mask:
            mask = self.mask_head(features)
            output = torch.cat([coords, mask], dim=1)
        else:
            output = coords

        return output
