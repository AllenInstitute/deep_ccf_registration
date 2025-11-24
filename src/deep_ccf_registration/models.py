import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNetWithRegressionHeads(nn.Module):
    """
    UNet backbone with separate regression heads for coordinates and classification head for tissue mask.

    This architecture uses UNet for feature extraction and then separate task-specific heads:
    - Coordinate regression head: predicts 3D template coordinates
    - Tissue mask classification head: predicts tissue presence (binary)
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
        """
        super().__init__()

        self.out_coords = out_coords
        self.include_tissue_mask = include_tissue_mask

        # UNet backbone for feature extraction
        self.unet_backbone = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_channels,
            dropout=dropout,
            channels=channels,
            strides=strides,
        )

        # Coordinate regression head
        if head_size == "small":
            self.coord_head = nn.Sequential(
                nn.Conv2d(feature_channels, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, out_coords, kernel_size=1),
            )
        elif head_size == "medium":
            self.coord_head = nn.Sequential(
                nn.Conv2d(feature_channels, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 48, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_coords, kernel_size=1),
            )
        elif head_size == "large":
            self.coord_head = nn.Sequential(
                nn.Conv2d(feature_channels, 128, kernel_size=1),
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
                nn.Conv2d(feature_channels, 32, kernel_size=1),
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
        # Extract features from UNet backbone
        features = self.unet_backbone(x)

        # Predict coordinates using regression head
        coords = self.coord_head(features)

        # Predict tissue mask using classification head
        if self.include_tissue_mask:
            mask = self.mask_head(features)
            output = torch.cat([coords, mask], dim=1)
        else:
            output = coords

        return output
