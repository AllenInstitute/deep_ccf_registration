from enum import Enum
from typing import Optional

import monai.networks.nets
import torch
import torch.nn as nn


class CoordConv(nn.Module):
    """from An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, Liu et al"""

    def __init__(self):
        super().__init__()

    def forward(self, batch_size: int, H: int, W: int, device):
        coords = torch.stack(
            torch.meshgrid(torch.arange(H, device=device),
                           torch.arange(W, device=device),
                           indexing='ij'),
            dim=0
        ).float()  # (2, H, W)
        coords[0] /= (H - 1)
        coords[1] /= (W - 1)

        return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)


class PositionalEmbeddingType(Enum):
    LEARNED = 'LEARNED'
    COORD_CONV = 'COORD_CONV'

class PositionalEmbeddingPlacement(Enum):
    EARLY = 'EARLY'
    LATE = 'LATE'

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
        input_dims: tuple[int, int],
        coord_head_channels: tuple[int, ...],
        spatial_dims: int = 2,
        in_channels: int = 1,
        feature_channels: int = 64,
        dropout: float = 0.0,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        strides: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        out_coords: int = 3,
        include_tissue_mask: bool = False,
        use_positional_encoding: bool = False,
        pos_encoding_channels: Optional[int] = None,
        positional_embedding_type: Optional[PositionalEmbeddingType] = None,
        positional_embedding_placement: Optional[PositionalEmbeddingPlacement] = None,
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

        if use_positional_encoding and positional_embedding_type is None:
            raise ValueError('provide positional_embedding_type')
        if use_positional_encoding and positional_embedding_placement is None:
            raise ValueError('provide positional_embedding_placement')
        self.out_coords = out_coords
        self.include_tissue_mask = include_tissue_mask
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_channels = pos_encoding_channels
        self._positional_encoding_type = positional_embedding_type
        self._positional_embedding_placement = positional_embedding_placement

        coord_feature_channels = feature_channels
        tissue_mask_channels = 1 if include_tissue_mask else 0
        coord_head_input_channels = coord_feature_channels

        if self.use_positional_encoding:
            if positional_embedding_type == PositionalEmbeddingType.COORD_CONV:
                self._coord_conv_positional_encoding = CoordConv()

                if positional_embedding_placement == PositionalEmbeddingPlacement.EARLY:
                    in_channels += 2
                else:
                    coord_head_input_channels += 2
            else:
                if pos_encoding_channels is None:
                    raise ValueError('provide pos_encoding_channels')
                self._positional_embedding = nn.Parameter(
                    torch.randn(1, pos_encoding_channels, input_dims[0], input_dims[1])
                )
                if positional_embedding_placement == PositionalEmbeddingPlacement.EARLY:
                    in_channels += pos_encoding_channels
                else:
                    coord_head_input_channels += pos_encoding_channels

        # UNet backbone for feature extraction
        self.unet_backbone = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=coord_feature_channels+tissue_mask_channels,
            dropout=dropout,
            channels=channels,
            strides=strides,
        )

        layers = []
        in_ch = coord_head_input_channels
        for out_ch in coord_head_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(p=0.1))
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, out_coords, kernel_size=1))
        self.coord_head = nn.Sequential(*layers)

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

        if self.use_positional_encoding:
            if self._positional_encoding_type == PositionalEmbeddingType.COORD_CONV:
                pos_encoding = self._coord_conv_positional_encoding(batch_size=B, H=H, W=W,
                                                                    device=x.device)
            else:
                pos_encoding = self._positional_embedding.expand(B, -1, -1, -1)

            if self._positional_embedding_placement == PositionalEmbeddingPlacement.EARLY:
                x = torch.cat([x, pos_encoding], dim=1)
        else:
            pos_encoding = None

        # Extract features from UNet backbone
        features = self.unet_backbone(x)

        if self.include_tissue_mask:
            coord_features, mask_logits = features[:, :-1], features[:, -1].unsqueeze(1)
        else:
            coord_features, mask_logits = features, None

        if self.use_positional_encoding and self._positional_embedding_placement == PositionalEmbeddingPlacement.LATE:
            coord_features = torch.cat([coord_features, pos_encoding], dim=1)

        # Predict coordinates using regression head
        coords = self.coord_head(coord_features)

        # Predict tissue mask using classification head
        if self.include_tissue_mask:
            output = torch.cat([coords, mask_logits], dim=1)
        else:
            output = coords

        return output
