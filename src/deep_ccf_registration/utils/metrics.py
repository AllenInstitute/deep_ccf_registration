from typing import Optional

import torch
from torch import nn


class MSE(nn.Module):
    """
    Computes root mean squared Euclidean distance between predicted and target points.
    """

    def __init__(self, coordinate_dim: int = 1, reduction: Optional[str] = None):
        """
        Parameters
        ----------
        coordinate_dim : int
            The dimension containing coordinates (default=1 for shape B, C, H, W)
        """
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self._reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted coordinates, shape (B, C, H, W)
        target : torch.Tensor
            Target coordinates, shape (B, C, H, W)
        """
        squared_errors = (pred - target) ** 2
        per_point_squared_distance = squared_errors.sum(dim=self.coordinate_dim)

        if mask is not None:
            if mask.dim() == per_point_squared_distance.dim() + 1:
                mask = mask.sum(dim=self.coordinate_dim)
            if mask.dim() != per_point_squared_distance.dim():
                raise ValueError("Mask must have same spatial dimensions as coordinates")

            # Cast to float32 for numerically stable reduction (important for mixed precision)
            mask = mask.to(per_point_squared_distance.device).float()
            squared_errors = per_point_squared_distance.float() * mask
            valid_points = mask.sum(dim=(1, 2)).clamp(min=1.0)
            mse = squared_errors.sum(dim=(1, 2)) / valid_points
        else:
            mse = per_point_squared_distance.float().mean(dim=(1, 2))

        if self._reduction == 'mean':
            mse = mse.mean()
        return mse
