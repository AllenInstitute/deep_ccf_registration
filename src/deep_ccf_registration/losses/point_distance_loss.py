import torch
from torch import nn


class PointMSELoss(nn.Module):
    """
    Computes mean squared Euclidean distance between predicted and target points.

    For coordinate tensors of shape (B, C, H, W) where C is the coordinate dimension,
    this computes the squared L2 distance for each spatial point and returns the mean.

    This is preferred over PointDistanceLoss for training since MSE has smoother
    gradients. Use sqrt of this loss for RMSE logging.
    """

    def __init__(self, coordinate_dim: int = 1):
        """
        Parameters
        ----------
        coordinate_dim : int
            The dimension containing coordinates (default=1 for shape B, C, H, W)
        """
        super().__init__()
        self.coordinate_dim = coordinate_dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared point distance loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted coordinates, shape (B, C, H, W)
        target : torch.Tensor
            Target coordinates, shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Scalar mean squared Euclidean distance
        """
        squared_errors = (pred - target) ** 2
        per_point_squared_distance = squared_errors.sum(dim=self.coordinate_dim)
        return per_point_squared_distance.mean()
