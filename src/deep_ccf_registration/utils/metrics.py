from typing import Optional

import numpy as np
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


class SparseDiceMetric:
    """Custom dice metric since other implementations construct dense one-hot encoding
    which blows up memory when there are 1k+ classes"""
    def __init__(self, class_ids: np.ndarray):
        self.num_classes = len(class_ids)
        self._sample_scores = []
        self._label_to_idx = {label: i for i, label in enumerate(class_ids)}


    def _remap(self, arr: np.ndarray) -> np.ndarray:
        """Mapping the noncontiguous ccf ids to contiguous ones"""
        remapped = np.zeros_like(arr)
        for label, idx in self._label_to_idx.items():
            remapped[arr == label] = idx
        return remapped

    def update(self, pred: np.ndarray, target: np.ndarray):
        pred = self._remap(arr=pred)
        target = self._remap(arr=target)

        match_mask  = pred == target

        intersection = np.bincount(target[match_mask].astype('int'), minlength=self.num_classes)
        pred_count   = np.bincount(pred.astype('int'),               minlength=self.num_classes)
        target_count = np.bincount(target.astype('int'),             minlength=self.num_classes)

        denom = pred_count + target_count
        dice = np.where(denom > 0, 2.0 * intersection / (denom+1e-9), np.nan)
        self._sample_scores.append(dice)

    def per_class(self) -> np.ndarray:
        """Mean Dice per class, averaged over samples. Shape: (C,)"""
        stacked = np.stack(self._sample_scores)  # (N_samples, C)
        return np.nanmean(stacked, axis=0)

    def compute(self) -> float:
        """Mean Dice averaged over classes and samples."""
        return float(np.nanmean(self.per_class()))