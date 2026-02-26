from typing import Optional

import numpy as np
import torch
from torch import nn

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import mirror_points
from deep_ccf_registration.metadata import SliceOrientation


def _calc_lowest_err_sagittal_orientation(
    pred: torch.Tensor, target: torch.Tensor,
    template_parameters: TemplateParameters,
    orientations: list[str],
) -> torch.Tensor:
    """Return pred or flipped pred, whichever is closer to target.
    """
    flipped_pred = mirror_points(points=pred, template_parameters=template_parameters)
    orig_mse = ((pred - target) ** 2).sum(dim=1).mean(dim=(1, 2))
    flip_mse = ((flipped_pred - target) ** 2).sum(dim=1).mean(dim=(1, 2))
    sagittal_mask = torch.tensor(
        [SliceOrientation(o) == SliceOrientation.SAGITTAL for o in orientations],
        device=pred.device, dtype=torch.bool
    )
    use_flipped = (flip_mse < orig_mse) & sagittal_mask
    pred = torch.where(use_flipped[:, None, None, None], flipped_pred, pred)
    return pred


class PerAxisError(nn.Module):
    def __init__(self,
                 template_parameters: TemplateParameters,
                 coordinate_dim: int = 1,
                 ):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self._template_parameters = template_parameters

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                orientations: list[str],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert len(pred.shape) == 4 and pred.shape[1] == 3
        assert len(target.shape) == 4 and target.shape[1] == 3
        if mask is not None:
            assert len(mask.shape) == 3 and list(mask.shape) == [pred.shape[0]] + list(pred.shape[-2:])

        pred = _calc_lowest_err_sagittal_orientation(
            pred=pred, target=target, template_parameters=self._template_parameters,
            orientations=orientations,
        )
        squared_errors = (pred - target) ** 2
        return squared_errors


class SparseDiceMetric:
    """Custom dice metric since other implementations construct dense one-hot encoding
    which blows up memory when there are 1k+ classes"""
    def __init__(self, class_ids: np.ndarray):
        self.num_classes = len(class_ids)
        self._sample_scores: list[np.ndarray] = []
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

    def compute_for_sample_idx(self, idx: int) -> float:
        return float(np.nanmean(self._sample_scores[idx]))