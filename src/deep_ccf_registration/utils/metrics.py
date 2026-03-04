import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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
    which blows up memory when there are 1k+ classes.

    If class_ids are not leaf nodes, then all children are pulled and dice is calculated
    aggregated for the parent
    """
    def __init__(
        self,
        class_ids: np.ndarray,
        terminology_path: Path,
        exclude_background: bool = True,
    ):
        self._exclude_background = exclude_background
        if exclude_background:
            self._label_to_idx = {label: i + 1 for i, label in enumerate(class_ids)}
            self._idx_to_label = {i+1: label for i, label in enumerate(class_ids)}
            self.num_classes = len(class_ids) + 1
        else:
            self._label_to_idx = {label: i for i, label in enumerate(class_ids)}
            self._idx_to_label = {i: label for i, label in enumerate(class_ids)}
            self.num_classes = len(class_ids)
        self._sample_scores: list[np.ndarray] = []

        self._class_ids = class_ids
        self._child_to_parent = self._construct_child_id_to_parent(terminology_path=terminology_path)

    def _construct_child_id_to_parent(self, terminology_path: Path) -> dict[int, int]:
        """
        Returns child, parent mapping

        :param terminology_path:
        :return:
        """
        terminology = pd.read_csv(terminology_path)
        annotation_descendents = terminology[['annotation_value', 'descendant_annotation_values']].copy()
        annotation_descendents['descendant_annotation_values'] = annotation_descendents[
            'descendant_annotation_values'].apply(lambda x: ast.literal_eval(x))

        child_to_parent = {}
        for _, row in annotation_descendents.iterrows():
            parent = row['annotation_value']
            # only map to a parent node if it's one of the nodes we care about in self._class_ids
            if parent in self._class_ids:
                for child in row['descendant_annotation_values']:
                    child_to_parent[child] = parent
                child_to_parent[parent] = parent
        return child_to_parent

    def _remap(self, arr: np.ndarray) -> np.ndarray:
        """Mapping the noncontiguous ccf ids to contiguous ones, and maps children to their parent node ids"""
        remapped = np.zeros_like(arr)
        for child, parent in self._child_to_parent.items():
            if parent in self._label_to_idx:
                remapped[arr == child] = self._label_to_idx[parent]
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
        if self._exclude_background:
            dice = dice[1:]
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