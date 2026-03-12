import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import mirror_points
from deep_ccf_registration.metadata import SliceOrientation


def calc_lowest_err_sagittal_orientation(
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

        pred = calc_lowest_err_sagittal_orientation(
            pred=pred, target=target, template_parameters=self._template_parameters,
            orientations=orientations,
        )
        squared_errors = (pred - target) ** 2
        return squared_errors


class SparseDiceMetric:
    def __init__(
            self,
            class_ids: np.ndarray,
            terminology_path: Path,
            terminology_correction_path: Path,
            exclude_background: bool = True,
    ):
        self._terminology = pd.read_csv(terminology_path).set_index('annotation_value')
        with open(terminology_correction_path) as f:
            self._terminology_correction = json.load(f)
        class_ids = [int(x) for x in class_ids]
        if exclude_background:
            class_ids = [x for x in class_ids if x != 0]
        self._class_ids = class_ids

        self._parent_to_children_ids = self._build_parent_to_children_map()
        self._total_intersection = np.zeros(len(class_ids), dtype=np.int64)
        self._total_pred_count = np.zeros(len(class_ids), dtype=np.int64)
        self._total_target_count = np.zeros(len(class_ids), dtype=np.int64)

    def _build_parent_to_children_map(self):
        parent_to_children = defaultdict(set)
        for parent in self._class_ids:
            entries = self._get_terminology_entry_for_id(id=parent)
            for entry in entries:
                children = ast.literal_eval(entry['descendant_annotation_values'])
                parent_to_children[parent].update(children)
        return {k: np.array(list(v)) for k, v in parent_to_children.items()}

    def _get_terminology_entry_for_id(self, id: int) -> list[pd.Series]:
        """
        Note: this returns multiple due to bug in terminology, patched by _terminology_correction,
        otherwise it would be just 1

        :param id:
        :return:
        """
        try:
            entry = self._terminology.loc[id]
            entries = [entry]
        except KeyError:
            ids = self._terminology_correction[str(id)]['id']
            entries = []
            for id in ids:
                entry = self._terminology[self._terminology['identifier'] == id].iloc[0]
                entries.append(entry)
        return entries

    def update(self, pred: np.ndarray, target: np.ndarray):
        for i, parent in enumerate(self._class_ids):
            children = self._parent_to_children_ids[parent]
            pred_mask = np.isin(pred, children)
            target_mask = np.isin(target, children)
            self._total_intersection[i] += np.sum(pred_mask & target_mask)
            self._total_pred_count[i] += np.sum(pred_mask)
            self._total_target_count[i] += np.sum(target_mask)

    def compute(self, reduce: Optional[str] = 'mean'):
        denom = self._total_pred_count + self._total_target_count
        dice = np.where(denom > 0, 2.0 * self._total_intersection / denom, np.nan)

        if reduce == 'mean':
            return float(np.nanmean(dice))
        elif reduce is None:
            return dice
        else:
            raise ValueError(f'{reduce} not supported')
