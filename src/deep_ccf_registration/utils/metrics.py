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
    def __init__(
        self,
        class_ids: np.ndarray,
        terminology_path: Path,
        terminology_correction_path: Path,
        exclude_background: bool = True,
    ):
        """

        :param class_ids:
        :param terminology_path:
        :param terminology_correction_path: Due to unknown issue, some ids in annotation volume
            are not in terminology. Ashwin Bhandiwad created this json file which maps these
            missing ids
        :param exclude_background:
        """
        self._terminology = pd.read_csv(terminology_path).set_index('annotation_value')
        with open(terminology_correction_path) as f:
            self._terminology_correction = json.load(f)
        class_ids = [int(x) for x in class_ids]
        if exclude_background:
            class_ids = [x for x in class_ids if x != 0]
        self._class_ids = class_ids

        all_children = self._get_all_ids()

        self._annotation_id_to_idx = {label: i + 1 if exclude_background else i for i, label in enumerate(sorted(all_children))}
        self.num_classes = len(all_children) + 1 if exclude_background else len(all_children)

        self._parent_to_children_ids = self._build_parent_to_children_map()
        self._total_intersection = np.zeros(self.num_classes, dtype=np.int64)
        self._total_pred_count = np.zeros(self.num_classes, dtype=np.int64)
        self._total_target_count = np.zeros(self.num_classes, dtype=np.int64)

    def _build_parent_to_children_map(self):
        parent_to_children_idx = defaultdict(list)
        for parent in self._class_ids:
            entries = self._get_terminology_entry_for_id(id=parent)
            for entry in entries:
                children = ast.literal_eval(entry['descendant_annotation_values'])
                parent_to_children_idx[parent] += [self._annotation_id_to_idx[c] for c in children]
            parent_to_children_idx[parent] = list(set(parent_to_children_idx[parent]))
        return parent_to_children_idx

    def _get_all_ids(self):
        """
        For given class_ids, find all child ids

        :return:
        """
        all_children = set()
        for parent in self._class_ids:
            entries = self._get_terminology_entry_for_id(id=parent)
            for entry in entries:
                children = ast.literal_eval(entry['descendant_annotation_values'])
                all_children.update(children)
        return all_children

    def _get_terminology_entry_for_id(self, id: int) -> list[pd.Series]:
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

    def _remap(self, arr: np.ndarray) -> np.ndarray:
        remapped = np.zeros_like(arr)
        for label, idx in self._annotation_id_to_idx.items():
            remapped[arr == label] = idx
        return remapped

    def update(self, pred: np.ndarray, target: np.ndarray):
        pred = self._remap(arr=pred)
        target = self._remap(arr=target)

        match_mask = pred == target
        self._total_intersection += np.bincount(target[match_mask].astype('int'), minlength=self.num_classes)
        self._total_pred_count += np.bincount(pred.astype('int'), minlength=self.num_classes)
        self._total_target_count += np.bincount(target.astype('int'), minlength=self.num_classes)

    def compute(self, reduce: Optional[str] = 'mean') -> float:
        dice = np.empty(len(self._class_ids))
        for i, parent in enumerate(self._class_ids):
            idx = self._parent_to_children_ids[parent]
            inter = self._total_intersection[idx].sum()
            pred = self._total_pred_count[idx].sum()
            tgt = self._total_target_count[idx].sum()
            denom = pred + tgt
            dice[i] = 2.0 * inter / denom if denom > 0 else np.nan

        if reduce == 'mean':
            res = float(np.nanmean(dice))
        elif reduce is None:
            res = dice
        else:
            raise ValueError(f'{reduce} not supported')
        return res
