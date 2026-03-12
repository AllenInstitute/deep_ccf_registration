import math
from typing import Optional

import torch
from torch import nn

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.utils.metrics import calc_lowest_err_sagittal_orientation


class DynamicWeightAverageScheduler:
    """
    From "End-to-End Multi-Task Learning with Attention", Liu et al., 2019
    """
    def __init__(
        self,
        num_tasks: int,
        temperature: float = 2.0,
        w_init: Optional[list[float]] = None
    ):
        self._T = temperature
        self._K = num_tasks
        # index 0 = t-2, index 1 = t-1
        self._loss_history = []

        if w_init is None:
            w_init = [1.0 for _ in range(self._K)]
        else:
            if len(w_init) != num_tasks:
                raise ValueError(f'w_init must be length {num_tasks}')

        self._w_init = w_init

    def get_weights(self):
        if len(self._loss_history) < 2:
            w = self._w_init
        else:
            w = [self._loss_history[1][k] / (self._loss_history[0][k] + 1e-8)
                 for k in range(self._K)]
        exp_w = [math.exp(wk / self._T) for wk in w]
        sum_exp = sum(exp_w)
        return [self._K * e / sum_exp for e in exp_w]

    def state_dict(self) -> dict:
        return {
            'loss_history': [list(h) for h in self._loss_history],
        }

    def load_state_dict(self, state_dict: dict):
        self._loss_history = [list(h) for h in state_dict['loss_history']]

    def update(self, avg_point_loss: float, avg_spatial_gradient_loss: float, avg_tissue_mask_loss: Optional[float] = None):
        task_losses = [avg_point_loss, avg_spatial_gradient_loss]
        if avg_tissue_mask_loss is not None:
            task_losses.append(avg_tissue_mask_loss)
        assert len(task_losses) == self._K
        if len(self._loss_history) < 2:
            self._loss_history.append(task_losses)
        else:
            self._loss_history[0] = self._loss_history[1]
            self._loss_history[1] = task_losses

def calc_spatial_gradient_loss(
    # (B, 3, H, W)
    pred: torch.Tensor,
    mask: torch.Tensor,
):
    d2x = torch.diff(pred, n=2, dim=-1)  # (B, 3, H, W-2)
    d2y = torch.diff(pred, n=2, dim=-2)  # (B, 3, H-2, W)

    mask = mask.float()
    mask_x = mask[..., 1:-1].unsqueeze(1)   # (B, 1, H, W-2)
    mask_y = mask[:, 1:-1, :].unsqueeze(1)  # (B, 1, H-2, W)

    loss_x = (d2x ** 2 * mask_x).sum() / mask_x.sum().clamp(min=1.0)
    loss_y = (d2y ** 2 * mask_y).sum() / mask_y.sum().clamp(min=1.0)

    return loss_x + loss_y

def calc_multi_task_loss(
    point_loss: torch.Tensor,
    grad_loss: torch.Tensor,
    dwa_scheduler: DynamicWeightAverageScheduler,
    tissue_mask_loss: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weights = dwa_scheduler.get_weights()
    loss = weights[0] * point_loss + weights[1] * grad_loss
    if tissue_mask_loss is not None:
        loss += weights[2] * tissue_mask_loss
    return loss


class MSE(nn.Module):
    def __init__(self,
                 template_parameters: TemplateParameters,
                 coordinate_dim: int = 1,
                 reduction: Optional[str] = None,
                 ):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self._reduction = reduction
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
        per_point_squared_error = ((pred - target) ** 2).sum(dim=self.coordinate_dim)

        if mask is not None:
            mask = mask.to(per_point_squared_error.device).float()
            per_point_squared_error = per_point_squared_error.float() * mask
            valid_points = mask.sum(dim=(1, 2)).clamp(min=1.0)
            mse = per_point_squared_error.sum(dim=(1, 2)) / valid_points
        else:
            mse = per_point_squared_error.float().mean(dim=(1, 2))

        if self._reduction == 'mean':
            mse = mse.mean()
        return mse
