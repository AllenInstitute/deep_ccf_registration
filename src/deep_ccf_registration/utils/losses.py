import math
from typing import Optional

import torch



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
        if len(self._loss_history) != 2:
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

    def update(self, avg_point_loss: float, avg_tissue_mask_loss: Optional[float] = None):
        task_losses = [avg_point_loss]
        if avg_tissue_mask_loss is not None:
            task_losses.append(avg_tissue_mask_loss)
        assert len(task_losses) == self._K
        if len(self._loss_history) < 2:
            self._loss_history.append(task_losses)
        else:
            self._loss_history[0] = self._loss_history[1]
            self._loss_history[1] = task_losses

def calc_multi_task_loss(
    point_loss: torch.Tensor,
    dwa_scheduler: DynamicWeightAverageScheduler,
    tissue_mask_loss: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    :param model:
    :param point_loss:
    :param tissue_mask_loss:
    """
    weights = dwa_scheduler.get_weights()
    loss = weights[0] * point_loss
    if tissue_mask_loss is not None:
        loss += weights[1] * tissue_mask_loss
    return loss
