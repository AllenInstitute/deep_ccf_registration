import torch



def _calc_tissue_mask_weight(
    step_num: int,
    max_steps: int,
    alpha: float = 0.1,
    beta: float = 0.25,
):
    """decay over time since less important and easier"""
    beta *= max_steps
    if step_num <= beta:
        # linear decrease from step 0...beta starting from 1...alpha
        slope = (alpha - 1) / beta
        intercept = 1
        weight = slope * step_num + intercept
    else:
        # linear decrease from alpha...0
        weight = alpha * (max_steps - step_num) / (max_steps - beta)
    return weight

def calc_multi_task_loss(
    point_loss: torch.Tensor,
    tissue_mask_loss: torch.Tensor,
    step_num: int,
    max_steps: int,
) -> tuple[torch.Tensor, float]:
    """
    :param model:
    :param point_loss:
    :param tissue_mask_loss:
    :return: (loss, tissue_mask_loss_weight)
    """
    tissue_mask_loss_weight = _calc_tissue_mask_weight(step_num=step_num, max_steps=max_steps)
    loss = point_loss + tissue_mask_loss_weight * tissue_mask_loss
    return loss, tissue_mask_loss_weight
