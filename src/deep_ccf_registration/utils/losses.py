import torch



def calc_multi_task_loss(
    point_loss: torch.Tensor,
    tissue_mask_loss: torch.Tensor,
    tissue_mask_loss_weight: float = 1.0
):
    """
    :param model:
    :param point_loss:
    :param tissue_mask_loss:
    :return:
    """
    loss = point_loss + tissue_mask_loss_weight * tissue_mask_loss
    return loss
