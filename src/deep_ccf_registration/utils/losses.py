import torch

from deep_ccf_registration.models import UNetWithRegressionHeads


def calc_multi_task_loss(model: UNetWithRegressionHeads, point_loss: torch.Tensor, tissue_mask_loss: torch.Tensor):
    """
    From from Kendall et al. 2018 to weight multi-task loss function with different scales

    :param model:
    :param point_loss:
    :param tissue_mask_loss:
    :return:
    """
    log_var_point = model.log_variance_point_loss
    log_var_seg = model.log_variance_tissue_segmentation_loss
    loss = (
            0.5 * torch.exp(-log_var_point) * point_loss + 0.5 * log_var_point +
            0.5 * torch.exp(-log_var_seg) * tissue_mask_loss + 0.5 * log_var_seg
    )
    return loss
