import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from torch import nn as nn

from deep_ccf_registration.metadata import SliceOrientation


class HemisphereAgnosticCoordLoss(nn.Module):
    """
    MSE loss (L2) with hemisphere-agnostic handling for sagittal slices.
    """

    def __init__(
            self, ml_dim_size: float,
            template_parameters: AntsImageParameters,
            lambda_background: float = 0.0
    ):
        super().__init__()
        self._ml_dim_size = ml_dim_size
        self._template_parameters = template_parameters
        self._lambda_background = lambda_background

    def forward(self,
                pred_template_points: torch.Tensor,
                true_template_points: torch.Tensor,
                orientations: list[SliceOrientation],
                tissue_masks: torch.Tensor,
                per_channel_error: bool = False
                ) -> torch.Tensor:
        device = pred_template_points.device
        sagittal_mask = torch.tensor(
            [o == SliceOrientation.SAGITTAL for o in orientations],
            device=device, dtype=torch.bool
        )

        if per_channel_error:
            standard_error = (pred_template_points - true_template_points) ** 2
        else:
            standard_error = torch.sum((pred_template_points - true_template_points) ** 2,
                                       dim=1)

        hemisphere_agnostic_err = self._calc_hemisphere_agnostic_error(
            pred_template_points,
            true_template_points,
            per_channel=per_channel_error,
            error_direct=standard_error,
        )

        weight_mask = tissue_masks + self._lambda_background * (1 - tissue_masks)

        if per_channel_error:
            standard_loss = standard_error * weight_mask.unsqueeze(1)
            hemisphere_agnostic_loss = hemisphere_agnostic_err * weight_mask.unsqueeze(1)
        else:
            weight_sum = weight_mask.sum(dim=(1, 2)).clamp(min=1)
            standard_loss = (standard_error * weight_mask).sum(dim=(1, 2)) / weight_sum
            hemisphere_agnostic_loss = (hemisphere_agnostic_err * weight_mask).sum(
                dim=(1, 2)) / weight_sum

        loss_per_sample = torch.where(sagittal_mask, hemisphere_agnostic_loss, standard_loss)

        return loss_per_sample if per_channel_error else loss_per_sample.mean()

    def _calc_hemisphere_agnostic_error(
            self,
            pred_template_points: torch.Tensor,
            true_template_points: torch.Tensor,
            error_direct: torch.Tensor,
            per_channel=False,
    ) -> torch.Tensor:
        ml_flipped = self._mirror_points(pred=pred_template_points)

        if per_channel:
            err_flipped = (ml_flipped - true_template_points) ** 2
        else:
            err_flipped = torch.sum((ml_flipped - true_template_points) ** 2,
                                    dim=1)

        return torch.minimum(error_direct, err_flipped)

    def _mirror_points(self, pred: torch.Tensor):
        flipped = pred.clone()

        # 1. Convert to index space
        for dim in range(self._template_parameters.dims):
            flipped[:, dim] -= self._template_parameters.origin[dim]
            flipped[:, dim] *= self._template_parameters.direction[dim]
            flipped[:, dim] /= self._template_parameters.scale[dim]

        # 2. Flip ML in index space
        flipped[..., 0] = self._ml_dim_size-1 - flipped[..., 0]

        # 3. Convert back to physical
        for dim in range(self._template_parameters.dims):
            flipped[:, dim] *= self._template_parameters.scale[dim]
            flipped[:, dim] *= self._template_parameters.direction[dim]
            flipped[:, dim] += self._template_parameters.origin[dim]

        return flipped
