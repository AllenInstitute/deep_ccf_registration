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
                masks: torch.Tensor,
                ) -> torch.Tensor:
        device = pred_template_points.device
        sagittal_mask = torch.tensor(
            [o == SliceOrientation.SAGITTAL for o in orientations],
            device=device, dtype=torch.bool
        )

        standard_error = torch.sum((pred_template_points - true_template_points) ** 2,
                                       dim=1)

        hemisphere_agnostic_err = self._calc_hemisphere_agnostic_error(
            pred_template_points,
            true_template_points,
            error_direct=standard_error,
        )

        weight_mask = masks + self._lambda_background * (1 - masks)

        weight_sum = weight_mask.sum(dim=(1, 2)).clamp(min=1)
        standard_loss = (standard_error * weight_mask).sum(dim=(1, 2)) / weight_sum
        hemisphere_agnostic_loss = (hemisphere_agnostic_err * weight_mask).sum(
            dim=(1, 2)) / weight_sum

        loss_per_sample = torch.where(sagittal_mask, hemisphere_agnostic_loss, standard_loss)

        return loss_per_sample.mean()

    def _calc_hemisphere_agnostic_error(
            self,
            pred_template_points: torch.Tensor,
            true_template_points: torch.Tensor,
            error_direct: torch.Tensor,
    ) -> torch.Tensor:
        ml_flipped = mirror_points(points=pred_template_points, template_parameters=self._template_parameters, ml_dim_size=self._ml_dim_size)

        err_flipped = torch.sum((ml_flipped - true_template_points) ** 2,
                                dim=1)

        return torch.minimum(error_direct, err_flipped)

def mirror_points(points: torch.Tensor, template_parameters: AntsImageParameters, ml_dim_size: int):
    flipped = points.clone()

    # 1. Convert to index space
    for dim in range(template_parameters.dims):
        flipped[:, dim] -= template_parameters.origin[dim]
        flipped[:, dim] *= template_parameters.direction[dim]
        flipped[:, dim] /= template_parameters.scale[dim]

    # 2. Flip ML in index space
    flipped[:, 0] = ml_dim_size-1 - flipped[:, 0]

    # 3. Convert back to physical
    for dim in range(template_parameters.dims):
        flipped[:, dim] *= template_parameters.scale[dim]
        flipped[:, dim] *= template_parameters.direction[dim]
        flipped[:, dim] += template_parameters.origin[dim]

    return flipped
