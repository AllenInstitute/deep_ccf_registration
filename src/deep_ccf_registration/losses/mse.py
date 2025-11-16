import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space, \
    convert_to_ants_space
from torch import nn as nn

from deep_ccf_registration.metadata import SliceOrientation


class HemisphereAgnosticMSE(nn.Module):
    """
    Because of the medial-lateral symmetry of the brain, a sagittal slice sampled from the left
    hemisphere looks identical to a sagittal slice sampled from the right if it is equidistant
    from a central point.

    If the orientation is sagittal, this loss will ignore whether the predicted ML-axis position
    is in the left or right hemisphere.
    Otherwise, it will calculate standard MSE.
    """

    def __init__(
            self, ml_dim_size: float,
            template_parameters: AntsImageParameters,
            lambda_background: float = 0.0
    ):
        """
        :param ml_dim_size: medial-lateral axis dimension in template index space
        :param lambda_background: amount to weight background pixels in loss function
        """
        super().__init__()
        self._ml_dim_size = ml_dim_size
        self._template_parameters = template_parameters
        self._lambda_background = lambda_background

    def forward(self,
                pred_template_points: torch.Tensor,
                true_template_points: torch.Tensor,
                orientations: list[SliceOrientation],
                tissue_masks: torch.Tensor,
                return_mean: bool = True
                ) -> torch.Tensor:
        """
        :param pred_template_points: Predicted points in light sheet template physical space, shape (batch_size, 3, H, W) (RAS)
        :param true_template_points: Ground truth points in light sheet template physical space, shape (batch_size, 3, H, W) (RAS)
        :param orientations: Orientation for each sample, shape (batch_size,)
        :param tissue_masks: tissue mask shape (batch_size, H, W)
        :return: Mean squared error loss
        """
        device = pred_template_points.device
        sagittal_mask = torch.tensor(
            [o == SliceOrientation.SAGITTAL for o in orientations],
            device=device, dtype=torch.bool
        )

        # Precompute squared errors - shape: (batch_size, H, W)
        standard_se = torch.sum((pred_template_points - true_template_points) ** 2, dim=1)
        hemi_agnostic_se = self._calc_hemisphere_agnostic_se(pred_template_points,
                                                             true_template_points)

        # Weight mask: tissue pixels = 1.0, background pixels = lambda_background
        weight_mask = tissue_masks + self._lambda_background * (1 - tissue_masks)

        if return_mean:
            # Only consider tissue pixels
            weight_sum = weight_mask.sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
            standard_loss = (standard_se * weight_mask).sum(dim=(1, 2)) / weight_sum
            hemi_loss = (hemi_agnostic_se * weight_mask).sum(dim=(1, 2)) / weight_sum
        else:
            standard_loss = standard_se * weight_mask
            hemi_loss = hemi_agnostic_se * weight_mask

        # Choose hemisphere-agnostic or standard depending on orientation
        loss_per_sample = torch.where(sagittal_mask, hemi_loss, standard_loss)

        return loss_per_sample.mean() if return_mean else loss_per_sample

    def _calc_hemisphere_agnostic_se(
            self,
            pred_template_points: torch.Tensor,
            true_template_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate hemisphere-agnostic squared error for sagittal slices.

        :param pred_template_points: Predicted points in template physical space, shape (batch_size, 3, H, W)
        :param true_template_points: Ground truth points in template physical space, shape (batch_size, 3, H, W)
        :return: Squared errors, shape (batch_size, H, W)
        """
        # Compute total SE for direct prediction
        se_direct = torch.sum((pred_template_points - true_template_points) ** 2, dim=1)

        # Compute total SE for hemisphere-flipped prediction
        ml_flipped = self._mirror_points(pred=pred_template_points)
        se_flipped = torch.sum((ml_flipped - true_template_points) ** 2, dim=1)

        # Take minimum of the two at each pixel
        return torch.minimum(se_direct, se_flipped)

    def _mirror_points(self, pred: torch.Tensor):
        # 1. Convert to index space
        flipped = pred.clone()

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
