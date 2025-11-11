import torch
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

    def __init__(self, ml_dim_size: float):
        """
        :param ml_dim_size: medial-lateral axis dimension in light sheet template space.
            Must be in the same unit as `pred_template_points` and `true_template_points` passed to `self.forward`
        """
        super().__init__()
        self._ml_dim_size = ml_dim_size

    def forward(self,
                pred_template_points: torch.Tensor,
                true_template_points: torch.Tensor,
                orientations: list[SliceOrientation]
                ) -> torch.Tensor:
        """
        :param pred_template_points: Predicted points in light sheet template physical space, shape (batch_size, 3) (RAS)
        :param true_template_points: Ground truth points in light sheet template physical space, shape (batch_size, 3) (RAS)
        :param orientations: Orientation for each sample, shape (batch_size,)
        :return: Mean squared error loss
        """
        sagittal_mask = torch.tensor([orientations[i] == SliceOrientation.SAGITTAL for i in range(len(orientations))], device=pred_template_points.device).bool()

        # Calculate hemisphere-agnostic SE for all samples
        hemisphere_agnostic_se = self._calc_hemisphere_agnostic_se(
            pred_template_points,
            true_template_points
        )

        # Calculate standard SE for all samples
        standard_se = torch.sum((pred_template_points - true_template_points) ** 2, dim=1)

        # Use sagittal mask to select appropriate loss for each sample
        squared_errors = torch.where(sagittal_mask[:, None, None], hemisphere_agnostic_se, standard_se)

        # Return mean squared error
        return torch.mean(squared_errors)

    def _calc_hemisphere_agnostic_se(
            self,
            pred_template_points: torch.Tensor,
            true_template_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate hemisphere-agnostic squared error for sagittal slices.

        :param pred_template_points: Predicted points in light sheet template physical space, shape (batch_size, 3) (RAS)
        :param true_template_points: Ground truth points in light sheet template physical space, shape (batch_size, 3) (RAS)
        :return: Squared errors, shape (batch_size,)
        """
        # LS template is in orientation RAS
        # ML error: consider both direct and flipped across midline
        ml_error_direct = (pred_template_points[:, 0] - true_template_points[:, 0]) ** 2
        ml_error_flipped = ((self._ml_dim_size - pred_template_points[:, 0]) -
                            true_template_points[:, 0]) ** 2
        ml_error = torch.minimum(ml_error_direct, ml_error_flipped)

        # AP and DV errors are standard
        ap_error = (pred_template_points[:, 1] - true_template_points[:, 1]) ** 2
        dv_error = (pred_template_points[:, 2] - true_template_points[:, 2]) ** 2

        squared_errors = ml_error + ap_error + dv_error
        return squared_errors
