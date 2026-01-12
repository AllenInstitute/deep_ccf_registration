from typing import Optional

import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

from deep_ccf_registration.datasets.transforms import TemplateParameters
from deep_ccf_registration.metrics.point_wise_rmse import PointwiseRMSE, PointwiseMAE
from deep_ccf_registration.utils.utils import fetch_complete_colormap, visualize_ccf_annotations


def create_diagnostic_image(
        input_image: torch.Tensor,
        pred_template_points: torch.Tensor,
        gt_template_points: torch.Tensor,
        pred_ccf_annotations: np.ndarray,
        gt_ccf_annotations: np.ndarray,
        errors: np.ndarray,
        pad_mask: np.ndarray,
        slice_idx: int,
        vmax_percentile: float = 95.0,
        pred_mask: Optional[np.ndarray] = None,
        tissue_mask: Optional[np.ndarray] = None,
        iteration: Optional[int] = None,
        exclude_background: bool = False
):
    """
    Visualize error heatmap, coordinate predictions, and ccf annotation predictions

    :param input_image: normalized input image (H, W)
    :param pred_template_points: predicted coordinates (3, H, W) in LS template space
    :param gt_template_points: ground truth coordinates (3, H, W) in LS template space
    :param pred_ccf_annotations: predicted CCF annotations (H, W)
    :param gt_ccf_annotations: ground truth CCF annotations (H, W)
    :param errors: squared error for all dims in ANTs space (millimeters²) shape (3, H, W)
    :param slice_idx: slice index for title
    :param vmax_percentile: percentile for colormap max
    :param iteration: optional iteration number
    :param pad_mask
    :param tissue_mask
    """
    if exclude_background:
        gt_mask = tissue_mask
    else:
        gt_mask = pad_mask
    abs_errors = np.sqrt(errors)
    abs_error_total = abs_errors.sum(axis=0)

    rmse = PointwiseRMSE(coordinate_dim=0)
    rmse = (rmse(
        preds=pred_template_points,
        target=gt_template_points,
        mask=gt_mask,
    ) * 1000).item()

    mae = PointwiseMAE(coordinate_dim=0)
    mae = (mae(
        preds=pred_template_points,
        target=gt_template_points,
        mask=gt_mask,
    ) * 1000).item()

    pred_template_points = pred_template_points.cpu().numpy()
    gt_template_points = gt_template_points.cpu().numpy()

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_error_ml = fig.add_subplot(gs[0, 1])
    ax_error_ap = fig.add_subplot(gs[0, 2])
    ax_error_dv = fig.add_subplot(gs[0, 3])

    ax_input.imshow(input_image, cmap='gray')
    ax_input.set_title('Input Image', fontsize=12)

    error_ml = abs_errors[0] * 1000
    error_ap = abs_errors[1] * 1000
    error_dv = abs_errors[2] * 1000

    error_ml[~gt_mask] = np.nan
    error_ap[~gt_mask] = np.nan
    error_dv[~gt_mask] = np.nan

    vmax = np.percentile(abs_error_total[gt_mask] * 1000, vmax_percentile)

    im_ml = ax_error_ml.imshow(error_ml, cmap='turbo', vmin=0, vmax=vmax)
    ax_error_ml.set_title('ML Error (micrometer)', fontsize=12)
    plt.colorbar(im_ml, ax=ax_error_ml, fraction=0.046, pad=0.04)

    im_ap = ax_error_ap.imshow(error_ap, cmap='turbo', vmin=0, vmax=vmax)
    ax_error_ap.set_title('AP Error (micrometer)', fontsize=12)
    plt.colorbar(im_ap, ax=ax_error_ap, fraction=0.046, pad=0.04)

    im_dv = ax_error_dv.imshow(error_dv, cmap='turbo', vmin=0, vmax=vmax)
    ax_error_dv.set_title('DV Error (micrometer)', fontsize=12)
    plt.colorbar(im_dv, ax=ax_error_dv, fraction=0.046, pad=0.04)

    ax_pred_ccf = fig.add_subplot(gs[1, 0:2])
    ax_gt_ccf = fig.add_subplot(gs[1, 2:4])

    colormap = fetch_complete_colormap()
    pred_ccf_vis = visualize_ccf_annotations(annotations=pred_ccf_annotations, colormap=colormap,
                                             return_image=True)
    gt_ccf_vis = visualize_ccf_annotations(annotations=gt_ccf_annotations, colormap=colormap,
                                           return_image=True)

    ax_pred_ccf.imshow(pred_ccf_vis)
    ax_pred_ccf.set_title('Predicted CCF annotations', fontsize=12)

    ax_gt_ccf.imshow(gt_ccf_vis)
    ax_gt_ccf.set_title('Ground truth CCF annotations', fontsize=12)

    for i in range(3):
        gt_template_points[i][~gt_mask] = np.nan
        pred_template_points[i][~gt_mask] = np.nan

    coord_labels = ['ML', 'AP', 'DV']
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(gt_template_points[i] * 1000, cmap='viridis')
        ax.set_title(f'GT {coord_labels[i]}', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_coord_error = fig.add_subplot(gs[2, 3])
    mean_coord_errors = [abs_errors[i][gt_mask].mean() * 1000 for i in range(3)]
    ax_coord_error.bar(['ML', 'AP', 'DV'], mean_coord_errors, color=['red', 'green', 'blue'])
    ax_coord_error.set_title('Mean Absolute Error per Coord (μm)', fontsize=12)
    ax_coord_error.set_ylabel('Error (microns)')
    ax_coord_error.grid(True, alpha=0.3)

    for i in range(3):
        ax = fig.add_subplot(gs[3, i])
        im = ax.imshow(pred_template_points[i] * 1000, cmap='viridis')
        ax.set_title(f'Pred {coord_labels[i]}', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if pred_mask is not None:
        ax_seg = fig.add_subplot(gs[3, 3])

        intersection = (pred_mask & tissue_mask).sum()
        dice = (2 * intersection) / (pred_mask.sum() + tissue_mask.sum() + 1e-8)

        ax_seg.imshow(input_image, cmap='gray')

        false_positive = pred_mask & ~tissue_mask
        false_negative = ~pred_mask & tissue_mask

        overlay = np.zeros((*tissue_mask.shape, 4))
        overlay[false_positive] = [1, 0, 0, 0.7]
        overlay[false_negative] = [0, 0, 1, 0.7]

        ax_seg.imshow(overlay)
        ax_seg.set_title(f'Segmentation Errors (Dice: {dice:.4f})\nRed=False Pos, Blue=False Neg',
                         fontsize=12)

    iteration_title = f'iter {iteration} ' if iteration else ''
    title = f'{iteration_title}slice: {slice_idx} MAE {mae:.3f} RMSE {rmse:.3f} microns'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig


def viz_sample(
    input_image: np.ndarray,
    predicted_template_points: np.ndarray,
    gt_template_points: np.ndarray,
    ls_template_info: TemplateParameters,
    mask: np.ndarray,
    template_parameters: AntsImageParameters,
):
    """
    Visualize predicted vs ground truth registration.
    """
    intensities = input_image.flatten()

    valid_mask = mask.flatten()
    intensities = intensities[valid_mask]
    predicted_template_points = predicted_template_points[valid_mask]
    gt_template_points = gt_template_points[valid_mask]

    pred_template_points_index_space = convert_from_ants_space(
        template_parameters=template_parameters,
        physical_pts=predicted_template_points
    )
    gt_template_points_index_space = convert_from_ants_space(
        template_parameters=template_parameters,
        physical_pts=gt_template_points
    )

    ML_axis, DV_axis, SI_axis = 0, 1, 2

    # Template grid for SI-DV plane
    si_grid = np.arange(ls_template_info.shape[SI_axis])
    dv_grid = np.arange(ls_template_info.shape[DV_axis])
    SI, DV = np.meshgrid(si_grid, dv_grid)

    def rasterize(pts, values=intensities):
        return griddata(pts[:, [SI_axis, DV_axis]], values, (SI, DV), method='linear')

    # Rasterize both predicted and GT
    predicted_raster = rasterize(pred_template_points_index_space)
    gt_raster = rasterize(gt_template_points_index_space)

    # Compute displacement/error
    displacement = predicted_template_points - gt_template_points
    displacement *= 1000  # convert to microns

    fig = plt.figure(figsize=(24, 10))

    # ===== Row 1: 2D Rasterized Images =====

    sns.set_style("darkgrid")

    # Ground truth registration
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(gt_raster, origin='lower', cmap='gray')
    ax1.set_title('Ground Truth Registration')
    ax1.set_xlabel('SI')
    ax1.set_ylabel('DV')

    # Predicted registration
    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(predicted_raster, origin='lower', cmap='gray')
    ax2.set_title('Predicted Registration')
    ax2.set_xlabel('SI')
    ax2.set_ylabel('DV')

    # Overlay comparison
    ax3 = fig.add_subplot(2, 5, 3)
    ax3.imshow(gt_raster, origin='lower', cmap='gray', alpha=0.7)
    ax3.imshow(predicted_raster, origin='lower', cmap='Reds', alpha=0.5)
    ax3.set_title('GT (gray) vs Predicted (red)')
    ax3.set_xlabel('SI')
    ax3.set_ylabel('DV')

    # SI displacement component
    ax5 = fig.add_subplot(2, 5, 4)
    si_error_image = rasterize(gt_template_points_index_space, values=displacement[:, SI_axis])
    im = ax5.imshow(si_error_image, origin='lower', cmap='RdBu_r',
                    vmin=-np.percentile(np.abs(displacement[:, SI_axis]), 95),
                    vmax=np.percentile(np.abs(displacement[:, SI_axis]), 95))
    ax5.set_title('SI Error')
    ax5.set_xlabel('SI')
    ax5.set_ylabel('DV')
    plt.colorbar(im, ax=ax5, label='SI (microns)')

    # DV displacement component
    ax6 = fig.add_subplot(2, 5, 5)
    dv_error_image = rasterize(gt_template_points_index_space, values=displacement[:, DV_axis])
    im = ax6.imshow(dv_error_image, origin='lower', cmap='RdBu_r',
                    vmin=-np.percentile(np.abs(displacement[:, DV_axis]), 95),
                    vmax=np.percentile(np.abs(displacement[:, DV_axis]), 95))
    ax6.set_title('DV Error')
    ax6.set_xlabel('SI')
    ax6.set_ylabel('DV')
    plt.colorbar(im, ax=ax6, label='DV (microns)')

    # ===== Row 2: 3D Visualizations =====

    step = 10
    idx = np.arange(0, len(intensities), step)

    # Use template shape for axis limits
    ml_lim = [0, ls_template_info.shape[ML_axis]]
    si_lim = [0, ls_template_info.shape[SI_axis]]
    dv_lim = [0, ls_template_info.shape[DV_axis]]

    # Ground truth 3D
    ax7 = fig.add_subplot(2, 5, 6, projection='3d')
    ax7.scatter(gt_template_points_index_space[idx, ML_axis],
                gt_template_points_index_space[idx, SI_axis],
                gt_template_points_index_space[idx, DV_axis],
                c=intensities[idx], cmap='gray', s=0.5)
    ax7.view_init(elev=20, azim=45)
    ax7.set_xlim(ml_lim)
    ax7.set_ylim(si_lim)
    ax7.set_zlim(dv_lim)
    ax7.set_xlabel('ML')
    ax7.set_ylabel('SI')
    ax7.set_zlabel('DV')
    ax7.set_title('Ground Truth 3D')
    ax7.set_box_aspect([ls_template_info.shape[ML_axis],
                        ls_template_info.shape[SI_axis],
                        ls_template_info.shape[DV_axis]])

    # Predicted 3D
    ax8 = fig.add_subplot(2, 5, 7, projection='3d')
    ax8.scatter(pred_template_points_index_space[idx, ML_axis],
                pred_template_points_index_space[idx, SI_axis],
                pred_template_points_index_space[idx, DV_axis],
                c=intensities[idx], cmap='gray', s=0.5)
    ax8.view_init(elev=20, azim=45)
    ax8.set_xlim(ml_lim)
    ax8.set_ylim(si_lim)
    ax8.set_zlim(dv_lim)
    ax8.set_xlabel('ML')
    ax8.set_ylabel('SI')
    ax8.set_zlabel('DV')
    ax8.set_title('Predicted 3D')
    ax8.set_box_aspect([ls_template_info.shape[ML_axis],
                        ls_template_info.shape[SI_axis],
                        ls_template_info.shape[DV_axis]])

    # ML displacement colored 3D
    ax10 = fig.add_subplot(2, 5, 8, projection='3d')
    ml_vmax = np.percentile(np.abs(displacement[:, ML_axis]), 95)
    sc = ax10.scatter(gt_template_points_index_space[idx, ML_axis],
                      gt_template_points_index_space[idx, SI_axis],
                      gt_template_points_index_space[idx, DV_axis],
                      c=displacement[idx, ML_axis],
                      cmap='RdBu_r', s=0.5, vmin=-ml_vmax, vmax=ml_vmax)
    ax10.view_init(elev=20, azim=45)
    ax10.set_xlim(ml_lim)
    ax10.set_ylim(si_lim)
    ax10.set_zlim(dv_lim)
    ax10.set_xlabel('ML')
    ax10.set_ylabel('SI')
    ax10.set_zlabel('DV')
    ax10.set_title('ML Error')
    ax10.set_box_aspect([ls_template_info.shape[ML_axis],
                         ls_template_info.shape[SI_axis],
                         ls_template_info.shape[DV_axis]])
    plt.colorbar(sc, ax=ax10, label='ML (microns)', shrink=0.5)

    plt.tight_layout()

    return fig
