from typing import Optional

import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from matplotlib import pyplot as plt

from deep_ccf_registration.metrics.point_wise_rmse import PointwiseRMSE, PointwiseMAE
from deep_ccf_registration.utils.transforms import convert_from_ants_space_tensor
from deep_ccf_registration.utils.utils import fetch_complete_colormap, visualize_ccf_annotations, \
    get_ccf_annotations


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
    ls_template_parameters: AntsImageParameters,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    ccf_annotations: np.ndarray,
    iteration: int,
    pad_mask: torch.Tensor,
    input_image: torch.Tensor,
    slice_idx: int,
    errors: np.ndarray,
    pred_tissue_mask: Optional[torch.Tensor] = None,
    tissue_mask: Optional[torch.Tensor] = None,
    exclude_background: bool = False
):
    pred_index_space = convert_from_ants_space_tensor(template_parameters=ls_template_parameters,
                                                      physical_pts=pred_coords.permute(
                                                          (1, 2, 0)).reshape((-1, 3)))
    gt_index_space = convert_from_ants_space_tensor(template_parameters=ls_template_parameters,
                                                    physical_pts=gt_coords.permute(
                                                        (1, 2, 0)).reshape((-1, 3)))

    pred_ccf_annot = get_ccf_annotations(ccf_annotations, pred_index_space,
                                         return_np=False).reshape(
        pred_coords.shape[1:])
    if exclude_background:
        pred_ccf_annot[~tissue_mask] = 0
    else:
        pred_ccf_annot[~pad_mask] = 0
    gt_ccf_annot = get_ccf_annotations(ccf_annotations, gt_index_space, return_np=False).reshape(
        gt_coords.shape[1:])

    if exclude_background:
        gt_ccf_annot[~tissue_mask] = 0
    else:
        gt_ccf_annot[~pad_mask] = 0


    fig = create_diagnostic_image(
        input_image=input_image,
        slice_idx=slice_idx,
        errors=errors,
        pred_ccf_annotations=pred_ccf_annot.cpu().numpy(),
        gt_ccf_annotations=gt_ccf_annot.cpu().numpy(),
        iteration=iteration,
        pred_template_points=pred_coords,
        gt_template_points=gt_coords,
        pad_mask=pad_mask.cpu().bool().numpy(),
        exclude_background=exclude_background,
        tissue_mask=tissue_mask.cpu().bool().numpy() if tissue_mask is not None else None,
        pred_mask=pred_tissue_mask.cpu().bool().numpy() if pred_tissue_mask is not None else None,
    )
    return fig
