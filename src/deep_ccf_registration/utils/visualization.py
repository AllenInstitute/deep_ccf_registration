from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy.ndimage import map_coordinates

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import physical_to_index_space
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.utils.metrics import MSE, PerAxisError
from deep_ccf_registration.utils.utils import visualize_ccf_annotations


def viz_sample(
    input_image: np.ndarray,
    predicted_template_points: torch.Tensor,
    gt_template_points: torch.Tensor,
    ls_template_info: TemplateParameters,
    pad_mask: torch.Tensor,
    ccf_annotations: np.ndarray,
    terminology_path: Path,
    orientation: SliceOrientation,
    predict_tissue_mask: bool = False,
    tissue_mask: Optional[torch.Tensor] = None,
    predicted_tissue_masks: Optional[torch.Tensor] = None,
):
    """
    Visualize predicted vs ground truth registration.
    """
    pad_mask = pad_mask.bool()
    predicted_tissue_mask_binary = (predicted_tissue_masks > 0.5)

    gt_mask = (tissue_mask.bool() & pad_mask) if tissue_mask is not None else pad_mask
    pred_mask = predicted_tissue_mask_binary & pad_mask

    pred_template_points_index_space = physical_to_index_space(
        template_parameters=ls_template_info,
        physical_pts=predicted_template_points.unsqueeze(0),
        channel_dim=1
    )
    # remove batch dim and reshape to N, 3
    pred_template_points_index_space = pred_template_points_index_space[0].moveaxis(0, -1).view(-1, 3)
    gt_template_points_index_space = physical_to_index_space(
        template_parameters=ls_template_info,
        physical_pts=gt_template_points.unsqueeze(0),
        channel_dim=1
    )
    gt_template_points_index_space = gt_template_points_index_space[0].moveaxis(0, -1).view(-1, 3)

    ML_axis, AP_axis, SI_axis = 0, 1, 2

    per_axis_error = PerAxisError(
        template_parameters=ls_template_info,
    )
    error = per_axis_error(
        pred=predicted_template_points.unsqueeze(0),
        target=gt_template_points.unsqueeze(0),
        mask=gt_mask.unsqueeze(0),
        orientations=[orientation.value],
    )[0]

    total_error = error.sum(dim=0).sqrt() * 1000
    error *= 1000

    # reshape to N
    total_error = total_error.view(-1).numpy()

    # reshape to N, 3
    error = error.permute((1, 2, 0)).view(-1, 3).numpy()

    gt_mask = gt_mask.numpy()
    predicted_tissue_masks = predicted_tissue_masks.numpy()
    pad_mask = pad_mask.numpy()
    pred_mask = pred_mask.numpy()
    predicted_tissue_mask_binary = predicted_tissue_mask_binary.numpy()
    tissue_mask = tissue_mask.numpy()

    ml_lim = [0, ls_template_info.shape[ML_axis]]
    si_lim = [0, ls_template_info.shape[SI_axis]]
    ap_lim = [0, ls_template_info.shape[AP_axis]]

    fig = plt.figure(figsize=(25, 25), constrained_layout=True)
    gs = gridspec.GridSpec(4, 4, figure=fig)

    sns.set_style("darkgrid")

    #################
    # Input image #
    #################
    input_image_no_padding = input_image.copy()
    input_image_no_padding[~pad_mask] = np.nan
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(input_image_no_padding, cmap='gray')
    ax.set_title('Input')

    #################
    # tissue mask #
    #################

    if predict_tissue_mask and predicted_tissue_masks is not None:
        valid = pad_mask
        gt_mask_binary = tissue_mask.astype('bool')
        tp = np.sum(gt_mask_binary[valid] & predicted_tissue_mask_binary[valid])
        fp = np.sum(~gt_mask_binary[valid] & predicted_tissue_mask_binary[valid])
        fn = np.sum(gt_mask_binary[valid] & ~predicted_tissue_mask_binary[valid])

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        ax = fig.add_subplot(gs[0, 1])

        overlay = np.zeros((*input_image.shape, 3))
        agree = predicted_tissue_mask_binary & gt_mask_binary
        fp_mask = predicted_tissue_mask_binary & ~gt_mask_binary
        fn_mask = ~predicted_tissue_mask_binary & gt_mask_binary
        overlay[agree] = [0, 1, 0]
        overlay[fp_mask] = [1, 0, 0]
        overlay[fn_mask] = [0, 0, 1]
        overlay[~pad_mask] = np.nan

        ax.imshow(input_image_no_padding, cmap='gray')
        ax.imshow(overlay, alpha=0.4)
        ax.set_title(f'Mask Diff (IoU={iou:.3f}, Dice={dice:.3f})')
        ax.legend(handles=[
            Patch(color='green', alpha=0.4, label='TP'),
            Patch(color='red', alpha=0.4, label='FP'),
            Patch(color='blue', alpha=0.4, label='FN'),
        ], loc='lower right', fontsize='small')

    #################
    # GT front on 3d #
    #################
    ax = fig.add_subplot(gs[0, 2], projection='3d')
    ax.scatter(
        gt_template_points_index_space[:, ML_axis].reshape(input_image.shape)[gt_mask],
        gt_template_points_index_space[:, SI_axis].reshape(input_image.shape)[gt_mask],
        gt_template_points_index_space[:, AP_axis].reshape(input_image.shape)[gt_mask],
        c=input_image[gt_mask], cmap='gray', s=0.5
    )
    ax.view_init(elev=0, azim=0)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('GT')

    #################
    # Pred front on 3d #
    #################
    ax = fig.add_subplot(gs[0, 3], projection='3d')
    ax.scatter(
        pred_template_points_index_space[:, ML_axis].reshape(input_image.shape)[gt_mask],
        pred_template_points_index_space[:, SI_axis].reshape(input_image.shape)[gt_mask],
        pred_template_points_index_space[:, AP_axis].reshape(input_image.shape)[gt_mask],
        c=input_image[gt_mask], cmap='gray', s=0.5
    )
    ax.view_init(elev=0, azim=0)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('Predicted')

    terminology = pd.read_csv(terminology_path).set_index('annotation_value')

    #################
    # GT segmentation map #
    #################
    gt_ccf_annotations = map_coordinates(
        input=ccf_annotations,
        coordinates=gt_template_points_index_space.T,
        order=0,
        mode='nearest'
    )
    gt_ccf_annotations = gt_ccf_annotations.reshape(input_image.shape)
    gt_ccf_vis = visualize_ccf_annotations(
        annotations=gt_ccf_annotations, terminology=terminology, return_image=True
    )
    gt_colors = gt_ccf_vis[gt_mask] / 255.0

    gt_index_space = gt_template_points_index_space.reshape((*input_image.shape, 3))[gt_mask]
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    ax.scatter(
        gt_index_space[:, ML_axis], gt_index_space[:, SI_axis], gt_index_space[:, AP_axis],
        c=gt_colors, s=0.5
    )
    ax.view_init(elev=0, azim=0)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('GT segmentation map')

    #################
    # pred segmentation map #
    #################
    pred_ccf_annotations = map_coordinates(
        input=ccf_annotations,
        coordinates=pred_template_points_index_space.T,
        order=0,
        mode='nearest'
    )
    pred_ccf_annotations = pred_ccf_annotations.reshape(input_image.shape)
    pred_ccf_vis = visualize_ccf_annotations(
        annotations=pred_ccf_annotations, terminology=terminology, return_image=True
    )
    pred_ccf_vis[~pred_mask] = 0

    # Filter out background annotations
    valid_annotation = pred_ccf_annotations > 0
    pred_vis_mask = pred_mask & valid_annotation
    pred_colors = pred_ccf_vis[pred_vis_mask] / 255.0

    pred_index_space_masked = pred_template_points_index_space.reshape((*input_image.shape, 3))[
        pred_vis_mask]
    ax = fig.add_subplot(gs[1, 1], projection='3d')
    ax.scatter(
        pred_index_space_masked[:, ML_axis], pred_index_space_masked[:, SI_axis],
        pred_index_space_masked[:, AP_axis],
        c=pred_colors, s=0.5
    )
    ax.view_init(elev=0, azim=0)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('Pred segmentation map')


    #################
    # Heat maps #
    #################

    # Shared colorbar ranges
    gt_mask_flat = gt_mask.ravel()
    total_error_vmin = np.percentile(total_error[gt_mask_flat], 5)
    total_error_vmax = np.percentile(total_error[gt_mask_flat], 95)

    all_axis_errors = error[gt_mask_flat].ravel()
    axis_error_vmin = np.percentile(all_axis_errors, 5)
    axis_error_vmax = np.percentile(all_axis_errors, 95)

    pred_index_space = pred_template_points_index_space.reshape((*input_image.shape, 3))[gt_mask]

    #################
    # total error heatmap #
    #################
    ax = fig.add_subplot(gs[2, 0], projection='3d')
    im = ax.scatter(
        pred_index_space[:, ML_axis], pred_index_space[:, SI_axis], pred_index_space[:, AP_axis],
        c=total_error[gt_mask_flat], cmap='hot', vmin=total_error_vmin, vmax=total_error_vmax,
        s=0.5
    )
    ax.view_init(elev=0, azim=0)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('Total Error (microns)')
    plt.colorbar(im, ax=ax, shrink=0.5)

    #################
    # per-axis error heatmap #
    #################
    axis_configs = [
        (gs[2, 1], SI_axis, 'SI Error (microns)'),
        (gs[2, 2], AP_axis, 'AP Error (microns)'),
        (gs[2, 3], ML_axis, 'ML Error (microns)'),
    ]
    for gs_pos, axis_idx, title in axis_configs:
        ax = fig.add_subplot(gs_pos, projection='3d')
        im = ax.scatter(
            pred_index_space[:, ML_axis], pred_index_space[:, SI_axis],
            pred_index_space[:, AP_axis],
            c=error[:, axis_idx][gt_mask_flat], cmap='hot', vmin=axis_error_vmin,
            vmax=axis_error_vmax, s=0.5
        )
        ax.view_init(elev=0, azim=0)
        ax.set_xlim(ml_lim)
        ax.set_ylim(si_lim)
        ax.set_zlim(ap_lim)
        ax.set_xlabel('ML')
        ax.set_ylabel('SI')
        ax.set_zlabel('AP')
        ax.set_box_aspect([
            ls_template_info.shape[ML_axis],
            ls_template_info.shape[SI_axis],
            ls_template_info.shape[AP_axis]
        ])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.5)


    #################
    # GT side view 3d #
    #################
    ax_gt3d = fig.add_subplot(gs[3, 0], projection='3d')
    ax_gt3d.scatter(
        gt_template_points_index_space[:, ML_axis].reshape(input_image.shape)[gt_mask],
        gt_template_points_index_space[:, SI_axis].reshape(input_image.shape)[gt_mask],
        gt_template_points_index_space[:, AP_axis].reshape(input_image.shape)[gt_mask],
        c=input_image[gt_mask], cmap='gray', s=0.5
    )
    ax_gt3d.view_init(elev=20, azim=90)
    ax_gt3d.set_xlim(ml_lim)
    ax_gt3d.set_ylim(si_lim)
    ax_gt3d.set_zlim(ap_lim)
    ax_gt3d.set_xlabel('ML')
    ax_gt3d.set_ylabel('SI')
    ax_gt3d.set_zlabel('AP')
    ax_gt3d.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax_gt3d.set_title('Ground Truth')

    #################
    # Pred side view 3d #
    #################
    ax = fig.add_subplot(gs[3, 1], projection='3d')
    ax.scatter(
        pred_template_points_index_space[:, ML_axis].reshape(input_image.shape)[gt_mask],
        pred_template_points_index_space[:, SI_axis].reshape(input_image.shape)[gt_mask],
        pred_template_points_index_space[:, AP_axis].reshape(input_image.shape)[gt_mask],
        c=input_image[gt_mask], cmap='gray', s=0.5
    )
    ax.view_init(elev=20, azim=90)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('Predicted')

    #################
    # GT side view different angle 3d #
    #################
    ax_gt3d = fig.add_subplot(gs[3, 2], projection='3d')
    ax_gt3d.scatter(
        gt_template_points_index_space[:, ML_axis].reshape(input_image.shape)[gt_mask],
        gt_template_points_index_space[:, SI_axis].reshape(input_image.shape)[gt_mask],
        gt_template_points_index_space[:, AP_axis].reshape(input_image.shape)[gt_mask],
        c=input_image[gt_mask], cmap='gray', s=0.5
    )
    ax_gt3d.view_init(elev=20, azim=60)
    ax_gt3d.set_xlim(ml_lim)
    ax_gt3d.set_ylim(si_lim)
    ax_gt3d.set_zlim(ap_lim)
    ax_gt3d.set_xlabel('ML')
    ax_gt3d.set_ylabel('SI')
    ax_gt3d.set_zlabel('AP')
    ax_gt3d.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax_gt3d.set_title('Ground Truth')

    #################
    # Pred side view different angle 3d #
    #################
    ax = fig.add_subplot(gs[3, 3], projection='3d')
    ax.scatter(
        pred_template_points_index_space[:, ML_axis].reshape(input_image.shape)[gt_mask],
        pred_template_points_index_space[:, SI_axis].reshape(input_image.shape)[gt_mask],
        pred_template_points_index_space[:, AP_axis].reshape(input_image.shape)[gt_mask],
        c=input_image[gt_mask], cmap='gray', s=0.5
    )
    ax.view_init(elev=20, azim=60)
    ax.set_xlim(ml_lim)
    ax.set_ylim(si_lim)
    ax.set_zlim(ap_lim)
    ax.set_xlabel('ML')
    ax.set_ylabel('SI')
    ax.set_zlabel('AP')
    ax.set_box_aspect([
        ls_template_info.shape[ML_axis],
        ls_template_info.shape[SI_axis],
        ls_template_info.shape[AP_axis]
    ])
    ax.set_title('Predicted')

    return fig