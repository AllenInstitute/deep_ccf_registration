from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from deep_ccf_registration.utils.utils import fetch_complete_colormap, visualize_ccf_annotations


def create_diagnostic_image(
        input_image: torch.Tensor,
        pred_template_points: torch.Tensor,
        gt_template_points: torch.Tensor,
        pred_ccf_annotations: np.ndarray,
        gt_ccf_annotations: np.ndarray,
        squared_errors: np.ndarray,
        mask: np.ndarray,
        slice_idx: int,
        vmax_percentile: float = 95.0,
        iteration: Optional[int] = None,
):
    """
    Visualize error heatmap, coordinate predictions, and ccf annotation predictions

    :param input_image: normalized input image (H, W)
    :param pred_template_points: predicted coordinates (3, H, W) in LS template space
    :param gt_template_points: ground truth coordinates (3, H, W) in LS template space
    :param pred_ccf_annotations: predicted CCF annotations (H, W)
    :param gt_ccf_annotations: ground truth CCF annotations (H, W)
    :param squared_errors: squared error for all dims  in ANTs space (millimeters) shape (3, H, W)
    :param slice_idx: slice index for title
    :param vmax_percentile: percentile for colormap max
    :param iteration: optional iteration number
    :param exclude_background_pixels: whether to use a tissue mask to exclude background pixels
        Otherwise, just excludes pad pixels
    :param mask where to calculate metrics
    """
    sse = squared_errors.sum(axis=0)
    rmse = np.sqrt(sse[mask].mean()) * 1000

    pred_template_points = pred_template_points.cpu().numpy()
    gt_template_points = gt_template_points.cpu().numpy()

    error_heatmap = np.sqrt(sse) * 1000

    # Mask coordinates - set background to NaN for visualization
    for i in range(3):
        gt_template_points[i][~mask] = np.nan
        pred_template_points[i][~mask] = np.nan
    error_heatmap[~mask] = np.nan

    # Create figure with multiple rows
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Input, Error, CCF annotations
    ax_input = fig.add_subplot(gs[0, 0])
    ax_error = fig.add_subplot(gs[0, 1])
    ax_pred_ccf = fig.add_subplot(gs[0, 2])
    ax_gt_ccf = fig.add_subplot(gs[0, 3])

    # Plot 1: Input image
    ax_input.imshow(input_image, cmap='gray')
    ax_input.set_title('Input Image', fontsize=12)

    # Plot 2: Error heatmap
    im_error = ax_error.imshow(error_heatmap, cmap='turbo', alpha=0.7, vmin=0, vmax=np.percentile(error_heatmap[mask], vmax_percentile))
    ax_error.set_title('Error (microns)', fontsize=12)
    plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)

    # Plot 3 & 4: CCF annotations
    colormap = fetch_complete_colormap()
    pred_ccf_vis = visualize_ccf_annotations(annotations=pred_ccf_annotations, colormap=colormap,
                                             return_image=True)
    gt_ccf_vis = visualize_ccf_annotations(annotations=gt_ccf_annotations, colormap=colormap,
                                           return_image=True)

    ax_pred_ccf.imshow(pred_ccf_vis)
    ax_pred_ccf.set_title('Predicted CCF annotations', fontsize=12)

    ax_gt_ccf.imshow(gt_ccf_vis)
    ax_gt_ccf.set_title('Ground truth CCF annotations', fontsize=12)

    # Row 2: Ground truth coordinate channels (ML, AP, DV) - masked
    coord_labels = ['ML (Medial-Lateral)', 'AP (Anterior-Posterior)', 'DV (Dorsal-Ventral)']
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(gt_template_points[i] * 1000, cmap='viridis')
        ax.set_title(f'GT {coord_labels[i]}', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add coordinate error per dimension (tissue only)
    ax_coord_error = fig.add_subplot(gs[1, 3])
    mean_coord_errors = [np.sqrt(squared_errors[i][mask].mean()) * 1000 for i in range(3)]
    ax_coord_error.bar(['ML', 'AP', 'DV'], mean_coord_errors, color=['red', 'green', 'blue'])
    ax_coord_error.set_title('Mean Error per Coordinate (μm)', fontsize=12)
    ax_coord_error.set_ylabel('Error (microns)')
    ax_coord_error.grid(True, alpha=0.3)

    # Row 3: Predicted coordinate channels (ML, AP, DV) - masked
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(pred_template_points[i] * 1000, cmap='viridis')
        ax.set_title(f'Pred {coord_labels[i]}', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overall title
    iteration_title = f'iter {iteration} ' if iteration else ''
    title = f'{iteration_title}slice: {slice_idx} RMSE {rmse:.3f} microns'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig