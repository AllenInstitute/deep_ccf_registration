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
    :param squared_errors: sum of squares for all points across all dims (H, W)
    :param slice_idx: slice index for title
    :param vmax_percentile: percentile for colormap max
    :param iteration: optional iteration number
    """
    error_map = np.sqrt(squared_errors) * 1000

    # Calculate vmax from percentile on non-zero values
    error_nonzero = error_map[error_map > 0]
    if len(error_nonzero) > 0:
        vmax = np.percentile(error_nonzero, vmax_percentile)
        rmse = np.sqrt(np.mean(error_nonzero ** 2))  # RMSE over tissue only
    else:
        vmax = None
        rmse = 0.0

    # Convert coordinates to numpy if needed
    if isinstance(pred_template_points, torch.Tensor):
        pred_coords = pred_template_points.cpu().numpy()  # (3, H, W)
    else:
        pred_coords = pred_template_points

    if isinstance(gt_template_points, torch.Tensor):
        gt_coords = gt_template_points.cpu().numpy()  # (3, H, W)
    else:
        gt_coords = gt_template_points

    # Create tissue mask from ground truth annotations (background = 0)
    tissue_mask = gt_ccf_annotations != 0

    # Mask coordinates - set background to NaN for visualization
    gt_coords_masked = gt_coords.copy()
    pred_coords_masked = pred_coords.copy()
    for i in range(3):
        gt_coords_masked[i][~tissue_mask] = np.nan
        pred_coords_masked[i][~tissue_mask] = np.nan

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
    ax_input.axis('off')

    # Plot 2: Error heatmap
    ax_error.imshow(input_image, cmap='gray', alpha=0.3)
    im_error = ax_error.imshow(error_map, cmap='turbo', alpha=0.7, vmin=0, vmax=vmax)
    ax_error.set_title('Error (microns)', fontsize=12)
    ax_error.axis('off')
    plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)

    # Plot 3 & 4: CCF annotations
    colormap = fetch_complete_colormap()
    pred_ccf_vis = visualize_ccf_annotations(annotations=pred_ccf_annotations, colormap=colormap,
                                             return_image=True)
    gt_ccf_vis = visualize_ccf_annotations(annotations=gt_ccf_annotations, colormap=colormap,
                                           return_image=True)

    ax_pred_ccf.imshow(pred_ccf_vis)
    ax_pred_ccf.set_title('Predicted CCF annotations', fontsize=12)
    ax_pred_ccf.axis('off')

    ax_gt_ccf.imshow(gt_ccf_vis)
    ax_gt_ccf.set_title('Ground truth CCF annotations', fontsize=12)
    ax_gt_ccf.axis('off')

    # Row 2: Ground truth coordinate channels (ML, AP, DV) - masked
    coord_labels = ['ML (Medial-Lateral)', 'AP (Anterior-Posterior)', 'DV (Dorsal-Ventral)']
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(gt_coords_masked[i], cmap='viridis')
        ax.set_title(f'GT {coord_labels[i]}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add coordinate error per dimension (tissue only)
    ax_coord_error = fig.add_subplot(gs[1, 3])
    coord_errors = np.abs(pred_coords - gt_coords) * 1000  # Convert to microns
    # Calculate mean only over tissue pixels
    mean_coord_errors = [coord_errors[i][tissue_mask].mean() for i in range(3)]
    ax_coord_error.bar(['ML', 'AP', 'DV'], mean_coord_errors, color=['red', 'green', 'blue'])
    ax_coord_error.set_title('Mean Error per Coordinate (μm)', fontsize=12)
    ax_coord_error.set_ylabel('Error (microns)')
    ax_coord_error.grid(True, alpha=0.3)

    # Row 3: Predicted coordinate channels (ML, AP, DV) - masked
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(pred_coords_masked[i], cmap='viridis')
        ax.set_title(f'Pred {coord_labels[i]}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add spatial gradient magnitude for GT coordinates (tissue only)
    ax_gradient = fig.add_subplot(gs[2, 3])
    # Calculate gradient magnitude across all coordinate dimensions
    grad_x = np.gradient(gt_coords, axis=2)  # (3, H, W)
    grad_y = np.gradient(gt_coords, axis=1)  # (3, H, W)
    grad_magnitude = np.sqrt((grad_x ** 2 + grad_y ** 2).sum(axis=0)) * 1000  # Microns per pixel
    grad_magnitude_masked = np.where(tissue_mask, grad_magnitude, np.nan)
    im_grad = ax_gradient.imshow(grad_magnitude_masked, cmap='plasma')
    ax_gradient.set_title('GT Coordinate Gradient (μm/px)', fontsize=12)
    ax_gradient.axis('off')
    plt.colorbar(im_grad, ax=ax_gradient, fraction=0.046, pad=0.04)

    # Overall title
    iteration_title = f'iter {iteration} ' if iteration else ''
    title = f'{iteration_title}slice: {slice_idx} RMSE {rmse:.3f} microns'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig