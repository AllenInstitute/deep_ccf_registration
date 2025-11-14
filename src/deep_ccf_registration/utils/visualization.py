import numpy as np
import torch
from matplotlib import pyplot as plt

from deep_ccf_registration.utils.utils import fetch_complete_colormap, visualize_ccf_annotations


def create_diagnostic_image(
        input_image: torch.Tensor,
        pred_ccf_annotations: np.ndarray,
        gt_cff_annotations: np.ndarray,
        squared_errors: np.ndarray,
        slice_idx: int,
        vmax_percentile: float = 95.0,
):
    """
    Visualize error heatmap, and ccf annotation predictions

    :param input_image: normalized input image (H, W)
    :param squared_errors: sum of squares for all points across all dims (H, W)
    :param slice_idx: optional slice index for title
    :param vmax_percentile: percentile for colormap max
    :param pred_ccf_annotations
    :param gt_cff_annotations
    """
    # convert to microns
    squared_errors_microns = squared_errors * 1000

    error_map = np.sqrt(squared_errors_microns)

    # Calculate vmax from percentile on non-zero values
    error_nonzero = error_map[error_map > 0]
    if len(error_nonzero) > 0:
        vmax = np.percentile(error_nonzero, vmax_percentile)
        rmse = np.sqrt(np.mean(error_nonzero ** 2))  # RMSE over tissue only
    else:
        vmax = None
        rmse = 0.0

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # Plot 1: Input image
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot 2: Input image with MSE heatmap overlay
    axes[1].imshow(input_image, cmap='gray', alpha=0.3)
    im = axes[1].imshow(error_map, cmap='turbo', alpha=0.7, vmin=0, vmax=vmax)
    axes[1].set_title('Error (microns)')
    axes[1].axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Error (microns)', rotation=270, labelpad=20)

    colormap = fetch_complete_colormap()
    pred_ccf_annotations = visualize_ccf_annotations(annotations=pred_ccf_annotations, colormap=colormap, return_image=True)
    gt_ccf_annotations = visualize_ccf_annotations(annotations=gt_cff_annotations, colormap=colormap, return_image=True)

    # Plot 3: predicted ccf annotations
    axes[2].imshow(pred_ccf_annotations)
    axes[2].set_title('Predicted CCF annotations')

    # Plot 4: predicted ccf annotations
    axes[3].imshow(gt_ccf_annotations)
    axes[3].set_title('Ground truth CCF annotations')

    # Overall title
    title = f'Slice: {slice_idx} RMSE {rmse:.3f} microns'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig