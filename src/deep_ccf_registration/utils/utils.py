from typing import Optional

import ants
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity


def visualize_alignment(
    input_slice: np.ndarray | torch.Tensor,
    template_on_input: np.ndarray,
    registered_slice: Optional[np.ndarray] = None,
):
    if isinstance(input_slice, torch.Tensor):
        input_slice = input_slice.numpy()

    input_slice = rescale_intensity(
        input_slice,
        in_range=tuple(np.percentile(input_slice, (1, 99))),
        out_range=(0, 1)
    )
    template_on_input = np.array(template_on_input).reshape(input_slice.shape)
    template_on_input = rescale_intensity(template_on_input, out_range=(0, 1))

    raw_rgb = np.zeros_like(input_slice, shape=(*input_slice.shape, 3))
    raw_rgb[:, :, 0] = input_slice

    template_on_input_rgb = np.zeros_like(template_on_input,
                                             shape=(*input_slice.shape, 3))
    template_on_input_rgb[:, :, 2] = template_on_input

    height, width = input_slice.shape
    if width > height:
        figsize = (30, 15)
    else:
        figsize = (15, 30)

    ncols = 3
    if registered_slice is not None:
        ncols += 1
    fig, ax = plt.subplots(figsize=figsize, ncols=ncols, dpi=100)
    ax[0].imshow(input_slice, alpha=0.8)
    ax[0].imshow(template_on_input_rgb, alpha=0.4)
    ax[1].imshow(input_slice, cmap='gray')
    ax[1].set_title('input')
    ax[2].imshow(template_on_input, cmap='gray')
    ax[2].set_title('Template')

    if registered_slice is not None:
        ax[3].imshow(registered_slice, cmap='gray')
        ax[3].set_title('registered slice')
    return fig

def create_3d_plot(points: pd.DataFrame, space: ants.ANTsImage) -> plt.Figure:
    points = points[(points >= 0).all(axis=1)]
    points = points[(points.loc[:, 'ML'] <= space.shape[0]) & (points.loc[:, 'AP'] <= space.shape[1]) & (points.loc[:, 'DV'] <= space.shape[2])]

    ap_coords = points['AP']
    dv_coords = points['DV']
    ml_coords = points['ML']

    fig = plt.figure(figsize=(15, 10))

    # Main 3D plot - ML view to see slice angle
    ax_3d = fig.add_subplot(projection='3d')

    ax_3d.scatter(ap_coords, ml_coords, dv_coords,
                           alpha=0.8, s=30, edgecolors='black', linewidth=0.2)

    ax_3d.set_ylim(0, space.shape[0])

    # Labels for ML view
    ax_3d.set_xlabel('Anterior ← → Posterior', fontsize=11, fontweight='bold')
    ax_3d.set_ylabel('Right ← → Left', fontsize=11, fontweight='bold')
    ax_3d.set_zlabel('Superior ← → Inferior', fontsize=11, fontweight='bold')
    ax_3d.set_title('Sagittal slice in LS template space', fontsize=13, fontweight='bold')

    ax_3d.view_init(elev=30, azim=30)

    plt.tight_layout()
    return fig

def sample_template_at_points(
    template_points: torch.Tensor,
    template: ants.ANTsImage,
    interpolation='nearest'
) -> np.ndarray:
    # flatten
    template_points = template_points.reshape((-1, 3))

    # Check bounds
    valid_mask = (
            (template_points[:, 0] >= 0) & (template_points[:, 0] < template.shape[0]) &
            (template_points[:, 1] >= 0) & (template_points[:, 1] < template.shape[1]) &
            (template_points[:, 2] >= 0) & (template_points[:, 2] < template.shape[2])
    )

    values = np.zeros(len(template_points))

    if interpolation == 'nearest':
        # Round to nearest integer
        template_points = template_points[valid_mask].floor().int().cpu().numpy()
        values[valid_mask.cpu().numpy()] = template.numpy()[
            template_points[:, 0],
            template_points[:, 1],
            template_points[:, 2]
        ]
    else:
        raise NotImplementedError

    return values


def visualize_ccf_annotations(
        annotations: np.ndarray,
        colormap: dict[int, list],
        return_image: bool = True
) -> np.ndarray:
    """
    Visualize CCF annotations with official Allen colors.

    Args:
        annotations: (H, W) array of annotation IDs
        colormap: Optional custom colormap dict. If None, uses Allen official colors.
        return_image: If True, returns RGB image. If False, displays with matplotlib.

    Returns:
        RGB image array (H, W, 3) with uint8 values [0-255]
    """
    # Create RGB image
    h, w = annotations.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Map each annotation ID to its color
    unique_ids = np.unique(annotations)
    for ann_id in unique_ids:
        if ann_id == 0:  # Skip background
            continue
        mask = annotations == ann_id
        if ann_id in colormap:
            rgb_image[mask] = colormap[ann_id]
        else:
            # Fallback: generate a random color for unmapped IDs
            np.random.seed(int(ann_id))  # Deterministic color
            rgb_image[mask] = np.random.randint(50, 255, 3)

    if not return_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title('CCF Annotations')
        plt.show()

    return rgb_image


def fetch_complete_colormap() -> dict[int, list]:
    """
    Fetch the complete colormap from Allen Brain Atlas API.
    Requires internet connection.

    Returns:
        Complete dictionary mapping all annotation IDs -> RGB [0-255]
    """
    import requests

    def hex_to_rgb(hex_color):
        return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]

    def extract_recursive(node, colormap):
        if 'id' in node and 'color_hex_triplet' in node:
            colormap[node['id']] = hex_to_rgb(node['color_hex_triplet'])
        if 'children' in node:
            for child in node['children']:
                extract_recursive(child, colormap)

    url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
    response = requests.get(url)
    data = response.json()

    colormap = {0: [0, 0, 0]}
    if 'msg' in data and len(data['msg']) > 0:
        extract_recursive(data['msg'][0], colormap)

    return colormap