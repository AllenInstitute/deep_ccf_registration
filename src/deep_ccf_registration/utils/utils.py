import ants
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity


def visualize_alignment(input_slice: torch.Tensor, template_points: torch.Tensor, template: ants.ANTsImage):
    raw_rgb = np.zeros_like(input_slice, shape=(*input_slice.shape, 3))
    raw_rgb[:, :, 0] = input_slice

    template_on_input = sample_template_at_points(
        template=template,
        template_points=template_points
    )
    template_on_input_rgb = np.zeros_like(template_on_input,
                                             shape=(*input_slice.shape, 3))
    template_on_input_rgb[:, :, 2] = template_on_input.reshape(input_slice.shape)

    fig, ax = plt.subplots(figsize=(10, 10), ncols=3)
    ax[0].imshow(rescale_intensity(raw_rgb, out_range=(0, 1)), alpha=0.8)
    ax[0].imshow(rescale_intensity(template_on_input_rgb, out_range=(0, 1)), alpha=0.4)
    ax[1].imshow(input_slice, cmap='gray')
    ax[1].set_title('input')
    ax[2].imshow(template_on_input.reshape(input_slice.shape), cmap='gray')
    ax[2].set_title('Template')
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