import json
from pathlib import Path

import aind_smartspim_transform_utils
import numpy as np
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import get_orientation, \
    convert_to_ants_space, convert_from_ants_space
from scipy.ndimage import map_coordinates

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import AcquisitionAxis


def create_coordinate_grid(
        patch_height: int,
        patch_width: int,
        start_x: int,
        start_y: int,
        fixed_index_value: int,
        slice_axis: AcquisitionAxis,
        axes: list[AcquisitionAxis]
) -> np.ndarray:
    """
    Create coordinate grid for a patch at specific position.

    Parameters
    ----------
    patch_height : int
        Height of the patch in pixels.
    patch_width : int
        Width of the patch in pixels.
    start_x : int
        Starting x coordinate of the patch.
    start_y : int
        Starting y coordinate of the patch.
    fixed_index_value : int
        Index value for the fixed slice dimension.
    slice_axis : AcquisitionAxis
        Axis along which slicing occurs.
    axes : list[AcquisitionAxis]
        List of all acquisition axes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing coordinate points for the patch.
    """
    # Create meshgrid with actual coordinates
    axis1_coords, axis2_coords = np.meshgrid(
        np.arange(start_y, start_y + patch_height),
        np.arange(start_x, start_x + patch_width),
        indexing='ij'
    )

    axis1_flat = axis1_coords.flatten()
    axis2_flat = axis2_coords.flatten()

    n_points = len(axis1_flat)

    slice_index = np.full(n_points, fixed_index_value)

    axes = sorted(axes, key=lambda x: x.dimension)

    points = np.zeros((n_points, 3))

    points[:, slice_axis.dimension] = slice_index
    points[:, [x for x in axes if x != slice_axis][0].dimension] = axis1_flat
    points[:, [x for x in axes if x != slice_axis][1].dimension] = axis2_flat
    return points

def transform_points_to_template_space(
    acquisition_axes: list[AcquisitionAxis],
    ls_template_info: AntsImageParameters,
    points: np.ndarray,
    input_volume_shape: tuple[int, int, int],
    template_resolution: int = 25,
    registration_downsample: float = 3.0,
) -> np.ndarray:
    """
    Transform points from input index space to physical template ANTs space.

    Performs orientation alignment, scaling, and coordinate system conversion
    to map points from acquisition space to template space.

    Parameters
    ----------
    acquisition_axes : list[AcquisitionAxis]
        Acquisition axes defining the input volume orientation.
    ls_template_info : AntsImageParameters
        Template image parameters.
    points : np.ndarray
        Points in input volume coordinates.
    input_volume_shape : tuple[int, int, int]
        Shape of the input volume.
    template_resolution : int, default=25
        Resolution of the template in micrometers.
    registration_downsample : float, default=3.0
        Downsample factor used during registration.

    Returns
    -------
    np.ndarray
        Points in ANTs template space.
    """
    acquisition_axes = sorted(acquisition_axes, key=lambda x: x.dimension)

    orient = get_orientation([json.loads(x.model_dump_json()) for x in acquisition_axes])

    _, swapped, mat = aind_smartspim_transform_utils.utils.utils.get_orientation_transform(
        orient, ls_template_info.orientation
    )

    # flip axis based on the template orientation relative to input image
    for idx, dim_orient in enumerate(mat.sum(axis=1)):
        if dim_orient < 0:
            points[:, idx] = input_volume_shape[idx] - points[:, idx]

    # scale points
    points_resolution = [x.resolution * 2 ** registration_downsample for x in acquisition_axes]
    scaling = [res_1 / res_2 for res_1, res_2 in zip(points_resolution, [template_resolution] * 3)]
    scaled_pts = scale_points(points=points, scaling=scaling)

    # orient axes to template
    orient_pts = scaled_pts[:, swapped]

    # convert points into ccf space
    ants_pts = convert_to_ants_space(
        ls_template_info, orient_pts
    )

    return ants_pts

def scale_points(points: np.ndarray, scaling: list[float]) -> np.ndarray:
    assert points.shape[1] == 3 and len(points.shape) == 2
    assert len(scaling) == 3
    scale = np.array([scaling])
    points *= scale
    return points

def apply_transforms_to_points(
    points: np.ndarray,
    affine_path: Path,
    warp: np.ndarray,
    template_parameters: AntsImageParameters,
) -> np.ndarray:
    """
    Apply affine and non-linear transformations to points in input space.

    Transforms points from input space to template space by applying inverse affine
    transformation followed by displacement field warping.

    Parameters
    ----------
    points : np.ndarray
        Points in physical input space to be transformed.
    affine_path:
        Affine path
    warp : tensorstore.TensorStore or np.ndarray
        Displacement field for non-linear transformation.
    template_parameters : AntsImageParameters
        Template image parameters.

    Returns
    -------
    np array of shape n points x 3. The 2nd dim is ordered ["ML", "AP", "DV"] according to light sheet template orientation.
    The points are in physical space.
    """
    # apply inverse affine to points in input space
    # this returns points in physical space
    affine_transformed_points = aind_smartspim_transform_utils.utils.utils.apply_transforms_to_points(
        ants_pts=points,
        transforms=[str(affine_path)],
        invert=(True,)
    )

    # convert physical points to voxels,
    # so we can index into the displacement field
    affine_transformed_voxels = convert_from_ants_space(
        template_parameters=template_parameters,
        physical_pts=affine_transformed_points
    )

    displacements = np.empty(affine_transformed_voxels.shape)

    for i in range(3):
        displacements[..., i] = map_coordinates(
            input=warp[..., i],
            coordinates=affine_transformed_voxels.T,
            order=1,
            mode="constant",
            cval=0,
        )

    # apply displacement vector to affine transformed points
    transformed_points = affine_transformed_points + displacements

    return transformed_points