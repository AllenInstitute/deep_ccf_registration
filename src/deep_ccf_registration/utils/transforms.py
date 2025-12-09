import json
from pathlib import Path

import aind_smartspim_transform_utils
import numpy as np
import pandas as pd
import tensorstore
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from deep_ccf_registration.metadata import AcquisitionAxis
from deep_ccf_registration.utils.interpolation import interpolate
from deep_ccf_registration.utils.logging_utils import timed_func


@timed_func
def transform_points_to_template_ants_space(
    acquisition_axes: list[AcquisitionAxis],
    ls_template_info: AntsImageParameters,
    points: pd.DataFrame,
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
    points : pd.DataFrame
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

    # order columns to align with imaging
    points = points[[x.name.value.lower() for x in acquisition_axes]].values

    orient = aind_smartspim_transform_utils.utils.utils.get_orientation([json.loads(x.model_dump_json()) for x in acquisition_axes])

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
    scaled_pts = aind_smartspim_transform_utils.utils.utils.scale_points(points, scaling)

    # orient axes to template
    orient_pts = scaled_pts[:, swapped]

    # convert points into ccf space
    ants_pts = aind_smartspim_transform_utils.utils.utils.convert_to_ants_space(
        ls_template_info, orient_pts
    )

    return ants_pts


@timed_func
def apply_transforms_to_points(
    points: np.ndarray,
    affine_path: Path,
    warp: tensorstore.TensorStore | np.ndarray,
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

    displacements = interpolate(
        array=warp,
        grid=affine_transformed_voxels,
        mode='bilinear'
    )

    # (1, 3, N, 1, 1) -> (N, 3)
    displacements = displacements.squeeze().T.numpy()

    # apply displacement vector to affine transformed points
    transformed_points = affine_transformed_points + displacements

    return transformed_points


def convert_from_ants_space_tensor(template_parameters: AntsImageParameters, physical_pts: torch.Tensor):
    """
    Convert points from the physical space of an ANTsImage and places
    them into the "index" space required for visualizing

    Parameters
    ----------
    template_parameters : `AntsImageParameters`
        parameters of the ANTsImage physical space from where you are
        converting the points
    physical_pts : torch.Tensor
        the location of cells in physical space

    Returns
    -------
    pts : np.ndarray
        pts converted for ANTsPy physical space to "index" space

    """

    pts = physical_pts.clone()

    for dim in range(template_parameters.dims):
        pts[:, dim] -= template_parameters.origin[dim]
        pts[:, dim] *= template_parameters.direction[dim]
        pts[:, dim] /= template_parameters.scale[dim]

    return pts

def map_points_to_left_hemisphere(
        template_points: np.ndarray,
        template_parameters: AntsImageParameters,
        ml_dim_size: int
):
    """
    Because which hemisphere a slice is in is ambiguous, this maps the ground truth template points
    to the right hemisphere.

    :param template_points:
    :param template_parameters:
    :param ml_dim_size: The ML dim size in template index space
    :return:
    """
    points_index_space = template_points.copy()
    for dim in range(template_parameters.dims):
        points_index_space[:, :, dim] -= template_parameters.origin[dim]
        points_index_space[:, :, dim] *= template_parameters.direction[dim]
        points_index_space[:, :, dim] /= template_parameters.scale[dim]

    # checks whether the ML points are > halfway in index space. the LS template iS RAS.
    # therefore this checks whether points are in left hemisphere
    need_mirror = (points_index_space[:, :, 0] > ml_dim_size / 2).all()
    if need_mirror:
        # map to right hemisphere
        template_points = mirror_points(points=np.permute_dims(np.expand_dims(template_points, axis=0), (0, 3, 1, 2)), template_parameters=template_parameters, ml_dim_size=ml_dim_size)
        template_points = np.permute_dims(template_points.squeeze(0), (1, 2, 0))
    return template_points


def mirror_points(points: torch.Tensor | np.ndarray, template_parameters: AntsImageParameters, ml_dim_size: int):
    flipped = points.clone() if isinstance(points, torch.Tensor) else points.copy()

    # 1. Convert to index space
    for dim in range(template_parameters.dims):
        flipped[:, dim] -= template_parameters.origin[dim]
        flipped[:, dim] *= template_parameters.direction[dim]
        flipped[:, dim] /= template_parameters.scale[dim]

    # 2. Flip ML in index space
    flipped[:, 0] = ml_dim_size-1 - flipped[:, 0]

    # 3. Convert back to physical
    for dim in range(template_parameters.dims):
        flipped[:, dim] *= template_parameters.scale[dim]
        flipped[:, dim] *= template_parameters.direction[dim]
        flipped[:, dim] += template_parameters.origin[dim]

    return flipped
