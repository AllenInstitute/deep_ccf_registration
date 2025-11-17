import json
from pathlib import Path

import aind_smartspim_transform_utils
import numpy as np
import pandas as pd
import tensorstore
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from aind_smartspim_transform_utils.utils import utils
from tensorstore import TensorStore

from deep_ccf_registration.metadata import AcquisitionAxis
from deep_ccf_registration.utils.interpolation import interpolate
from deep_ccf_registration.utils.logging_utils import timed_func, timed


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
    warp_interpolation_padding: int = 5,
    crop_warp_to_bounding_box: bool = True
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
    warp_interpolation_padding : int, default=5
        Padding for warp interpolation.
    crop_warp_to_bounding_box : bool, default=True
        Whether to crop warp to bounding box of transformed points.

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

    with timed():
        if crop_warp_to_bounding_box:
            warp = get_cropped_region_from_array(
                array=warp,
                points=affine_transformed_voxels,
                padding=warp_interpolation_padding
            )
        else:
            if isinstance(warp, TensorStore):
                warp = warp[:].read().result()

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


def get_cropped_region_from_array(
    array: tensorstore.TensorStore | np.ndarray,
    points: np.ndarray,
    padding: int = 5
) -> np.ndarray:
    """
    Crop array to region bounded by min/max coordinates in points. This avoids loading the entire
    array.

    It also modifies points in-place to set offset
    to 0 so that it can index into the cropped array.

    Parameters
    ----------
    array : tensorstore.TensorStore or np.ndarray
        Full displacement field.
    points : np.ndarray
        Voxels after applying inverse affine to input points. Modified in-place.
    padding : int, default=5
        Padding around the min/max coords to crop for interpolation.

    Returns
    -------
    np.ndarray
        Cropped displacement field.

    Raises
    ------
    ValueError
        If points are completely outside array bounds.
    """
    array_shape = np.array(array.shape[:-1])

    min_coords = np.floor(points.min(axis=0)).astype(
        int) - padding
    max_coords = np.ceil(points.max(axis=0)).astype(
        int) + padding

    orig_min, orig_max = min_coords.copy(), max_coords.copy()

    # Clamp to warp dimensions
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, array_shape)

    if np.any(max_coords <= min_coords):
        raise ValueError(
            f"Points are completely outside array.\n"
            f"Original bbox: {orig_min} to {orig_max}\n"
            f"After clamping: {min_coords} to {max_coords}\n"
            f"Array shape: {array_shape}\n"
        )

    # Crop the array
    if isinstance(array, tensorstore.TensorStore):
        cropped_array = array[
                       min_coords[0]:max_coords[0],
                       min_coords[1]:max_coords[1],
                       min_coords[2]:max_coords[2]
                       ].read().result()
    else:
        cropped_array = array[
                       min_coords[0]:max_coords[0],
                       min_coords[1]:max_coords[1],
                       min_coords[2]:max_coords[2]
                       ]

    # Adjust voxel coordinates relative to cropped region
    points -= min_coords

    return cropped_array


def transform_ls_space_to_ccf_space(
        points: torch.Tensor | np.ndarray,
        ls_template_to_ccf_affine_path: Path,
        ls_template_to_ccf_inverse_warp: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        ccf_template_parameters: AntsImageParameters,
) -> np.ndarray:
    """
    Transform points to CCF index space

    :param points: MxNx3 predicted points in LS template space
    :return: ccf points in index space
    """
    if isinstance(points, torch.Tensor):
        points = points.numpy()

    if points.shape[-1] != 3:
        # place channel dim last
        points = np.permute_dims(points, (1, 2, 0))
    points = points.reshape((-1, 3))

    ccf_pts = apply_transforms_to_points(
        points=points,
        affine_path=ls_template_to_ccf_affine_path,
        warp=ls_template_to_ccf_inverse_warp,
        template_parameters=ccf_template_parameters,
        crop_warp_to_bounding_box=False
    )

    ccf_pts = utils.convert_from_ants_space(
        ccf_template_parameters, ccf_pts
    )

    _, swapped, _ = utils.get_orientation_transform(
        ls_template_parameters.orientation,
        ccf_template_parameters.orientation,
    )

    ccf_pts = ccf_pts[:, swapped]

    return ccf_pts
