import json
from enum import Enum
from pathlib import Path
from typing import Optional

import aind_smartspim_transform_utils
import ants
import numpy as np
import pandas as pd
import tensorstore
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection, \
    apply_transforms_to_points, convert_from_ants_space, AcqusitionAxesName
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import Dataset
import torch.nn.functional as F


class SliceOrientation(Enum):
    SAGITTAL = 'sagittal'
    CORONAL = 'coronal'
    HORIZONTAL = 'horizontal'

class AxisResolution(BaseModel):
    value: float
    unit: str

class AcquisitionAxis(BaseModel):
    dimension: int
    direction: AcquisitionDirection
    name: AcqusitionAxesName
    unit: str
    resolution: AxisResolution

class SubjectMetadata(BaseModel):
    subject_id: str
    stitched_volume_path: Path
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registered_resolution: tuple[float, float, float]
    ls_to_template_affine_matrix_path: Path
    ls_to_template_inverse_warp_path: Path


def _create_coordinate_dataframe(height: int, width: int, fixed_index_value: int, slice_axis: AcquisitionAxis, axes: list[AcquisitionAxis]) -> pd.DataFrame:
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten the coordinate arrays
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()

    slice_index = np.full(len(y_flat), fixed_index_value)

    axes = sorted([x for x in axes if x != slice_axis], key=lambda x: x.dimension)

    df = pd.DataFrame({
        slice_axis.name.value.lower(): slice_index,
        axes[0].name.value.lower(): y_flat,
        axes[1].name.value.lower(): x_flat
    }).astype(float)

    return df


def _prepare_grid_sample(warp: np.ndarray, affine_transformed_voxels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This takes the affine_transformed_voxels and warp and converts to a format suitable
    for grid_sample

    :return:
    """
    warp_shape = warp.shape

    # Convert warp to torch tensor with shape (1, 3, D, H, W)
    # grid_sample expects (batch, channels, depth, height, width)
    warp = torch.from_numpy(warp)
    warp = warp.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, D, H, W)

    # Convert voxel indices to normalized coordinates [-1, 1]
    # grid_sample expects coordinates in (x, y, z) order for the last dimension
    warp_shape = np.array(warp_shape[:3])
    normalized_affine_transformed_voxels = 2.0 * affine_transformed_voxels / (warp_shape - 1) - 1.0

    # grid_sample expects coordinates in (W, H, D) order, but our voxels are in (D, H, W)
    # So we need to reorder: [D, H, W] -> [W, H, D]
    normalized_affine_transformed_voxels = normalized_affine_transformed_voxels[:, [2, 1, 0]]  # Reorder to (W, H, D)

    # Reshape for grid_sample: (1, N, 1, 1, 3) for 3D sampling
    # where N is the number of points
    n_points = len(affine_transformed_voxels)
    normalized_affine_transformed_voxels = torch.from_numpy(normalized_affine_transformed_voxels)

    if warp.dtype == torch.float16:
        normalized_affine_transformed_voxels = normalized_affine_transformed_voxels.half()
    else:
        normalized_affine_transformed_voxels = normalized_affine_transformed_voxels.float()

    normalized_affine_transformed_voxels = normalized_affine_transformed_voxels.reshape(1, n_points, 1, 1, 3)

    return warp, normalized_affine_transformed_voxels

def _apply_transforms_to_points(
        points: np.ndarray,
        experiment_meta: SubjectMetadata,
        warp: tensorstore.TensorStore | np.ndarray,
        template_parameters: AntsImageParameters,
):
    # apply inverse affine to points in input space
    # this returns points in physical space
    affine_transformed_points = apply_transforms_to_points(
        ants_pts=points,
        transforms=[str(experiment_meta.ls_to_template_affine_matrix_path)],
        invert=(True,)
    )

    # convert physical points to voxels,
    # so we can index into the displacement field
    affine_transformed_voxels = convert_from_ants_space(
        template_parameters=template_parameters,
        physical_pts=affine_transformed_points
    )

    warp, affine_transformed_voxels = _prepare_grid_sample(
        warp=warp[:].read().result() if isinstance(warp, tensorstore.TensorStore) else warp,
        affine_transformed_voxels=affine_transformed_voxels
    )

    displacements = F.grid_sample(
        input=warp,
        grid=affine_transformed_voxels,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # (1, 3, N, 1, 1) -> (N, 3)
    displacements = displacements.squeeze().T.numpy()

    # apply displacement vector to affine transformed points
    transformed_points = affine_transformed_points + displacements

    transformed_points = convert_from_ants_space(
        template_parameters, transformed_points
    )

    transformed_df = pd.DataFrame(
        transformed_points, columns=["ML", "AP", "DV"]
    )
    return transformed_df


def _transform_points_to_template_ants_space(
    acquisition_axes: list[AcquisitionAxis],
    ls_template_info: AntsImageParameters,
    points: pd.DataFrame,
    points_resolution: list[float],
    input_volume_shape: tuple[int, int, int],
    template_resolution: int = 25,
) -> np.ndarray:
    # order columns to align with imaging
    col_order = ["", "", ""]
    for dim in acquisition_axes:
        col_order[dim.dimension] = dim.name.value.lower()

    points = points[col_order].values

    # flip axis based on the template orientation relative to input image
    orient = aind_smartspim_transform_utils.utils.utils.get_orientation([json.loads(x.model_dump_json()) for x in acquisition_axes])

    _, swapped, mat = aind_smartspim_transform_utils.utils.utils.get_orientation_transform(
        orient, ls_template_info.orientation
    )

    for idx, dim_orient in enumerate(mat.sum(axis=1)):
        if dim_orient < 0:
            points[:, idx] = input_volume_shape[idx] - points[:, idx]

    # scale points and orient axes to template
    scaling = [res_1 / res_2 for res_1, res_2 in zip(points_resolution, [template_resolution] * 3)]
    scaled_pts = aind_smartspim_transform_utils.utils.utils.scale_points(points, scaling)
    orient_pts = scaled_pts[:, swapped]

    # convert points into ccf space
    ants_pts = aind_smartspim_transform_utils.utils.utils.convert_to_ants_space(
        ls_template_info, orient_pts
    )
    return ants_pts

class SliceDataset(Dataset):
    def __init__(self, dataset_meta: list[SubjectMetadata], ls_template_path: Path,
                 orientation: Optional[SliceOrientation] = None,
                 registration_downsample_factor: int = 3):
        super().__init__()
        self._dataset_meta = dataset_meta
        self._orientation = orientation
        self._registration_downsample_factor = registration_downsample_factor
        self._warps = self._load_warps()

        logger.info('Loading light sheet template')
        self._ls_template = ants.image_read(str(ls_template_path))

    def _load_warps(self) -> list[tensorstore.TensorStore]:
        warps = []
        for experiment_meta in self._dataset_meta:
            if experiment_meta.ls_to_template_inverse_warp_path.name.endswith('.nii.gz'):
                logger.info('Loading .nii.gz (slow!)')
                warp = ants.image_read(str(experiment_meta.ls_to_template_inverse_warp_path)).numpy()
            else:
                warp = tensorstore.open(
                    spec={
                        'driver': 'zarr3',
                        'kvstore': {
                            'driver': 'file',
                            'path': str(experiment_meta.ls_to_template_inverse_warp_path)
                        }
                    },
                    read=True
                ).result()
            warps.append(warp)
        return warps

    def _get_slice_from_idx(self, idx: int) -> tuple[int, int]:
        num_slices = [x.registered_shape[self._get_slice_axis(axes=x.axes).dimension] for x in
                      self._dataset_meta]
        num_slices_cumsum = np.cumsum([0] + num_slices)
        dataset_idx = int(np.searchsorted(num_slices_cumsum[1:], idx, side='right'))
        slice_idx = int(idx - num_slices_cumsum[dataset_idx])
        return dataset_idx, slice_idx

    def _get_slice_axis(self, axes: list[AcquisitionAxis]) -> AcquisitionAxis:
        if self._orientation == SliceOrientation.SAGITTAL:
            slice_axis = [i for i in range(len(axes)) if
                          axes[i].direction in (AcquisitionDirection.LEFT_TO_RIGHT,
                                                AcquisitionDirection.RIGHT_TO_LEFT)]
            if len(slice_axis) != 1:
                raise ValueError(f'expected to find 1 sagittal axis but found {len(slice_axis)}')
            slice_axis = axes[slice_axis[0]]
        else:
            raise NotImplementedError(f'{self._orientation} not supported')
        return slice_axis

    def __getitem__(self, idx):
        dataset_idx, slice_idx = self._get_slice_from_idx(idx=idx)
        experiment_meta = self._dataset_meta[dataset_idx]
        acquisition_axes = experiment_meta.axes

        volume = tensorstore.open(
            spec={
                'driver': 'file',
                'path': str(experiment_meta.stitched_volume_path / str(
                    self._registration_downsample_factor))
            },
            read=True
        ).result()

        slice_axis = self._get_slice_axis(axes=acquisition_axes)
        height, width = [experiment_meta.registered_shape[i] for i in range(3) if i != slice_axis.dimension]

        point_grid = _create_coordinate_dataframe(
            height=height,
            width=width,
            fixed_index_value=slice_idx,
            axes=experiment_meta.axes,
            slice_axis=slice_axis
        )

        points = _transform_points_to_template_ants_space(
            points=point_grid,
            points_resolution=list(experiment_meta.registered_resolution),
            input_volume_shape=volume.shape[2:],
            acquisition_axes=experiment_meta.axes,
            ls_template_info=AntsImageParameters.from_ants_image(image=self._ls_template)
        )

        ls_template_points = _apply_transforms_to_points(
            points=points,
            template_parameters=AntsImageParameters.from_ants_image(image=self._ls_template),
            experiment_meta=experiment_meta,
            warp=self._warps[dataset_idx]
        )

        volume_slice = [0, 0, slice(None), slice(None), slice(None)]
        volume_slice[slice_axis.dimension + 2] = slice_idx

        input_slice = volume[tuple(volume_slice)].read().result()

        output_points = ls_template_points.values.reshape((height, width, 3))
        return input_slice, output_points, dataset_idx, slice_idx

    def __len__(self):
        num_slices = [x.registered_shape[self._get_slice_axis(axes=x.axes).dimension] for x in
                      self._dataset_meta]
        return sum(num_slices)
