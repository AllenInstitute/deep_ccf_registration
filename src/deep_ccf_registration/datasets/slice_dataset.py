from enum import Enum
from pathlib import Path
from typing import Optional

import ants
import numpy as np
import pandas as pd
import tensorstore
import torch
from aind_smartspim_transform_utils.CoordinateTransform import CoordinateTransform
from aind_smartspim_transform_utils.utils.utils import AcquisitionAxis, AcquisitionDirection, \
    apply_transforms_to_points, convert_from_ants_space
from loguru import logger
from pydantic import BaseModel
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset
import torch.nn.functional as F


class SliceOrientation(Enum):
    SAGITTAL = 'sagittal'
    CORONAL = 'coronal'
    HORIZONTAL = 'horizontal'


class ExperimentMetadata(BaseModel):
    experiment_id: str
    stitched_volume_path: Path
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registered_resolution: tuple[float, float, float]
    ls_to_template_affine_matrix_path: Path
    ls_to_template_inverse_warp_path: Path


def _create_coordinate_dataframe(height: int, width: int, fixed_index_value: int) -> pd.DataFrame:
    """
    Create a DataFrame with all pixel coordinates from a slice.

    Parameters:
    height: Height of the image (y dimension)
    width: Width of the image (x dimension)
    fixed_index_value: Fixed index value

    Returns:
    pd.DataFrame: DataFrame with columns [z, y, x]
    """
    # Create coordinate arrays
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten the coordinate arrays
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()

    z_flat = np.full(len(y_flat), fixed_index_value)

    df = pd.DataFrame({
        'z': z_flat,
        'y': y_flat,
        'x': x_flat
    }).astype(float)

    return df


def _apply_transforms_to_points(
        points: np.ndarray,
        coord_transform: CoordinateTransform,
        experiment_meta: ExperimentMetadata,
        warp: tensorstore.TensorStore
):
    warp_shape = warp.shape

    # apply inverse affine to points in input space
    affine_transformed_points = apply_transforms_to_points(
        ants_pts=points,
        transforms=[str(experiment_meta.ls_to_template_affine_matrix_path)],
        invert=(True,)
    )

    # convert physical points to voxels,
    # so we can index into the displacement field
    voxel_indices = convert_from_ants_space(
        template_parameters=coord_transform.ls_template_info,
        physical_pts=affine_transformed_points
    )

    # Convert warp to torch tensor with shape (1, 3, D, H, W)
    # grid_sample expects (batch, channels, depth, height, width)
    warp = torch.from_numpy(warp[:].read().result())
    warp = warp.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, D, H, W)

    # Convert voxel indices to normalized coordinates [-1, 1]
    # grid_sample expects coordinates in (x, y, z) order for the last dimension
    warp_shape = np.array(warp_shape[:3])
    normalized_coords = 2.0 * voxel_indices / (warp_shape - 1) - 1.0

    # grid_sample expects coordinates in (W, H, D) order, but our voxels are in (D, H, W)
    # So we need to reorder: [D, H, W] -> [W, H, D]
    normalized_coords = normalized_coords[:, [2, 1, 0]]  # Reorder to (W, H, D)

    # Reshape for grid_sample: (1, N, 1, 1, 3) for 3D sampling
    # where N is the number of points
    n_points = len(normalized_coords)
    grid = torch.from_numpy(normalized_coords)

    if warp.dtype == torch.float16:
        grid = grid.half()
    else:
        grid = grid.float()

    grid = grid.reshape(1, n_points, 1, 1, 3)

    sampled = F.grid_sample(
        input=warp,
        grid=grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Extract displacements: (1, 3, N, 1, 1) -> (N, 3)
    displacements = sampled.squeeze().T.numpy()

    # apply displacement vector to affine transformed points
    transformed_points = affine_transformed_points + displacements

    transformed_points = convert_from_ants_space(
        coord_transform.ls_template_info, transformed_points
    )

    transformed_df = pd.DataFrame(
        transformed_points, columns=["ML", "AP", "DV"]
    )
    return transformed_df


class SliceDataset(Dataset):
    def __init__(self, dataset_meta: list[ExperimentMetadata], ls_template_path: Path,
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

        slice_axis = self._get_slice_axis(axes=acquisition_axes)
        height = experiment_meta.registered_shape[
            [x.dimension for x in acquisition_axes if x.name != slice_axis.name][0]]
        width = experiment_meta.registered_shape[
            [x.dimension for x in acquisition_axes if x.name != slice_axis.name][1]]

        point_grid = _create_coordinate_dataframe(
            height=height,
            width=width,
            fixed_index_value=slice_idx
        )

        coord_transform = CoordinateTransform(
            name='smartspim_lca',
            dataset_transforms={
                'points_to_ccf': [
                    str(experiment_meta.ls_to_template_affine_matrix_path),
                    str(experiment_meta.ls_to_template_inverse_warp_path),
                ]
            },
            acquisition_axes=acquisition_axes,
            image_metadata={'shape': experiment_meta.registered_shape},
            ls_template=self._ls_template
        )

        points = coord_transform.prepare_points_for_forward_transform(
            points=point_grid,
            points_resolution=list(experiment_meta.registered_resolution)
        )

        ls_template_points = _apply_transforms_to_points(
            points=points,
            coord_transform=coord_transform,
            experiment_meta=experiment_meta,
            warp=self._warps[dataset_idx]
        )

        volume_slice = [0, 0, slice(None), slice(None), slice(None)]
        volume_slice[slice_axis.dimension + 2] = slice_idx

        volume = tensorstore.open(
            spec={
                'driver': 'file',
                'path': str(experiment_meta.stitched_volume_path / str(
                    self._registration_downsample_factor))
            },
            read=True
        ).result()
        input_slice = volume[tuple(volume_slice)].read().result()

        output_points = ls_template_points.values.reshape((height, width, 3))
        return input_slice, output_points, dataset_idx, slice_idx

    def __len__(self):
        num_slices = [x.registered_shape[self._get_slice_axis(axes=x.axes).dimension] for x in
                      self._dataset_meta]
        return len(self._dataset_meta) * sum(num_slices)
