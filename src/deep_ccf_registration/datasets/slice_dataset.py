import datetime
import json
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

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

class AcquisitionAxis(BaseModel):
    dimension: int
    direction: AcquisitionDirection
    name: AcqusitionAxesName
    unit: str
    resolution: float

class SubjectMetadata(BaseModel):
    subject_id: str
    stitched_volume_path: Path
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registration_downsample: int
    ls_to_template_affine_matrix_path: Path
    ls_to_template_inverse_warp_path: Path
    #registration_date: datetime.datetime


def _create_coordinate_dataframe(height: int, width: int, fixed_index_value: int, slice_axis: AcquisitionAxis, axes: list[AcquisitionAxis]) -> pd.DataFrame:
    axis1_coords, axis2_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    axis1_flat = axis1_coords.flatten()
    axis2_flat = axis2_coords.flatten()

    n_points = len(axis1_flat)

    slice_index = np.full(n_points, fixed_index_value)

    axes = sorted(axes, key=lambda x: x.dimension)

    points = np.zeros((n_points, 3))

    points[:, slice_axis.dimension] = slice_index
    points[:, [x for x in axes if x != slice_axis][0].dimension] = axis1_flat
    points[:, [x for x in axes if x != slice_axis][1].dimension] = axis2_flat

    df = pd.DataFrame(data=points, columns=[x.name.value.lower() for x in axes]).astype(float)

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


def _create_kvstore(path: str, aws_credentials_method: str = "default"):
    """
    Create tensorstore kvstore

    Parameters
    ----------
    path
    aws_credentials_method

    Returns
    -------

    """

    def parse_s3_uri(s3_uri):
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key

    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(s3_uri=path)
        kvstore = {
            "driver": "s3",
            "bucket": bucket,
            "path": key,
            "aws_credentials": {"type": aws_credentials_method},
        }
    else:
        kvstore = {"driver": "file", "path": path}
    return kvstore

def _transform_points_to_template_ants_space(
    acquisition_axes: list[AcquisitionAxis],
    ls_template_info: AntsImageParameters,
    points: pd.DataFrame,
    input_volume_shape: tuple[int, int, int],
    template_resolution: int = 25,
    registration_downsample: float = 3.0,
) -> np.ndarray:
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

class SliceDataset(Dataset):
    def __init__(self, dataset_meta: list[SubjectMetadata], ls_template: ants.ANTsImage,
                 orientation: Optional[SliceOrientation] = None,
                 registration_downsample_factor: int = 3,
                 tensorstore_aws_credentials_method: str = "default"
                 ):
        super().__init__()
        self._dataset_meta = dataset_meta
        self._orientation = orientation
        self._registration_downsample_factor = registration_downsample_factor
        self._warps = self._load_warps(tensorstore_aws_credentials_method=tensorstore_aws_credentials_method)

        self._ls_template = ls_template

    def _load_warps(self, tensorstore_aws_credentials_method: str = "default") -> list[tensorstore.TensorStore]:
        warps = []
        for experiment_meta in self._dataset_meta:
            if experiment_meta.ls_to_template_inverse_warp_path.name.endswith('.nii.gz'):
                logger.info('Loading .nii.gz (slow!)')
                warp = ants.image_read(str(experiment_meta.ls_to_template_inverse_warp_path)).numpy()
            else:
                warp = tensorstore.open(
                    spec={
                        'driver': 'zarr3',
                        'kvstore': _create_kvstore(
                            path=str(experiment_meta.ls_to_template_inverse_warp_path),
                            aws_credentials_method=tensorstore_aws_credentials_method
                        )
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
            input_volume_shape=volume.shape[2:],
            acquisition_axes=experiment_meta.axes,
            ls_template_info=AntsImageParameters.from_ants_image(image=self._ls_template),
            registration_downsample=experiment_meta.registration_downsample
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
