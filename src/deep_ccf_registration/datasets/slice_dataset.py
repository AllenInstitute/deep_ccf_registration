import json
import random
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

from deep_ccf_registration.utils.logging_utils import timed, timed_func
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore


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
    stitched_volume_path: str | Path
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registration_downsample: int
    ls_to_template_affine_matrix_path: Path
    # the inverse warp was converted to ome-zarr via `point_transformation_to_ome_zarr.py`
    ls_to_template_inverse_warp_path: str | Path
    # this is the original niftii inverse warp just in case
    ls_to_template_inverse_warp_path_original: Optional[Path] = None
    # The index that splits the 2 hemispheres in voxels the same dim as the sagittal axis in the registered volume
    # obtained via `get_input_space_midline.py`
    sagittal_midline: Optional[int] = None

def _create_coordinate_dataframe(
        patch_height: int,
        patch_width: int,
        start_x: int,
        start_y: int,
        fixed_index_value: int,
        slice_axis: AcquisitionAxis,
        axes: list[AcquisitionAxis]
) -> pd.DataFrame:
    """
    Create coordinate dataframe for a patch at specific position

    :param patch_height:
    :param patch_width:
    :param start_x:
    :param start_y:
    :param fixed_index_value:
    :param slice_axis:
    :param axes:
    :return:
    """
    # Create meshgrid with actual coordinates
    axis1_coords, axis2_coords = np.meshgrid(
        np.arange(start_x, start_x + patch_height),
        np.arange(start_y, start_y + patch_width),
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

    df = pd.DataFrame(data=points, columns=[x.name.value.lower() for x in axes]).astype(float)

    return df



def _prepare_grid_sample(warp: np.ndarray,
                         affine_transformed_voxels: np.ndarray
                         ) -> tuple[torch.Tensor, torch.Tensor]:
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


def _get_cropped_region_from_warp(warp: tensorstore.TensorStore | np.ndarray,
                                  affine_transformed_voxels: np.ndarray,
                                  warp_interpolation_padding: int = 5) -> np.ndarray:
    """
    This crops the warp to the region bounded by the min/max coordinates after applying the
    affine transformation. This is so that we don't have to load the entire warp, but only the
    region that we need.

    This also modifies affine_transformed_voxels inplace to set offset to 0 so that it can index into
    the cropped warp

    :param warp:
    :param affine_transformed_voxels: voxels after applying inverse affine to input points
    :param warp_interpolation_padding: padding around the min/max coords to crop for interpolation
    :return: cropped warp
    """
    warp_shape = np.array(warp.shape[:-1])

    min_coords = np.floor(affine_transformed_voxels.min(axis=0)).astype(
        int) - warp_interpolation_padding
    max_coords = np.ceil(affine_transformed_voxels.max(axis=0)).astype(
        int) + warp_interpolation_padding

    orig_min, orig_max = min_coords.copy(), max_coords.copy()

    # Clamp to warp dimensions
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, warp_shape)

    if np.any(max_coords <= min_coords):
        raise ValueError(
            f"Points are completely outside template bounds after affine transform.\n"
            f"Original bbox: {orig_min} to {orig_max}\n"
            f"After clamping: {min_coords} to {max_coords}\n"
            f"Warp shape: {warp_shape}\n"
            f"This indicates a registration or coordinate system error."
        )

    # Crop the warp
    if isinstance(warp, tensorstore.TensorStore):
        cropped_warp = warp[
                       min_coords[0]:max_coords[0],
                       min_coords[1]:max_coords[1],
                       min_coords[2]:max_coords[2]
                       ].read().result()
    else:
        cropped_warp = warp[
                       min_coords[0]:max_coords[0],
                       min_coords[1]:max_coords[1],
                       min_coords[2]:max_coords[2]
                       ]

    # Adjust voxel coordinates relative to cropped region
    affine_transformed_voxels -= min_coords

    return cropped_warp

@timed_func
def _apply_transforms_to_points(
        points: np.ndarray,
        experiment_meta: SubjectMetadata,
        warp: tensorstore.TensorStore | np.ndarray,
        template_parameters: AntsImageParameters,
        warp_interpolation_padding: int = 5,
        crop_warp_to_bounding_box: bool = True
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

    with timed():
        if crop_warp_to_bounding_box:
            warp = _get_cropped_region_from_warp(
                warp=warp,
                affine_transformed_voxels=affine_transformed_voxels,
                warp_interpolation_padding=warp_interpolation_padding
            )
        else:
            warp = warp[:].read().result()

    warp, affine_transformed_voxels = _prepare_grid_sample(
        warp=warp,
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


@timed_func
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

class TrainMode(Enum):
    TRAIN = 0
    TEST = 1

class SliceDataset(Dataset):
    """
    Loads a slice and the mapped points in template space
    """
    def __init__(self, dataset_meta: list[SubjectMetadata], ls_template: ants.ANTsImage,
                 orientation: Optional[SliceOrientation] = None,
                 registration_downsample_factor: int = 3,
                 tensorstore_aws_credentials_method: str = "default",
                 crop_warp_to_bounding_box: bool = True,
                 patch_size: Optional[tuple[int, int]] = (256, 256),
                 mode: TrainMode = TrainMode.TRAIN,
                 normalize_orientation_map: Optional[dict[SliceOrientation: list[AcquisitionDirection]]] = None,
                 limit_sagittal_slices_to_hemisphere: bool = False,
                 ):
        """

        :param dataset_meta: `list[SubjectMetadata]`
        :param ls_template: the smartSPIM light sheet template
        :param orientation: what orientation to load a slice
        :param registration_downsample_factor: downsample used during registration
        :param tensorstore_aws_credentials_method: credentials lookup method for tensorstore. see ts docs
        :param crop_warp_to_bounding_box: whether to load a cropped region of warp (faster) rather than full warp
        :param patch_size: patch size
        :param mode: `TrainMode``
        :param normalize_orientation_map: Map between slice axis and desired normalized orientation
            Example: {SliceOrientation.SAGITTAL: [AcquisitionDirection.Superior, AcquisitionDirection.Anterior]}. 3 different slices with orientations
            SAL, RPI, SPR.
            SA -> SA
            PI -> SA
            SP -> SA
        :param limit_sagittal_slices_to_hemisphere: Due to the symmetry of the brain, the model
            won't be able to differentiate sagittal slices from each hemisphere. Use this to limit
            sampling to the LEFT hemisphere
        """
        super().__init__()
        self._dataset_meta = dataset_meta
        if orientation is None:
            orientation = [SliceOrientation.SAGITTAL, SliceOrientation.CORONAL, SliceOrientation.HORIZONTAL]
        else:
            orientation = [orientation]
        self._orientation = orientation
        self._registration_downsample_factor = registration_downsample_factor
        self._warps = self._load_warps(tensorstore_aws_credentials_method=tensorstore_aws_credentials_method)
        self._crop_warp_to_bounding_box = crop_warp_to_bounding_box
        self._patch_size = patch_size
        self._mode = mode
        self._limit_sagittal_slices_to_hemisphere = limit_sagittal_slices_to_hemisphere

        if normalize_orientation_map is not None:
            for axis, orientation in normalize_orientation_map.items():
                if len(orientation) != 2:
                    raise ValueError('Orientation must be 2d for a 2d slice')
        self._normalize_orientation_map = normalize_orientation_map

        self._ls_template = ls_template

        if mode == TrainMode.TEST:
            # TODO Pre-compute all (volume_idx, slice_idx, patch_x, patch_y) combinations
            # self._patch_index = self._build_patch_index()
            pass

    def _build_patch_index(self):
        """Build index of all patches for inference"""
        raise NotImplementedError

    def _get_patch_positions(self, slice_shape):
        """Get all patch positions to tile the entire slice"""
        raise NotImplementedError

    def _load_warps(self, tensorstore_aws_credentials_method: str = "default") -> list[tensorstore.TensorStore]:
        warps = []
        for experiment_meta in self._dataset_meta:
            if (isinstance(experiment_meta.ls_to_template_inverse_warp_path, Path) and
                    experiment_meta.ls_to_template_inverse_warp_path.name.endswith('.nii.gz')):
                logger.info('Loading .nii.gz (slow!)')
                warp = ants.image_read(str(experiment_meta.ls_to_template_inverse_warp_path)).numpy()
            else:
                warp = tensorstore.open(
                    spec={
                        'driver': 'zarr3',
                        'kvstore': create_kvstore(
                            path=str(experiment_meta.ls_to_template_inverse_warp_path),
                            aws_credentials_method=tensorstore_aws_credentials_method
                        )
                    },
                    read=True
                ).result()
            warps.append(warp)
        return warps

    def _get_slice_from_idx(self, idx: int, orientation: SliceOrientation) -> tuple[int, int]:
        num_slices = self._get_num_slices_in_axis(
            orientation=orientation
        )
        num_slices_cumsum = np.cumsum([0] + num_slices)
        dataset_idx = int(np.searchsorted(num_slices_cumsum[1:], idx, side='right'))
        slice_idx = int(idx - num_slices_cumsum[dataset_idx])

        # For sagittal slices, adjust the slice index to sample from left hemisphere
        if orientation == SliceOrientation.SAGITTAL and self._limit_sagittal_slices_to_hemisphere:
                subject = self._dataset_meta[dataset_idx]
                sagittal_axis = self._get_slice_axis(
                    axes=subject.axes,
                    orientation=orientation
                )
                if sagittal_axis.direction == AcquisitionDirection.RIGHT_TO_LEFT:
                    # invert to get slice in left hemisphere
                    slice_idx = slice_idx + subject.sagittal_midline

        return dataset_idx, slice_idx

    def _get_num_slices_in_axis(self, orientation: SliceOrientation) -> list[int]:
        num_slices = []
        if orientation == SliceOrientation.SAGITTAL and self._limit_sagittal_slices_to_hemisphere:
            for subject in self._dataset_meta:
                sagittal_axis = self._get_slice_axis(
                    axes=subject.axes,
                    orientation=orientation
                )
                sagittal_dim = subject.registered_shape[sagittal_axis.dimension]
                # always sample from the left hemisphere due to brain symmetry,
                # the model would have no way to know which hemisphere a slice was sampled from
                if sagittal_axis.direction == AcquisitionDirection.LEFT_TO_RIGHT:
                    num_slices.append(subject.sagittal_midline)
                else:
                    num_slices.append(sagittal_dim - subject.sagittal_midline)
        else:
            num_slices = [x.registered_shape[self._get_slice_axis(axes=x.axes, orientation=orientation).dimension] for x in
                      self._dataset_meta]
        return num_slices

    def _get_slice_axis(self, axes: list[AcquisitionAxis], orientation: SliceOrientation) -> AcquisitionAxis:
        if orientation == SliceOrientation.SAGITTAL:
            slice_axis = [i for i in range(len(axes)) if
                          axes[i].direction in (AcquisitionDirection.LEFT_TO_RIGHT,
                                                AcquisitionDirection.RIGHT_TO_LEFT)]
            if len(slice_axis) != 1:
                raise ValueError(f'expected to find 1 sagittal axis but found {len(slice_axis)}')
            slice_axis = axes[slice_axis[0]]
        else:
            raise NotImplementedError(f'{self._orientation} not supported')
        return slice_axis

    @timed_func
    def __getitem__(self, idx):
        if self._mode == TrainMode.TRAIN:
            orientation = random.choice(self._orientation)
        else:
            if len(self._orientation) != 1:
                raise ValueError('Must provide single orientation if not train')
            orientation = self._orientation[0]

        dataset_idx, slice_idx = self._get_slice_from_idx(idx=idx, orientation=orientation)
        experiment_meta = self._dataset_meta[dataset_idx]
        acquisition_axes = experiment_meta.axes

        slice_axis = self._get_slice_axis(axes=acquisition_axes, orientation=orientation)

        volume = tensorstore.open(
            spec={
                'driver': 'auto',
                'kvstore': create_kvstore(
                    path=str(experiment_meta.stitched_volume_path) + f'/{self._registration_downsample_factor}',
                    aws_credentials_method="anonymous"
                )
            },
            read=True
        ).result()


        volume_slice = [0, 0, slice(None), slice(None), slice(None)]
        volume_slice[slice_axis.dimension + 2] = slice_idx  # +2 because first 2 axes unused

        with timed():
            if self._mode == TrainMode.TRAIN:
                input_slice, patch_x, patch_y = self._get_random_patch(
                    slice_2d=volume[tuple(volume_slice)]
                )
            else:
                if self._patch_size is None:
                    input_slice, patch_x, patch_y = volume[tuple(volume_slice)][:].read().result(), 0, 0
                else:
                    raise NotImplementedError
        height, width = input_slice.shape

        point_grid = _create_coordinate_dataframe(
            patch_height=height,
            patch_width=width,
            start_x=patch_x,
            start_y=patch_y,
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
            warp=self._warps[dataset_idx],
            crop_warp_to_bounding_box=self._crop_warp_to_bounding_box
        )

        output_points = ls_template_points.values.reshape((height, width, 3))

        if self._normalize_orientation_map is not None:
            input_slice, output_points = self._normalize_orientation(
                slice=input_slice,
                template_points=output_points,
                acquisition_axes=acquisition_axes,
                orientation=orientation,
                slice_axis=slice_axis
            )

        return input_slice, output_points, dataset_idx, slice_idx

    def __len__(self):
        num_slices = 0
        for orientation in self._orientation:
            num_slices_in_axis = self._get_num_slices_in_axis(
                orientation=orientation
            )
            num_slices += sum(num_slices_in_axis)
        return num_slices

    def _get_random_patch(self, slice_2d: tensorstore.TensorStore):
        """Extract random patch from slice"""
        h, w = slice_2d.shape
        ph, pw = self._patch_size

        # Adjust patch size to what's available
        ph = min(h, ph)
        pw = min(w, pw)

        # Random position (0 if slice is smaller than patch)
        x = random.randint(0, max(0, h - ph))
        y = random.randint(0, max(0, w - pw))

        # Extract what we can
        patch = torch.from_numpy(slice_2d[x:x + ph, y:y + pw].read().result())

        # Pad to patch_size if needed
        patch = self._pad_patch_to_size(patch)

        return patch, x, y

    def _pad_patch_to_size(self, patch):
        """Pad extracted patch to patch_size if needed"""
        h, w = patch.shape
        ph, pw = self._patch_size

        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)

        if pad_h > 0 or pad_w > 0:
            patch = torch.nn.functional.pad(
                patch,
                (0, pad_w, 0, pad_h),
                mode='constant',
                value=0
            )

        return patch

    def _normalize_orientation(
        self,
        slice: np.ndarray,
        template_points: np.ndarray,
        acquisition_axes: list[AcquisitionAxis],
        orientation: SliceOrientation,
        slice_axis: AcquisitionAxis,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms slice and template points to have a uniform
        orientation

        :param slice:
        :param template_points:
        :param acquisition_axes:
        :param orientation:
        :param slice_axis:
        :return: transformed slice and template_points
        """
        desired_orientation: list[AcquisitionDirection] = self._normalize_orientation_map[orientation]
        acquisition_axes = sorted(acquisition_axes, key=lambda x: x.dimension)

        desired_orientation = desired_orientation.copy()
        desired_orientation.insert(slice_axis.dimension, slice_axis.direction)

        _, swapped, mat = aind_smartspim_transform_utils.utils.utils.get_orientation_transform(
            orientation_in=''.join([x.direction.value.lower()[0] for x in acquisition_axes]),
            orientation_out=''.join([x.value.lower()[0] for x in desired_orientation])
        )

        # exclude the slice axis, since just dealing with 2d slices
        mat = mat[[x for x in range(3) if x != slice_axis.dimension]]
        mat = mat[:, [x for x in range(3) if x != slice_axis.dimension]]

        # flip axis to desired orientation
        for idx, dim_orient in enumerate(mat.sum(axis=1)):
            if dim_orient < 0:
                if idx == 0:
                    slice = np.flipud(slice)
                    template_points = np.flipud(template_points)
                else:
                    slice = np.fliplr(slice)
                    template_points = np.fliplr(template_points)

        if swapped.tolist() != range(3):
            slice = np.transpose(slice)
            template_points = np.permute_dims(template_points, axes=[1, 0, 2])

        return slice, template_points

