import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import aind_smartspim_transform_utils
import albumentations
import ants
import numpy as np
import pandas as pd
import tensorstore
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection
from loguru import logger
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset

from deep_ccf_registration.metadata import AcquisitionAxis, SubjectMetadata, SliceOrientation
from deep_ccf_registration.utils.logging_utils import timed, timed_func
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore
from deep_ccf_registration.utils.transforms import transform_points_to_template_ants_space, \
    apply_transforms_to_points


@dataclass
class Patch:
    # subject index
    dataset_idx: int
    # slice index within subject
    slice_idx: int
    # patch start x
    x: int
    # patch start y
    y: int
    # patch orientation
    orientation: SliceOrientation

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
    Create coordinate dataframe for a patch at specific position.

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

    df = pd.DataFrame(data=points, columns=[x.name.value.lower() for x in axes]).astype(float)

    return df

class TrainMode(Enum):
    TRAIN = 0
    TEST = 1

class SliceDataset(Dataset):
    """
    Dataset for loading slices and their mapped points in template space.

    This dataset loads 2D slices from 3D volumes along with their corresponding
    coordinates in template space after applying registration transformations.
    """
    def __init__(self, dataset_meta: list[SubjectMetadata], ls_template_parameters: AntsImageParameters,
                 orientation: Optional[SliceOrientation] = None,
                 registration_downsample_factor: int = 3,
                 tensorstore_aws_credentials_method: str = "default",
                 crop_warp_to_bounding_box: bool = True,
                 patch_size: Optional[tuple[int, int]] = (256, 256),
                 mode: TrainMode = TrainMode.TRAIN,
                 normalize_orientation_map: Optional[dict[
                                                     SliceOrientation: list[AcquisitionDirection]]] = None,
                 limit_sagittal_slices_to_hemisphere: bool = False,
                 input_image_transforms: Optional[list[albumentations.BasicTransform]] = None,
                 output_points_transforms: Optional[list[albumentations.BasicTransform]] = None,
                 tissue_mask_transforms: Optional[list[albumentations.BasicTransform]] = None,
                 ):
        """
        Initialize SliceDataset.

        Parameters
        ----------
        dataset_meta : list[SubjectMetadata]
            List of subject metadata for all subjects in the dataset.
        ls_template_parameters : AntsImageParameters
            Light sheet template parameters
        orientation : SliceOrientation, optional
            What orientation to load slices. If None, loads all orientations
            (SAGITTAL, CORONAL, HORIZONTAL).
        registration_downsample_factor : int, default=3
            Downsample factor used during registration.
        tensorstore_aws_credentials_method : str, default="default"
            Credentials lookup method for tensorstore. See tensorstore documentation.
        crop_warp_to_bounding_box : bool, default=True
            Whether to load a cropped region of warp (faster) rather than full warp.
        patch_size : tuple[int, int], optional, default=(256, 256)
            Patch size (height, width). If None, returns full slice.
        mode : TrainMode, default=TrainMode.TRAIN
            Dataset mode (TRAIN or TEST).
        normalize_orientation_map : dict[SliceOrientation, list[AcquisitionDirection]], optional
            Map between slice axis and desired normalized orientation.
            Example: {SliceOrientation.SAGITTAL: [AcquisitionDirection.Superior,
            AcquisitionDirection.Anterior]}. For 3 different slices with orientations
            SAL, RPI, SPR:
                SA -> SA
                PI -> SA
                SP -> SA
        limit_sagittal_slices_to_hemisphere : bool, default=False
            Due to the symmetry of the brain, the model won't be able to differentiate
            sagittal slices from each hemisphere. Use this to limit sampling to the
            LEFT hemisphere.
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
        self._input_image_transforms = input_image_transforms
        self._output_points_transforms = output_points_transforms
        self._tissue_mask_transforms = tissue_mask_transforms

        if normalize_orientation_map is not None:
            for axis, orientation in normalize_orientation_map.items():
                if len(orientation) != 2:
                    raise ValueError('Orientation must be 2d for a 2d slice')
        self._normalize_orientation_map = normalize_orientation_map

        self._ls_template_parameters = ls_template_parameters
        self._precomputed_patches = self._build_patch_index() if mode == TrainMode.TEST and patch_size is not None else None

    @property
    def patch_size(self) -> Optional[tuple[int, int]]:
        return self._patch_size

    def _get_patch_positions(self, slice_shape: tuple[int, int]):
        """
        Get all patch positions to tile the entire slice.

        Parameters
        ----------
        slice_shape : tuple[int, int]
            Shape of the slice (height, width).

        Returns
        -------
        list[tuple[int, int]]
            List of (y, x) positions for patch extraction.
        """
        if self._patch_size is None:
            # Return single patch covering whole slice
            return [(0, 0)]

        h, w = slice_shape
        ph, pw = self._patch_size

        positions = []

        # Calculate number of patches needed (with overlap to cover edges)
        for y in range(0, h, ph):
            for x in range(0, w, pw):
                # Adjust position if it would go past the edge
                y_start = min(y, max(0, h - ph))
                x_start = min(x, max(0, w - pw))

                # Avoid duplicate patches at the edge
                if not positions or (y_start, x_start) != positions[-1]:
                    positions.append((y_start, x_start))

        return positions

    def _load_warps(self, tensorstore_aws_credentials_method: str = "default") -> list[tensorstore.TensorStore]:
        """
        Load displacement fields for all subjects in the dataset.

        Parameters
        ----------
        tensorstore_aws_credentials_method : str, default="default"
            Credentials lookup method for tensorstore.

        Returns
        -------
        list[tensorstore.TensorStore]
            List of displacement field arrays/tensorstores for each subject.
        """
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
        """
        Convert global slice index to dataset and slice indices.

        Parameters
        ----------
        idx : int
            Global slice index across all subjects.
        orientation : SliceOrientation
            Slice orientation.

        Returns
        -------
        tuple[int, int]
            Tuple of (dataset_idx, slice_idx) identifying the specific slice.
        """
        num_slices = self._get_num_slices_in_axis(
            orientation=orientation
        )
        num_slices_cumsum = np.cumsum([0] + num_slices)
        dataset_idx = int(np.searchsorted(num_slices_cumsum[1:], idx, side='right'))
        slice_idx = int(idx - num_slices_cumsum[dataset_idx])

        # For sagittal slices, adjust the slice index to sample from left hemisphere
        if orientation == SliceOrientation.SAGITTAL and self._limit_sagittal_slices_to_hemisphere:
                subject = self._dataset_meta[dataset_idx]
                sagittal_axis = subject.get_slice_axis(
                    orientation=orientation
                )
                if sagittal_axis.direction == AcquisitionDirection.RIGHT_TO_LEFT:
                    # invert to get slice in left hemisphere
                    slice_idx = slice_idx + subject.sagittal_midline

        return dataset_idx, slice_idx

    def _get_slice_range(self, subject: SubjectMetadata, orientation: SliceOrientation) -> list[int]:
        """
        Get the range of slice indices for a given subject and orientation.

        Parameters
        ----------
        subject : SubjectMetadata
            Subject metadata.
        orientation : SliceOrientation
            Slice orientation.

        Returns
        -------
        list of valid slice indices.
        """
        slice_axis = subject.get_slice_axis(orientation=orientation)
        if orientation == SliceOrientation.SAGITTAL and self._limit_sagittal_slices_to_hemisphere:
            sagittal_dim = subject.registered_shape[slice_axis.dimension]
            # Always sample from the left hemisphere due to brain symmetry
            if slice_axis.direction == AcquisitionDirection.LEFT_TO_RIGHT:
                slices = list(range(subject.sagittal_midline))
            else:
                slices = list(range(subject.sagittal_midline, sagittal_dim))
        else:
            slices = list(range(subject.registered_shape[slice_axis.dimension]))
        return slices

    def _get_num_slices_in_axis(self, orientation: SliceOrientation) -> list[int]:
        """
        Get number of slices per subject for given orientation.

        Parameters
        ----------
        orientation : SliceOrientation
            Slice orientation.

        Returns
        -------
        list[int]
            List of number of slices for each subject.
        """
        return [len(self._get_slice_range(subject, orientation)) for subject in self._dataset_meta]

    def _build_patch_index(self) -> list[Patch]:
        """
        Build index of all patches for inference.

        Creates a comprehensive list of all patches across all slices and subjects
        for deterministic iteration during test mode.

        Returns
        -------
        list[Patch]
            List of Patch objects specifying dataset_idx, slice_idx, x, y, and orientation.
        """
        patch_index = []

        for orientation in self._orientation:
            for dataset_idx, subject_meta in enumerate(self._dataset_meta):
                # Get slice range for this subject
                slice_range = self._get_slice_range(subject_meta, orientation)

                for slice_idx in slice_range:
                    slice_shape = subject_meta.get_slice_shape(
                        orientation=orientation
                    )

                    # Get all patch positions for this slice
                    patch_positions = self._get_patch_positions(slice_shape)

                    for patch_y, patch_x in patch_positions:
                        patch_index.append(
                            Patch(
                                dataset_idx=dataset_idx,
                                slice_idx=slice_idx,
                                x=patch_x,
                                y=patch_y,
                                orientation=orientation
                            )
                        )

        return patch_index

    @timed_func
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            Tuple containing:
            - input_slice : torch.Tensor
                2D slice from the input volume.
            - output_points : np.ndarray
                3D array of shape (height, width, 3) containing template coordinates.
            - dataset_idx : int
                Index of the subject/dataset.
            - slice_idx : int
                Index of the slice within the subject.
        """
        orientation = random.choice(self._orientation)
        dataset_idx, slice_idx = self._get_slice_from_idx(idx=idx, orientation=orientation)

        if self._precomputed_patches is not None:
            patch_x = self._precomputed_patches[idx].x
            patch_y = self._precomputed_patches[idx].y
        else:
            if self._patch_size is None:
                patch_x, patch_y = 0, 0
            else:
                patch_x, patch_y = None, None

        experiment_meta = self._dataset_meta[dataset_idx]
        acquisition_axes = experiment_meta.axes
        slice_axis = experiment_meta.get_slice_axis(orientation=orientation)

        # Load volume and extract patch
        volume = tensorstore.open(
            spec={
                'driver': 'auto',
                'kvstore': create_kvstore(
                    path=str(
                        experiment_meta.stitched_volume_path) + f'/{self._registration_downsample_factor}',
                    aws_credentials_method="anonymous"
                )
            },
            read=True
        ).result()

        volume_slice = [0, 0, slice(None), slice(None), slice(None)]
        volume_slice[slice_axis.dimension + 2] = slice_idx  # + 2 since first 2 dims unused

        with timed():
            if self._patch_size is None:
                input_image = volume[tuple(volume_slice)][:].read().result()
            else:
                input_image, patch_y, patch_x = self._get_patch(
                    slice_2d=volume[tuple(volume_slice)],
                    patch_x=patch_x,
                    patch_y=patch_y
                )

        height, width = input_image.shape

        point_grid = _create_coordinate_dataframe(
            patch_height=height,
            patch_width=width,
            start_x=patch_x,
            start_y=patch_y,
            fixed_index_value=slice_idx,
            axes=experiment_meta.axes,
            slice_axis=slice_axis
        )

        points = transform_points_to_template_ants_space(
            points=point_grid,
            input_volume_shape=volume.shape[2:],
            acquisition_axes=experiment_meta.axes,
            ls_template_info=self._ls_template_parameters,
            registration_downsample=experiment_meta.registration_downsample
        )

        ls_template_points = apply_transforms_to_points(
            points=points,
            template_parameters=self._ls_template_parameters,
            affine_path=experiment_meta.ls_to_template_affine_matrix_path,
            warp=self._warps[dataset_idx],
            crop_warp_to_bounding_box=self._crop_warp_to_bounding_box
        )

        output_points = ls_template_points.reshape((height, width, 3))

        if self._normalize_orientation_map is not None:
            input_image, output_points = self._normalize_orientation(
                slice=input_image,
                template_points=output_points,
                acquisition_axes=acquisition_axes,
                orientation=orientation,
                slice_axis=slice_axis
            )

        # prevent negative strides error when arrays are collated
        input_image = np.ascontiguousarray(input_image)
        output_points = np.ascontiguousarray(output_points)

        # add channel dim
        input_image = np.expand_dims(input_image, axis=-1)

        if self._input_image_transforms is not None:
            input_image_transforms = albumentations.ReplayCompose(self._input_image_transforms)(image=input_image)
            input_image_transformed = input_image_transforms['image']
        else:
            input_image_transforms = []
            input_image_transformed = input_image

        if self._output_points_transforms is not None:
            output_points = albumentations.Compose(self._output_points_transforms)(image=output_points)[
                'image']

        if input_image_transforms:
            pad_transform = [x for x in input_image_transforms['replay']['transforms'] if
                             x['__class_fullname__'] == 'PadIfNeeded']
        else:
            pad_transform = []
        if len(pad_transform) != 0:
            if len(pad_transform) > 1:
                raise ValueError('Expected 1 pad transform')
            pad_transform = pad_transform[0]
            pad_transform = pad_transform['params']
        else:
            pad_transform = {}

        # mask to downweight background in loss
        # it would be better to pull ccf label for background
        # but that would require mapping to ccf which adds complexity
        threshold = threshold_otsu(input_image)
        tissue_mask = (input_image > threshold).astype('uint8')

        if self._tissue_mask_transforms:
            tissue_mask = albumentations.Compose(self._tissue_mask_transforms)(image=tissue_mask)['image']

        tissue_mask = tissue_mask.squeeze()

        if self.patch_size is not None:
            res = input_image_transformed, output_points, dataset_idx, slice_idx, patch_y, patch_x, orientation.value, pad_transform, tissue_mask
        else:
            res = input_image_transformed, output_points, dataset_idx, slice_idx, orientation.value, pad_transform, tissue_mask
        return res

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns
        -------
        int
            Number of items (patches in TEST mode, slices in TRAIN mode).
        """
        if self._mode == TrainMode.TEST and self._patch_size is not None:
            return len(self._precomputed_patches)
        else:
            num_slices = 0
            for orientation in self._orientation:
                num_slices_in_axis = self._get_num_slices_in_axis(
                    orientation=orientation
                )
                num_slices += sum(num_slices_in_axis)
            return num_slices

    def _get_patch(self, slice_2d: tensorstore.TensorStore, patch_x: Optional[int] = None,
                   patch_y: Optional[int] = None):
        """
        Extract patch from slice.

        If patch_x/patch_y are None, choose randomly. Pads the patch to patch_size if needed.

        Parameters
        ----------
        slice_2d : tensorstore.TensorStore
            2D slice to extract patch from.
        patch_x : int, optional
            Starting x coordinate for patch. If None, chosen randomly.
        patch_y : int, optional
            Starting y coordinate for patch. If None, chosen randomly.

        Returns
        -------
        tuple
            Tuple containing:
            - patch : torch.Tensor
                Extracted and padded patch.
            - patch_y : int
                Actual starting y coordinate used.
            - patch_x : int
                Actual starting x coordinate used.
        """
        h, w = slice_2d.shape

        ph, pw = self._patch_size

        # Adjust patch size to what's available
        ph = min(h, ph)
        pw = min(w, pw)

        if patch_x is None or patch_y is None:
            # Random position (0 if slice is smaller than patch)
            patch_y = random.randint(0, max(0, h - ph))
            patch_x = random.randint(0, max(0, w - pw))

        # Extract patch
        patch = torch.from_numpy(
            slice_2d[patch_y:patch_y + ph, patch_x:patch_x + pw].read().result())

        # Pad to patch_size if needed
        patch = self._pad_patch_to_size(patch)

        return patch, patch_y, patch_x

    def _pad_patch_to_size(self, patch):
        """
        Pad extracted patch to patch_size if needed.

        Parameters
        ----------
        patch : torch.Tensor
            Patch tensor to pad.

        Returns
        -------
        torch.Tensor
            Padded patch with dimensions matching self._patch_size.
        """
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
        Transform slice and template points to have a uniform orientation.

        Applies flipping and transposition to ensure consistent orientation across
        different acquisition configurations.

        Parameters
        ----------
        slice : np.ndarray
            2D slice array.
        template_points : np.ndarray
            3D array of template coordinates with shape (height, width, 3).
        acquisition_axes : list[AcquisitionAxis]
            Acquisition axes defining the current orientation.
        orientation : SliceOrientation
            Slice orientation.
        slice_axis : AcquisitionAxis
            The axis along which slicing occurs.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (transformed_slice, transformed_template_points).
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