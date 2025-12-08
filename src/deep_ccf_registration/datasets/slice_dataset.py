import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import aind_smartspim_transform_utils
import albumentations
import ants
import numpy as np
import pandas as pd
import tensorstore
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection, \
    convert_from_ants_space
from loguru import logger
from pydantic import BaseModel
from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset

from deep_ccf_registration.metadata import AcquisitionAxis, SubjectMetadata, SliceOrientation
from deep_ccf_registration.utils.logging_utils import timed_func
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore
from deep_ccf_registration.utils.transforms import transform_points_to_template_ants_space, \
    apply_transforms_to_points, map_points_to_left_hemisphere
from deep_ccf_registration.utils.utils import get_ccf_annotations


def _read_slice_patch(slice_2d_bbox: tensorstore.TensorStore, patch_y: int, ph: int, patch_x: int, pw: int) -> np.ndarray:
    """Read a slice patch from tensorstore with retry logic for transient failures."""
    return slice_2d_bbox[patch_y:patch_y + ph, patch_x:patch_x + pw].read().result()


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


class TissueBoundingBox(BaseModel):
    """
    start y, x and width, height of tissue bounding boxes, obtained via
    `get_tissue_bounding_box.py`
    """
    y: int
    x: int
    width: int
    height: int

class TissueBoundingBoxes(BaseModel):
    bounding_boxes: dict[str, list[Optional[TissueBoundingBox]]]

class SliceDataset(Dataset):
    """
    Dataset for loading slices and their mapped points in template space.

    This dataset loads 2D slices from 3D volumes along with their corresponding
    coordinates in template space after applying registration transformations.
    """
    def __init__(self,
                 dataset_meta: list[SubjectMetadata],
                 ls_template_parameters: AntsImageParameters,
                 tissue_bboxes: TissueBoundingBoxes,
                 template_ml_dim_size: int,
                 ccf_annotations_path: Optional[Path] = None,
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
                 mask_transforms: Optional[list[albumentations.BasicTransform]] = None,
                 return_tissue_mask: bool = False
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
        return_tissue_mask: If true, returns a mask where the ccf annotations indicate a region is foreground (1) or background (0)
            Otherwise, just returns a mask of 1 for image and 0 for pad
        tissue_bboxes: tissue bounding boxes, obtained via `get_tissue_bounding_box.py`.
            Each subject id maps to a list of bounding boxes ordered by slice index, or null if no tissue is present in slices
        template_ml_dim_size: template ML shape in index space
        """
        super().__init__()
        self._dataset_meta = dataset_meta
        # TODO the schema needs to include orientation. Currently just generated for sagittal
        self._tissue_bboxes = tissue_bboxes.bounding_boxes
        if orientation is None:
            orientation = [SliceOrientation.SAGITTAL, SliceOrientation.CORONAL, SliceOrientation.HORIZONTAL]
        else:
            orientation = [orientation]
        self._orientation = orientation
        self._slice_ranges = {orientation: [self._get_slice_range(subject=subject, orientation=orientation) for subject in self._dataset_meta] for orientation in orientation}
        self._registration_downsample_factor = registration_downsample_factor
        self._warps = self._load_warps(tensorstore_aws_credentials_method=tensorstore_aws_credentials_method)
        self._volumes = self._load_volumes(tensorstore_aws_credentials_method=tensorstore_aws_credentials_method)
        self._volume_arrays: list[Optional[np.ndarray]] = [None] * len(self._volumes)
        self._warp_arrays: list[Optional[np.ndarray]] = [None] * len(self._warps)
        self._crop_warp_to_bounding_box = crop_warp_to_bounding_box
        self._patch_size = patch_size
        self._mode = mode
        self._limit_sagittal_slices_to_hemisphere = limit_sagittal_slices_to_hemisphere
        self._input_image_transforms = input_image_transforms
        self._output_points_transforms = output_points_transforms
        self._mask_transforms = mask_transforms
        self._template_ml_dim_size = template_ml_dim_size

        if normalize_orientation_map is not None:
            for axis, orientation in normalize_orientation_map.items():
                if len(orientation) != 2:
                    raise ValueError('Orientation must be 2d for a 2d slice')
        self._normalize_orientation_map = normalize_orientation_map

        self._ls_template_parameters = ls_template_parameters
        self._precomputed_patches = self._build_patch_index() if mode == TrainMode.TEST and patch_size is not None else None
        self._ccf_annotations_path = ccf_annotations_path
        self._ccf_annotations = None
        self._return_tissue_mask = return_tissue_mask

    @property
    def patch_size(self) -> Optional[tuple[int, int]]:
        return self._patch_size

    @property
    def ccf_annotations(self) -> np.ndarray:
        if self._ccf_annotations is None:
            self._ccf_annotations = np.load(self._ccf_annotations_path, mmap_mode='r')
        return self._ccf_annotations

    @property
    def subject_metadata(self) -> list[SubjectMetadata]:
        return self._dataset_meta

    def set_mode(self, mode: TrainMode):
        self._mode = mode
        if mode == TrainMode.TEST and self._precomputed_patches is None and self._patch_size is not None:
            self._precomputed_patches = self._build_patch_index()

    def _get_patch_positions(
            self,
            bounding_box: TissueBoundingBox
    ) -> list[tuple[int, int]]:
        """
        Get all patch positions to tile within a bounding box.

        Parameters
        ----------
        bounding_box : TissueBoundingBox
            Limit patch positions to within this bounding box.
            The bounding box coordinates should be relative to the slice.

        Returns
        -------
        list[tuple[int, int]]
            List of (y, x) positions for patch extraction.
        """
        if self._patch_size is None:
            # Return single patch at bounding box start
            return [(bounding_box.y, bounding_box.x)]

        ph, pw = self._patch_size

        start_y = bounding_box.y
        start_x = bounding_box.x
        end_y = bounding_box.y + bounding_box.height
        end_x = bounding_box.x + bounding_box.width

        positions = []

        # Calculate number of patches needed (with overlap to cover edges)
        for y in range(start_y, end_y, ph):
            for x in range(start_x, end_x, pw):
                # Adjust position if it would go past the edge of the region
                y_start = min(y, max(start_y, end_y - ph))
                x_start = min(x, max(start_x, end_x - pw))

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

    def _load_volumes(self, tensorstore_aws_credentials_method: str = "default") -> list[tensorstore.TensorStore]:
        """
        Load stitched volumes for all subjects in the dataset.

        Parameters
        ----------
        tensorstore_aws_credentials_method : str, default="default"
            Credentials lookup method for tensorstore.

        Returns
        -------
        list[tensorstore.TensorStore]
            List of volume tensorstores for each subject.
        """
        volumes = []
        for experiment_meta in self._dataset_meta:
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
            volumes.append(volume)
        return volumes

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
        slice_ranges = self._slice_ranges[orientation]

        num_slices_per_subject = [len(slice_range) for slice_range in slice_ranges]

        # Compute cumulative sum to find which dataset the idx falls into
        num_slices_cumsum = np.cumsum([0] + num_slices_per_subject)

        # Find which dataset this global index belongs to
        dataset_idx = int(np.searchsorted(num_slices_cumsum[1:], idx, side='right'))

        # Find the local index within that dataset's tissue slices
        local_idx = int(idx - num_slices_cumsum[dataset_idx])

        # Map local index to actual slice index from the list
        slice_idx = slice_ranges[dataset_idx][local_idx]

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
        slices = list(range(subject.registered_shape[slice_axis.dimension]))

        # exclude slices containing no tissue
        slices = [x for x in slices if self._tissue_bboxes[subject.subject_id][x] is not None]

        return slices

    def _get_num_slices_in_axis_per_subject(self, orientation: SliceOrientation) -> list[int]:
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
        return [len(x) for x in self._slice_ranges[orientation]]

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
                slice_range = self._slice_ranges[orientation][dataset_idx]

                for slice_idx in slice_range:
                    # Get all patch positions for this slice
                    patch_positions = self._get_patch_positions(
                        bounding_box=self._tissue_bboxes[subject_meta.subject_id][slice_idx]
                    )

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

        if self._precomputed_patches is not None:
            patch_x = self._precomputed_patches[idx].x
            patch_y = self._precomputed_patches[idx].y
            dataset_idx = self._precomputed_patches[idx].dataset_idx
            slice_idx = self._precomputed_patches[idx].slice_idx
        else:
            dataset_idx, slice_idx = self._get_slice_from_idx(idx=idx, orientation=orientation)
            patch_x, patch_y = None, None

        experiment_meta = self._dataset_meta[dataset_idx]
        subject_id = experiment_meta.subject_id
        acquisition_axes = experiment_meta.axes
        slice_axis = experiment_meta.get_slice_axis(orientation=orientation)

        # Get cached volume
        volume = self._volume_arrays[dataset_idx]

        volume_slice = [0, 0, slice(None), slice(None), slice(None)]
        volume_slice[slice_axis.dimension + 2] = slice_idx  # + 2 since first 2 dims unused

        input_image, patch_y, patch_x, patch_height, patch_width = self._extract_slice_image(
            slice_2d=volume[tuple(volume_slice)],
            patch_x=patch_x,
            patch_y=patch_y,
            tissue_bbox=self._tissue_bboxes[experiment_meta.subject_id][slice_idx]
        )

        height, width = patch_height, patch_width

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
            warp=self._warp_arrays[dataset_idx],
        )

        output_points = ls_template_points.reshape((height, width, 3))

        if orientation == SliceOrientation.SAGITTAL:
            output_points = map_points_to_left_hemisphere(
                template_points=output_points,
                template_parameters=self._ls_template_parameters,
                ml_dim_size=self._template_ml_dim_size
            )

        if self._normalize_orientation_map is not None:
            input_image, output_points = self._normalize_orientation(
                slice=input_image,
                template_points=output_points,
                acquisition_axes=acquisition_axes,
                orientation=orientation,
                slice_axis=slice_axis
            )

        input_image = rescale_intensity(
            input_image,
            in_range=tuple(np.percentile(input_image, (1, 99))),
            out_range=(0, 1)
        ).astype(np.float32)

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

        if self._return_tissue_mask:
            tissue_mask = self._calculate_tissue_mask(
                template_points=output_points
            )
        else:
            tissue_mask = np.array([])
        pad_mask = _calculate_non_pad_mask(
            shape=self._patch_size if self._patch_size is not None else input_image_transformed.shape, pad_transform=pad_transform)

        if self._mask_transforms:
            if self._return_tissue_mask:
                tissue_mask = albumentations.Compose(self._mask_transforms)(image=tissue_mask)['image']
            pad_mask = albumentations.Compose(self._mask_transforms)(image=pad_mask)['image']

        if len(tissue_mask.shape) == 3:
            tissue_mask = tissue_mask.squeeze()
        if len(pad_mask.shape) == 3:
            pad_mask = pad_mask.squeeze()

        if self._output_points_transforms is not None:
            output_points = albumentations.Compose(self._output_points_transforms)(image=output_points)[
                'image']

        return input_image_transformed, output_points, dataset_idx, slice_idx, patch_y, patch_x, orientation.value, pad_transform, tissue_mask, pad_mask, subject_id

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
                num_slices_in_axis = self._get_num_slices_in_axis_per_subject(
                    orientation=orientation
                )
                num_slices += sum(num_slices_in_axis)
            return num_slices

    def _extract_slice_image(
            self,
            slice_2d: np.ndarray,
            tissue_bbox: TissueBoundingBox,
            patch_x: Optional[int] = None,
            patch_y: Optional[int] = None,
    ) -> tuple[np.ndarray, int, int, int, int]:
        """
        Extract patch from slice or whole slice if patch_x, patch_y is None and self._patch_size is None.
        Only extracts a region from within `tissue_bbox`.

        If patch_x/patch_y are None and patch_size is not None, choose randomly.

        Parameters
        ----------
        slice_2d : np.ndarray
            2D slice to extract patch from.
        tissue_bbox : TissueBoundingBox
            Bounding box defining the region to extract from.
        patch_x : int, optional
            Starting x coordinate for patch (assumed to be within bbox). If None, chosen randomly.
        patch_y : int, optional
            Starting y coordinate for patch (assumed to be within bbox). If None, chosen randomly.

        Returns
        -------
        tuple of patch, start_y, start_x
        """
        h, w = tissue_bbox.height, tissue_bbox.width

        if self._patch_size is None:
            # Extract full bbox region
            ph, pw = h, w
            patch_y = tissue_bbox.y
            patch_x = tissue_bbox.x
        else:
            ph, pw = self._patch_size
            # Adjust patch size to available region
            ph = min(h, ph)
            pw = min(w, pw)

            if patch_y is None or patch_x is None:
                # Random position within bbox
                patch_y = random.randint(tissue_bbox.y, max(tissue_bbox.y, tissue_bbox.y + h - ph))
                patch_x = random.randint(tissue_bbox.x, max(tissue_bbox.x, tissue_bbox.x + w - pw))

        patch = slice_2d[patch_y:patch_y + ph, patch_x:patch_x + pw]

        return patch, patch_y, patch_x, ph, pw

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

        if swapped.tolist() != list(range(3)):
            slice = np.transpose(slice)
            template_points = np.permute_dims(template_points, axes=[1, 0, 2])

        return slice, template_points

    def _calculate_tissue_mask(self, template_points: np.ndarray):
        index_pts = convert_from_ants_space(template_parameters=self._ls_template_parameters, physical_pts=template_points.reshape((-1, 3)))
        ccf_annotations = get_ccf_annotations(self.ccf_annotations, index_pts).reshape(
            template_points.shape[:-1])

        tissue_mask = (ccf_annotations != 0).astype('uint8')
        return tissue_mask

    def get_subject_sample_idxs(self, subject_idxs: list[int]) -> list[int]:
        """
        Get all sample indices belonging to a list of subjects.

        Parameters
        ----------
        subject_idxs : list[int]
            List of subject indexes to get sample for.

        Returns
        -------
        list[int]
            List of global dataset indices belonging to the specified subjects.
        """
        all_indices = []
        current_global_idx = 0

        for orientation in self._orientation:
            # Get number of slices per subject for this orientation
            num_slices_per_subject = self._get_num_slices_in_axis_per_subject(orientation=orientation)

            for subject_idx in range(len(self._dataset_meta)):
                num_slices = num_slices_per_subject[subject_idx]
                if subject_idx in subject_idxs:
                    # Add all indices for this subject in this orientation
                    all_indices.extend(range(current_global_idx, current_global_idx + num_slices))
                current_global_idx += num_slices

        return all_indices

    def get_arrays(self, idxs: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        volume_arrs = []
        warp_arrs = []
        for idx in idxs:
            if self._volume_arrays[idx] is None:
                volume = self._volumes[idx][:].read().result()
            else:
                volume = self._volume_arrays[idx]

            if self._warp_arrays[idx] is None:
                warp = self._warps[idx][:].read().result()
            else:
                warp = self._warp_arrays[idx]
            volume_arrs.append(volume)
            warp_arrs.append(warp)
        return volume_arrs, warp_arrs

    def reset_data(self, subject_idxs: list[int], volumes: list[np.ndarray], warps: list[np.ndarray]):
        for i in range(len(self._volume_arrays)):
            if self._volume_arrays[i] is not None:
                self._volume_arrays[i] = None
        for i in range(len(self._warp_arrays)):
            if self._warp_arrays[i] is not None:
                self._warp_arrays[i] = None

        for i, idx in enumerate(subject_idxs):
            self._volume_arrays[idx] = volumes[i]
            self._warp_arrays[idx] = warps[i]

    def get_subject_batches(self, n_subjects_per_batch: int) -> list[list[int]]:
        subjects = self.subject_metadata
        subject_idxs = np.arange(len(subjects))
        np.random.shuffle(subject_idxs)
        subject_idxs = subject_idxs.tolist()
        subject_idx_batches = [subject_idxs[i:i + n_subjects_per_batch] for i in
                               range(0, len(subject_idxs), n_subjects_per_batch)]
        return subject_idx_batches

def _calculate_non_pad_mask(shape: tuple[int, int], pad_transform: dict[str, Any]):
    mask = np.zeros(shape, dtype="uint8")
    if pad_transform:
        mask[
            pad_transform['pad_top']:pad_transform['pad_top']+pad_transform['shape'][0],
            pad_transform['pad_left']:pad_transform['pad_left']+pad_transform['shape'][1]
        ] = 1
    else:
        mask[:] = 1
    return mask