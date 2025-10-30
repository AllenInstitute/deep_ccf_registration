import json
import random
from dataclasses import dataclass
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
    Convert affine_transformed_voxels and warp to format suitable for grid_sample.

    Takes the affine_transformed_voxels and warp and converts them to a format
    suitable for PyTorch's grid_sample function.

    Parameters
    ----------
    warp : np.ndarray
        Displacement field array.
    affine_transformed_voxels : np.ndarray
        Voxel coordinates after affine transformation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (warp_tensor, normalized_voxel_coordinates) ready for grid_sample.
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
    Crop warp to region bounded by min/max coordinates after affine transformation.

    This function crops the warp to only the region needed, avoiding loading the
    entire warp. It also modifies affine_transformed_voxels in-place to set offset
    to 0 so that it can index into the cropped warp.

    Parameters
    ----------
    warp : tensorstore.TensorStore or np.ndarray
        Full displacement field.
    affine_transformed_voxels : np.ndarray
        Voxels after applying inverse affine to input points. Modified in-place.
    warp_interpolation_padding : int, default=5
        Padding around the min/max coords to crop for interpolation.

    Returns
    -------
    np.ndarray
        Cropped displacement field.

    Raises
    ------
    ValueError
        If points are completely outside template bounds after affine transform.
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
    """
    Apply affine and non-linear transformations to points in input space.

    Transforms points from input space to template space by applying inverse affine
    transformation followed by displacement field warping.

    Parameters
    ----------
    points : np.ndarray
        Points in input space to be transformed.
    experiment_meta : SubjectMetadata
        Subject metadata containing transformation parameters.
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
    pd.DataFrame
        DataFrame with transformed points in template space with columns ["ML", "AP", "DV"].
    """
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
    """
    Transform points from input volume space to template ANTs space.

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

class TrainMode(Enum):
    TRAIN = 0
    TEST = 1

class SliceDataset(Dataset):
    """
    Dataset for loading slices and their mapped points in template space.

    This dataset loads 2D slices from 3D volumes along with their corresponding
    coordinates in template space after applying registration transformations.
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
        Initialize SliceDataset.

        Parameters
        ----------
        dataset_meta : list[SubjectMetadata]
            List of subject metadata for all subjects in the dataset.
        ls_template : ants.ANTsImage
            The smartSPIM light sheet template image.
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

        if normalize_orientation_map is not None:
            for axis, orientation in normalize_orientation_map.items():
                if len(orientation) != 2:
                    raise ValueError('Orientation must be 2d for a 2d slice')
        self._normalize_orientation_map = normalize_orientation_map

        self._ls_template = ls_template

        if mode == TrainMode.TEST:
            self._patch_index = self._build_patch_index()

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
            List of (x, y) positions for patch extraction.
        """
        if self._patch_size is None:
            # Return single patch covering whole slice
            return [(0, 0)]

        h, w = slice_shape
        ph, pw = self._patch_size

        positions = []

        # Calculate number of patches needed (with overlap to cover edges)
        for x in range(0, h, ph):
            for y in range(0, w, pw):
                # Adjust position if it would go past the edge
                x_start = min(x, max(0, h - ph))
                y_start = min(y, max(0, w - pw))

                # Avoid duplicate patches at the edge
                if not positions or (x_start, y_start) != positions[-1]:
                    positions.append((x_start, y_start))

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
                sagittal_axis = self._get_slice_axis(
                    axes=subject.axes,
                    orientation=orientation
                )
                if sagittal_axis.direction == AcquisitionDirection.RIGHT_TO_LEFT:
                    # invert to get slice in left hemisphere
                    slice_idx = slice_idx + subject.sagittal_midline

        return dataset_idx, slice_idx

    def _get_slice_range(self, subject: SubjectMetadata, orientation: SliceOrientation) -> range:
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
        range
            Range of valid slice indices.
        """
        slice_axis = self._get_slice_axis(axes=subject.axes, orientation=orientation)

        if orientation == SliceOrientation.SAGITTAL and self._limit_sagittal_slices_to_hemisphere:
            sagittal_dim = subject.registered_shape[slice_axis.dimension]
            # Always sample from the left hemisphere due to brain symmetry
            if slice_axis.direction == AcquisitionDirection.LEFT_TO_RIGHT:
                return range(subject.sagittal_midline)
            else:
                return range(subject.sagittal_midline, sagittal_dim)
        else:
            return range(subject.registered_shape[slice_axis.dimension])

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

    def _get_slice_axis(self, axes: list[AcquisitionAxis], orientation: SliceOrientation) -> AcquisitionAxis:
        """
        Get the acquisition axis corresponding to the slice orientation.

        Parameters
        ----------
        axes : list[AcquisitionAxis]
            List of acquisition axes.
        orientation : SliceOrientation
            Desired slice orientation.

        Returns
        -------
        AcquisitionAxis
            The axis along which slicing occurs.

        Raises
        ------
        ValueError
            If no unique axis matches the orientation.
        NotImplementedError
            If orientation other than SAGITTAL is requested.
        """
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
                slice_axis = self._get_slice_axis(axes=subject_meta.axes, orientation=orientation)

                # Get slice range for this subject
                slice_range = self._get_slice_range(subject_meta, orientation)

                for slice_idx in slice_range:
                    # Get slice shape (axes except the slice axis)
                    axes_except_slice = [ax for ax in subject_meta.axes if ax != slice_axis]
                    slice_shape = tuple(
                        subject_meta.registered_shape[ax.dimension] for ax in axes_except_slice)

                    # Get all patch positions for this slice
                    patch_positions = self._get_patch_positions(slice_shape)

                    for patch_x, patch_y in patch_positions:
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
        # Determine what to load
        if self._mode == TrainMode.TRAIN:
            orientation = random.choice(self._orientation)
            dataset_idx, slice_idx = self._get_slice_from_idx(idx=idx, orientation=orientation)
            patch_x, patch_y = None, None  # Will be determined randomly
        else:
            # TEST mode - use precomputed patch index
            patch_info = self._patch_index[idx]
            dataset_idx = patch_info.dataset_idx
            slice_idx = patch_info.slice_idx
            patch_x = patch_info.x
            patch_y = patch_info.y
            orientation = patch_info.orientation

        experiment_meta = self._dataset_meta[dataset_idx]
        acquisition_axes = experiment_meta.axes
        slice_axis = self._get_slice_axis(axes=acquisition_axes, orientation=orientation)

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
            input_slice, patch_x, patch_y = self._get_patch(
                slice_2d=volume[tuple(volume_slice)],
                patch_x=patch_x,
                patch_y=patch_y
            )

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
        """
        Get the total number of items in the dataset.

        Returns
        -------
        int
            Number of items (patches in TEST mode, slices in TRAIN mode).
        """
        if self._mode == TrainMode.TEST:
            return len(self._patch_index)
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
            - patch_x : int
                Actual starting x coordinate used.
            - patch_y : int
                Actual starting y coordinate used.
        """
        h, w = slice_2d.shape

        if self._patch_size is None:
            return slice_2d[:].read().result(), 0, 0

        ph, pw = self._patch_size

        # Adjust patch size to what's available
        ph = min(h, ph)
        pw = min(w, pw)

        if patch_x is None or patch_y is None:
            # Random position (0 if slice is smaller than patch)
            patch_x = random.randint(0, max(0, h - ph))
            patch_y = random.randint(0, max(0, w - pw))

        # Extract patch
        patch = torch.from_numpy(
            slice_2d[patch_x:patch_x + ph, patch_y:patch_y + pw].read().result())

        # Pad to patch_size if needed
        patch = self._pad_patch_to_size(patch)

        return patch, patch_x, patch_y

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