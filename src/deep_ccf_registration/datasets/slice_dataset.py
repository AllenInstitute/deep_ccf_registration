from dataclasses import dataclass
from pathlib import Path
from typing import Optional


import numpy as np
import pandas as pd
import tensorstore
from loguru import logger
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset

from deep_ccf_registration.datasets.aquisition_meta import AcquisitionDirection
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import get_subject_rotation_range
from deep_ccf_registration.datasets.utils.template_points import transform_points_to_template_space, apply_transforms_to_points, Affine
from deep_ccf_registration.datasets.utils.interpolation import map_coordinates_cropped
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation, TissueBoundingBoxes, \
    AcquisitionAxis, RotationAngles, SubjectRotationAngle, TissueBoundingBox
from deep_ccf_registration.utils.logging_utils import timed, timed_func
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore
from tenacity import retry, wait_random_exponential


@dataclass(frozen=True)
class PatchSample:
    slice_idx: int
    start_y: int
    start_x: int
    data: np.ndarray
    template_points: Optional[np.ndarray] = None
    dataset_idx: str = ""
    orientation: str = ""
    subject_id: str = ""
    tissue_mask: Optional[np.ndarray] = None
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for batch collation."""
        return {
            "slice_idx": self.slice_idx,
            "start_y": self.start_y,
            "start_x": self.start_x,
            "data": self.data,
            "template_points": self.template_points,
            "dataset_idx": self.dataset_idx,
            "orientation": self.orientation,
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True)
class SliceSampleSpec:
    """Slice to sample"""

    metadata: SubjectMetadata
    slice_idx: int
    orientation: SliceOrientation

@dataclass
class SliceRotationRanges:
    x: tuple[float, float] = (0, 0)
    y: tuple[float, float] = (0, 0)
    z: tuple[float, float] = (0, 0)


class SubjectSliceDataset(Dataset):
    """Map-style dataset that randomly samples slices from subjects."""

    def __init__(
        self,
        subjects: list[SubjectMetadata],
        samples: np.ndarray,
        template_parameters: TemplateParameters,
        is_train: bool,
        tissue_bboxes_path: Path,
        rotation_angles: RotationAngles,
        orientations: list[SliceOrientation],
        crop_size: Optional[tuple[int, int]] = None,
        transform: Optional[callable] = None,
        include_tissue_mask: bool = False,
        ccf_annotations_path: Optional[str] = None,
        rotate_slices: bool = False,
        is_debug: bool = False,
        debug_slice_idx: Optional[int] = None,
        aws_credentials_method: Optional[str] = None,
        # exclude slices with tissue bboxes smaller in area than this percentile
        # this only impacts slice rotation
        tissue_bbox_area_rejection_percentile: int = 20

    ):
        if include_tissue_mask and ccf_annotations_path is None:
            raise ValueError("include_tissue_mask=True requires ccf_annotations_path")

        self._is_train = is_train
        self._template_parameters = template_parameters
        self._transform = transform
        self._include_tissue_mask = include_tissue_mask
        self._ccf_annotations_path = ccf_annotations_path
        self._orientations = orientations

        self._tissue_bboxes_path = tissue_bboxes_path
        self._crop_size = crop_size
        self._rotate_slices = rotate_slices
        self._rotation_angles = rotation_angles
        self._subject_id_to_subject_map = {x.subject_id: x for x in subjects}
        self._is_debug = is_debug
        self._debug_slice_idx = debug_slice_idx
        self._aws_credentials_method = aws_credentials_method
        self._samples = samples

        tissue_bboxes = pd.read_parquet(self._tissue_bboxes_path)
        tissue_bboxes['area'] = tissue_bboxes['width'] * tissue_bboxes['height']
        self._tissue_bbox_area_rejection_threshold = np.percentile(tissue_bboxes['area'], tissue_bbox_area_rejection_percentile)
        logger.info(f"Dataset initialized with {len(self)} total samples")

    @retry(wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
    def _read_volume(self, subject: SubjectMetadata) -> tensorstore.TensorStore:
        volume = tensorstore.open(
            spec={
                "driver": "auto",
                "kvstore": create_kvstore(
                    path=str(subject.stitched_volume_path) + "/3",
                    aws_credentials_method="anonymous",
                ),
            },
            read=True,
        ).result()
        return volume

    @retry(wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
    def _read_warp(self, subject: SubjectMetadata):
        warp = tensorstore.open(
            spec={
                "driver": "auto",
                "kvstore": create_kvstore(
                    path=subject.get_warp_path(),
                    aws_credentials_method=self._aws_credentials_method,
                ),
            },
            read=True,
        ).result()
        return warp

    def __len__(self) -> int:
        return self._samples.shape[0]

    @timed_func
    def __getitem__(self, index: int) -> PatchSample:
        sample = self._samples[index]
        subject_id, slice_idx, orientation = str(sample[0]), int(sample[1]), SliceOrientation(sample[2])
        subject = self._subject_id_to_subject_map[subject_id]
        spec = SliceSampleSpec(metadata=subject, slice_idx=slice_idx, orientation=orientation)

        metadata = spec.metadata
        slice_axis = metadata.get_slice_axis(spec.orientation)

        with timed():
            subject_bboxes = pd.read_parquet(self._tissue_bboxes_path / f'subject_id={metadata.subject_id}')
        subject_bboxes['area'] = subject_bboxes['width'] * subject_bboxes['height']
        subject_bboxes = subject_bboxes[subject_bboxes['area'] > self._tissue_bbox_area_rejection_threshold]
        bbox = subject_bboxes[subject_bboxes['index'] == spec.slice_idx].iloc[0]
        start_y = bbox['y']
        start_x = bbox['x']
        patch_height = bbox['height']
        patch_width = bbox['width']

        coordinate_grid, y_rot, x_rot = self._get_coordinate_grid(
            experiment_meta=metadata,
            start_y=start_y,
            start_x=start_x,
            height=patch_height,
            width=patch_width,
            slice_axis=slice_axis,
            fixed_index_value=spec.slice_idx,
            orientation=spec.orientation,
            tissue_slices_indices=subject_bboxes['index'].tolist(),
        )

        volume = self._read_volume(subject=metadata)

        if self._rotate_slices:
            input_slice = self._get_rotated_slice(
                volume=volume,
                point_grid=coordinate_grid,
                patch_width=patch_width,
                patch_height=patch_height
            )
        else:
            input_slice = self._get_slice(
                volume=volume,
                patch_height=patch_height,
                patch_width=patch_width,
                experiment_meta=metadata,
                spec=spec,
                bbox=TissueBoundingBox(
                    y=bbox['y'],
                    x=bbox['x'],
                    width=bbox['width'],
                    height=bbox['height']
                )
            )

        try:
            template_points = self._get_template_points(
                point_grid=coordinate_grid,
                patch_height=patch_height,
                patch_width=patch_width,
                experiment_meta=metadata,
                volume=volume,
            )
        except IndexError:
            logger.error('Points completely out of bounds after affine')
            logger.error(f'index={index}, subject_id={subject.subject_id}, slice_idx={slice_idx}, y_rot={y_rot}, x_rot={x_rot}')
            raise

        tissue_mask = None
        if self._include_tissue_mask and self._ccf_annotations_path is not None:
            # Re-create memmap every access to avoid memory leak
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            ccf_annotations = np.load(self._ccf_annotations_path, mmap_mode='r')
            tissue_mask = _get_tissue_mask(
                annotations=ccf_annotations,
                template_patch=template_points,
                template_parameters=self._template_parameters,
            )

        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        if self._transform is not None:
            transforms = self._transform(
                image=input_slice,
                template_coords=template_points,
                mask=tissue_mask,
                slice_axis=slice_axis,
                acquisition_axes=metadata.axes,
                orientation=spec.orientation,
                subject_rotation=self._rotation_angles.rotation_angles[metadata.subject_id],
                subject_id=metadata.subject_id,
                slice_idx=spec.slice_idx,
            )
            input_slice = transforms["image"]
            template_points = transforms["template_coords"]
            tissue_mask = transforms["mask"]

        return PatchSample(
            slice_idx=spec.slice_idx,
            start_y=start_y,
            start_x=start_x,
            data=input_slice,
            template_points=template_points,
            dataset_idx=metadata.subject_id,
            orientation=spec.orientation.value,
            subject_id=metadata.subject_id,
            tissue_mask=tissue_mask,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
        )

    @retry(wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
    @timed_func
    def _get_slice(
        self,
        volume: tensorstore.TensorStore,
        experiment_meta: SubjectMetadata,
        patch_height: int,
        patch_width: int,
        spec: SliceSampleSpec,
        bbox: TissueBoundingBox,
    ) -> np.ndarray:
        slice_axis = experiment_meta.get_slice_axis(orientation=spec.orientation)
        axes = sorted(experiment_meta.axes, key=lambda axis: axis.dimension)
        in_plane_axes = [ax for ax in axes if ax.dimension != slice_axis.dimension]
        if len(in_plane_axes) != 2:
            raise ValueError("Expected exactly two in-plane axes")
        y_axis, x_axis = in_plane_axes

        start_y = bbox.y
        start_x = bbox.x

        spatial_slices = [0, 0, slice(None), slice(None), slice(None)]
        spatial_slices[2 + slice_axis.dimension] = spec.slice_idx
        spatial_slices[2 + y_axis.dimension] = slice(start_y, start_y + patch_height)
        spatial_slices[2 + x_axis.dimension] = slice(start_x, start_x + patch_width)
        data_patch = volume[
            tuple(spatial_slices)].read().result().astype("float32")
        return data_patch

    @retry(wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
    @timed_func
    def _get_rotated_slice(
        self,
        volume: tensorstore.TensorStore,
        point_grid: np.ndarray,
        patch_height: int,
        patch_width: int,
    ) -> np.ndarray:
        # Interpolate volume at rotated coordinate locations
        # coordinate_grid: (n_points, 3), map_coordinates needs (3, n_points)
        coords_for_interp = point_grid.T
        # Volume shape: (C, T, D0, D1, D2) - sample from spatial dims
        volume_3d = volume[0, 0]
        interpolated_flat = map_coordinates_cropped(
            volume=volume_3d,
            coords=coords_for_interp,
            order=1,  # linear interpolation
            mode='constant',
            cval=0.0,
        )
        data_patch = interpolated_flat.reshape(patch_height, patch_width).astype("float32")

        return data_patch

    @retry(wait=wait_random_exponential(multiplier=1, max=60), reraise=True)
    @timed_func
    def _get_template_points(
        self,
        volume: tensorstore.TensorStore,
        point_grid: np.ndarray,
        patch_height: int,
        patch_width: int,
        experiment_meta: SubjectMetadata,
    ) -> np.ndarray:

        with timed():
            points = transform_points_to_template_space(
                points=point_grid,
                input_volume_shape=volume.shape[2:],
                acquisition_axes=experiment_meta.axes,
                ls_template_info=self._template_parameters,
                registration_downsample=experiment_meta.registration_downsample
            )

        with timed():
            template_points = apply_transforms_to_points(
                points=points,
                template_parameters=self._template_parameters,
                affine=Affine.from_ants_file(affine_path=experiment_meta.ls_to_template_affine_matrix_path),
                warp=self._read_warp(subject=experiment_meta),
            )

        with timed():
            template_points = template_points.reshape((patch_height, patch_width, 3))

        return template_points

    def _get_coordinate_grid(
        self,
        experiment_meta: SubjectMetadata,
        start_x: int,
        start_y: int,
        height: int,
        width: int,
        fixed_index_value: int,
        slice_axis: AcquisitionAxis,
        tissue_slices_indices: list[int],
        orientation: Optional[SliceOrientation] = None,

    ):
        if self._rotate_slices:
            if orientation is None:
                raise ValueError('provide orientation')
            slice_rotation_ranges = self._get_slice_rotation_ranges(
                metadata=experiment_meta,
                orientation=orientation
            )

            # Compute tissue depth range from bounding boxes
            tissue_min = min(tissue_slices_indices)
            tissue_max = max(tissue_slices_indices)
            bounded_x_range, bounded_y_range = compute_bounded_rotation_ranges(
                slice_idx=fixed_index_value,
                patch_height=height,
                patch_width=width,
                tissue_min=tissue_min,
                tissue_max=tissue_max,
                desired_x_rot_range=slice_rotation_ranges.x,
                desired_y_rot_range=slice_rotation_ranges.y,
            )

            y_rot = np.random.uniform(*bounded_y_range)
            x_rot = np.random.uniform(*bounded_x_range)
        else:
            y_rot = 0.0
            x_rot = 0.0

        axis1_coords, axis2_coords = np.meshgrid(
            np.arange(start_y, start_y + height),
            np.arange(start_x, start_x + width),
            indexing='ij'
        )

        axis1_flat = axis1_coords.flatten()
        axis2_flat = axis2_coords.flatten()

        n_points = len(axis1_flat)

        slice_index = np.full(n_points, fixed_index_value)

        axes = sorted(experiment_meta.axes, key=lambda x: x.dimension)

        points = np.zeros((n_points, 3))

        points[:, slice_axis.dimension] = slice_index
        points[:, [x for x in axes if x != slice_axis][0].dimension] = axis1_flat
        points[:, [x for x in axes if x != slice_axis][1].dimension] = axis2_flat

        if x_rot != 0 or y_rot != 0:
            points = _extract_rotated_coords(
                start_yx=(start_y, start_x),
                x_rot=x_rot,
                y_rot=y_rot,
                width=width,
                height=height,
                slice_idx=fixed_index_value,
                slice_axis=slice_axis,
            )
        return points, y_rot, x_rot

    def _get_slice_rotation_ranges(
        self,
        metadata: SubjectMetadata,
        orientation: SliceOrientation
    ) -> SliceRotationRanges:
        subject_rotation: SubjectRotationAngle = self._rotation_angles.rotation_angles[metadata.subject_id]
        AP_rot_range = get_subject_rotation_range(
            subject_angle=subject_rotation.AP_rot,
            valid_range=self._rotation_angles.AP_range
        )
        SI_rot_range = get_subject_rotation_range(
            subject_angle=subject_rotation.SI_rot,
            valid_range=self._rotation_angles.SI_range
        )
        ML_rot_range = get_subject_rotation_range(
            subject_angle=subject_rotation.ML_rot,
            valid_range=self._rotation_angles.ML_range
        )

        axes = sorted(metadata.axes, key=lambda x: x.dimension)
        slice_axis = metadata.get_slice_axis(orientation=orientation)
        y_axis, x_axis = [axes[i] for i in range(3) if i != slice_axis.dimension]

        if orientation == SliceOrientation.SAGITTAL:
            direction_to_range = {
                AcquisitionDirection.SUPERIOR_TO_INFERIOR: SI_rot_range,
                AcquisitionDirection.INFERIOR_TO_SUPERIOR: SI_rot_range,
                AcquisitionDirection.ANTERIOR_TO_POSTERIOR: AP_rot_range,
                AcquisitionDirection.POSTERIOR_TO_ANTERIOR: AP_rot_range,
            }

            if y_axis.direction not in direction_to_range:
                raise ValueError(f'unexpected direction for y axis {y_axis.direction}')
            if x_axis.direction not in direction_to_range:
                raise ValueError(f'unexpected direction for x axis {x_axis.direction}')

            y_rot_range = direction_to_range[y_axis.direction]
            x_rot_range = direction_to_range[x_axis.direction]
            z_rot_range = ML_rot_range
        else:
            raise NotImplementedError(f'{orientation} not supported')

        return SliceRotationRanges(
            x=x_rot_range,
            y=y_rot_range,
            z=z_rot_range,
        )


def _get_tissue_mask(
    annotations: np.ndarray,
    template_patch: np.ndarray,
    template_parameters: TemplateParameters,
) -> np.ndarray:
    """Compute tissue masks by sampling the CCF annotations at template coords."""

    template_points = template_patch.copy()

    # convert to index space
    for dim in range(template_parameters.dims):
        template_points[..., dim] -= template_parameters.origin[dim]
        template_points[..., dim] *= template_parameters.direction[dim]
        template_points[..., dim] /= template_parameters.scale[dim]
    H, W, C = template_points.shape

    tissue_mask = (
        map_coordinates(
            input=annotations,
            coordinates=template_points.reshape(-1, C).T,
            order=0,
            mode="constant",
            cval=0,
        ) != 0
    ).astype("uint8")
    return tissue_mask.reshape(template_patch.shape[:-1])


def _extract_rotated_coords(
        slice_axis: AcquisitionAxis,
        start_yx: tuple[int, int],
        x_rot: float,
        y_rot: float,
        slice_idx: int,
        width: int,
        height: int,
):
    """
    Extract coordinates for a rotated (oblique) slice.

    X/Y rotations tilt the slice plane, causing sampling across multiple
    slice indices. Returns coordinates in volume space.

    Args:
        slice_axis: The axis perpendicular to the slice plane
        start_yx: (y, x) top-left corner of crop
        x_rot: Rotation angle around x-axis in degrees
        y_rot: Rotation angle around y-axis in degrees
        slice_idx: Slice index in volume coordinates
        width: Patch width
        height: Patch height

    Returns:
        Coordinate array of shape (n_points, 3) in volume coordinates.
    """
    logger.debug(f'rotating slice with y_rot={y_rot:.3f}, x_rot={x_rot:.3f}, width={width}, height={height}')
    center_of_rotation = (
        start_yx[0] + height / 2,
        start_yx[1] + width / 2,
    )

    # Build output grid centered at origin
    grid_0 = np.arange(height) - (height - 1) / 2.0
    grid_1 = np.arange(width) - (width - 1) / 2.0
    centered_0, centered_1 = np.meshgrid(grid_0, grid_1, indexing='ij')

    # Compute slice offset from x/y rotation (tilt angles)
    slice_offset = centered_0 * np.tan(np.radians(y_rot)) \
                   + centered_1 * np.tan(np.radians(x_rot))

    # Map to volume coordinates with shape (n_points, 3)
    in_plane_axes = [i for i in range(3) if i != slice_axis.dimension]

    slice_coords = (slice_idx + slice_offset).flatten()
    inplane_0_coords = (center_of_rotation[0] + centered_0).flatten()
    inplane_1_coords = (center_of_rotation[1] + centered_1).flatten()

    n_points = len(slice_coords)
    coords = np.zeros((n_points, 3))
    coords[:, slice_axis.dimension] = slice_coords
    coords[:, in_plane_axes[0]] = inplane_0_coords
    coords[:, in_plane_axes[1]] = inplane_1_coords

    return coords


def compute_bounded_rotation_ranges(
    slice_idx: int,
    patch_height: int,
    patch_width: int,
    tissue_min: int,
    tissue_max: int,
    desired_x_rot_range: tuple[float, float],
    desired_y_rot_range: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute rotation ranges bounded to keep sampled coordinates within tissue.

    When tilted by angle theta, points at patch edge have slice offset:
        offset = (height/2) * tan(y_rot) + (width/2) * tan(x_rot)

    Returns (bounded_x_range, bounded_y_range) in degrees.
    """
    headroom_low = slice_idx - tissue_min
    headroom_high = tissue_max - slice_idx

    half_height = patch_height / 2.0
    half_width = patch_width / 2.0

    max_abs_y_desired = max(abs(desired_y_rot_range[0]), abs(desired_y_rot_range[1]))
    max_abs_x_desired = max(abs(desired_x_rot_range[0]), abs(desired_x_rot_range[1]))

    tan_y_desired = np.tan(np.radians(max_abs_y_desired))
    tan_x_desired = np.tan(np.radians(max_abs_x_desired))

    max_offset_desired = half_height * tan_y_desired + half_width * tan_x_desired
    max_allowable_offset = min(headroom_low, headroom_high)

    if max_offset_desired <= max_allowable_offset or max_offset_desired == 0:
        return desired_x_rot_range, desired_y_rot_range

    # Scale down proportionally
    scale_factor = max_allowable_offset / max_offset_desired

    tan_y_bounded = tan_y_desired * scale_factor
    tan_x_bounded = tan_x_desired * scale_factor

    max_abs_y_bounded = np.degrees(np.arctan(tan_y_bounded))
    max_abs_x_bounded = np.degrees(np.arctan(tan_x_bounded))

    # Scale original ranges proportionally
    def scale_range(orig_range, max_abs):
        orig_max_abs = max(abs(orig_range[0]), abs(orig_range[1]))
        if orig_max_abs == 0:
            return 0.0, 0.0
        ratio = max_abs / orig_max_abs
        return orig_range[0] * ratio, orig_range[1] * ratio

    bounded_x = scale_range(orig_range=desired_x_rot_range, max_abs=max_abs_x_bounded)
    bounded_y = scale_range(orig_range=desired_y_rot_range, max_abs=max_abs_y_bounded)

    return bounded_x, bounded_y
