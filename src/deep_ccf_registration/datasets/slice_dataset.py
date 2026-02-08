import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ants
import boto3
import numpy as np
import tensorstore
from botocore import UNSIGNED
from botocore.config import Config
from loguru import logger
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset

from deep_ccf_registration.datasets.aquisition_meta import AcquisitionDirection
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import map_points_to_right_hemisphere, \
    apply_crop_pad_to_original, get_subject_rotation_range
from deep_ccf_registration.datasets.utils.template_points import transform_points_to_template_space, apply_transforms_to_points, Affine
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation, TissueBoundingBoxes, \
    AcquisitionAxis, RotationAngles, SubjectRotationAngle
from deep_ccf_registration.utils.logging_utils import timed, timed_func
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore


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
    # Eval template points at original resolution (no interpolation)
    eval_template_points: Optional[np.ndarray] = None
    eval_tissue_mask: Optional[np.ndarray] = None
    eval_shape: Optional[tuple[int, int]] = None  # (H, W) of eval targets

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
        template_parameters: TemplateParameters,
        tensorstore_aws_credentials_method: str,
        is_train: bool,
        tissue_bboxes: TissueBoundingBoxes,
        rotation_angles: RotationAngles,
        orientations: list[SliceOrientation],
        crop_size: Optional[tuple[int, int]] = None,
        registration_downsample_factor: int = 3,
        transform: Optional[callable] = None,
        include_tissue_mask: bool = False,
        ccf_annotations: Optional[np.ndarray] = None,
        scratch_path: Path = Path('/tmp'),
        rotate_slices: bool = False,
        is_debug: bool = False,
        debug_slice_idx: Optional[int] = None,
    ):
        if include_tissue_mask and ccf_annotations is None:
            raise ValueError("include_tissue_mask=True requires ccf_annotations")

        self._all_subjects = subjects
        self._template_parameters = template_parameters
        self._aws_credentials_method = tensorstore_aws_credentials_method
        self._registration_downsample_factor = registration_downsample_factor
        self._transform = transform
        self._include_tissue_mask = include_tissue_mask
        self._ccf_annotations = ccf_annotations
        self._orientations = orientations

        self._loaded_subject_id: Optional[str] = None
        self._volume: Optional[np.ndarray] = None
        self._warp: Optional[np.ndarray] = None
        self._cached_affine: Optional[Affine] = None
        self._tissue_bboxes = tissue_bboxes.bounding_boxes
        self._crop_size = crop_size
        self._scratch_path = scratch_path
        self._rotate_slices = rotate_slices
        self._rotation_angles = rotation_angles

        # Precompute valid slices per subject
        self._valid_slices_cache: dict[str, list[int]] = {}
        for s in self._all_subjects:
            bboxes = self._tissue_bboxes[s.subject_id]
            valid = [i for i, b in enumerate(bboxes) if b is not None]
            self._valid_slices_cache[s.subject_id] = valid

        # Debug mode: restrict to a single slice per subject
        if is_debug:
            for subject_id, valid_slices in self._valid_slices_cache.items():
                if debug_slice_idx is not None:
                    if debug_slice_idx in valid_slices:
                        self._valid_slices_cache[subject_id] = [debug_slice_idx]
                    else:
                        # Fall back to middle slice if debug_slice_idx not valid
                        self._valid_slices_cache[subject_id] = [valid_slices[len(valid_slices) // 2]]
                else:
                    # Default debug: use middle slice
                    self._valid_slices_cache[subject_id] = [valid_slices[len(valid_slices) // 2]]

        # Build flat index mapping for map-style access
        self._index_map: list[tuple[SubjectMetadata, int, SliceOrientation]] = []
        for subject in self._all_subjects:
            valid_slices = self._valid_slices_cache[subject.subject_id]
            for slice_idx in valid_slices:
                for orientation in self._orientations:
                    self._index_map.append((subject, slice_idx, orientation))
        self._epoch_length = len(self._index_map)

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, index: int) -> PatchSample:
        if index >= len(self._index_map):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._index_map)}")
        subject, slice_idx, orientation = self._index_map[index]
        self._ensure_subject_loaded(subject)
        spec = SliceSampleSpec(metadata=subject, slice_idx=slice_idx, orientation=orientation)
        return self._load_slice(spec=spec)

    def _load_slice(self, spec: SliceSampleSpec) -> PatchSample:
        metadata = spec.metadata
        slice_axis = metadata.get_slice_axis(spec.orientation)
        axes = sorted(metadata.axes, key=lambda axis: axis.dimension)
        in_plane_axes = [ax for ax in axes if ax.dimension != slice_axis.dimension]
        if len(in_plane_axes) != 2:
            raise ValueError("Expected exactly two in-plane axes")
        y_axis, x_axis = in_plane_axes

        bbox = self._tissue_bboxes[metadata.subject_id][spec.slice_idx]

        start_y = bbox.y
        start_x = bbox.x
        patch_height = bbox.height
        patch_width = bbox.width


        coordinate_grid = self._get_coordinate_grid(
            experiment_meta=metadata,
            start_y=start_y,
            start_x=start_x,
            height=patch_height,
            width=patch_width,
            slice_axis=slice_axis,
            fixed_index_value=spec.slice_idx,
            orientation=spec.orientation,
            subject_bboxes=self._tissue_bboxes[metadata.subject_id],
        )

        if self._rotate_slices:
            # Interpolate volume at rotated coordinate locations
            # coordinate_grid: (n_points, 3), map_coordinates needs (3, n_points)
            coords_for_interp = coordinate_grid.T
            # Volume shape: (C, T, D0, D1, D2) - sample from spatial dims
            volume_3d = self._volume[0, 0]

            interpolated_flat = map_coordinates(
                input=volume_3d,
                coordinates=coords_for_interp,
                order=1,  # linear interpolation
                mode='constant',
                cval=0.0
            )
            data_patch = interpolated_flat.reshape(patch_height, patch_width).astype("float32")
        else:
            spatial_slices = [0, 0, slice(None), slice(None), slice(None)]
            spatial_slices[2 + slice_axis.dimension] = spec.slice_idx
            spatial_slices[2 + y_axis.dimension] = slice(start_y, start_y + patch_height)
            spatial_slices[2 + x_axis.dimension] = slice(start_x, start_x + patch_width)
            data_patch = self._volume[tuple(spatial_slices)].astype("float32")

        template_patch = self._get_template_points(
            point_grid=coordinate_grid,
            patch_height=patch_height,
            patch_width=patch_width,
            experiment_meta=metadata,
        )

        template_patch = map_points_to_right_hemisphere(
            template_points=template_patch,
            template_parameters=self._template_parameters,
        )

        tissue_mask = None
        if self._include_tissue_mask and self._ccf_annotations is not None:
            tissue_mask = _get_tissue_mask(
                annotations=self._ccf_annotations,
                template_patch=template_patch,
                template_parameters=self._template_parameters,
            )

        # Store original template points before transforms for eval
        original_template_points = template_patch.copy()
        original_tissue_mask = tissue_mask.copy() if tissue_mask is not None else None
        original_shape = (data_patch.shape[0], data_patch.shape[1])

        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        eval_template_points = None
        eval_tissue_mask = None
        eval_shape = None
        if self._transform is not None:
            transforms = self._transform(
                image=data_patch,
                template_coords=template_patch,
                mask=tissue_mask,
                slice_axis=slice_axis,
                acquisition_axes=metadata.axes,
                orientation=spec.orientation,
                subject_rotation=self._rotation_angles.rotation_angles[metadata.subject_id],
                subject_id=metadata.subject_id,
                slice_idx=spec.slice_idx,
            )
            data_patch = transforms["image"]
            template_patch = transforms["template_coords"]
            tissue_mask = transforms["mask"]

            # Get the shape after resize transforms (before crop/pad)
            # This is needed to scale crop coordinates to original resolution
            resized_shape = self._get_shape_after_resize(transforms["replay"], original_shape)

            # Apply crop/pad to original points, scaling coordinates appropriately
            # This allows evaluation at original resolution without interpolating targets
            eval_template_points, eval_tissue_mask, eval_shape = apply_crop_pad_to_original(
                template_coords=original_template_points,
                replay=transforms["replay"],
                original_shape=original_shape,
                resized_shape=resized_shape,
                mask=original_tissue_mask,
            )

            # Extract padding info from replay
            for t in transforms["replay"]["transforms"]:
                if t.get('params') is None:
                    params = {}
                else:
                    params = t.get("params", {})
                if "pad_params" in params:
                    pad_params = t["params"]["pad_params"]
                    if pad_params is not None:
                        pad_top = pad_params.get("pad_top", 0)
                        pad_bottom = pad_params.get("pad_bottom", 0)
                        pad_left = pad_params.get("pad_left", 0)
                        pad_right = pad_params.get("pad_right", 0)
                        break

        return PatchSample(
            slice_idx=spec.slice_idx,
            start_y=start_y,
            start_x=start_x,
            data=data_patch,
            template_points=template_patch,
            dataset_idx=metadata.subject_id,
            orientation=spec.orientation.value,
            subject_id=metadata.subject_id,
            tissue_mask=tissue_mask,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            eval_template_points=eval_template_points,
            eval_tissue_mask=eval_tissue_mask,
            eval_shape=eval_shape,
        )

    @timed_func
    def _get_template_points(
        self,
        point_grid: np.ndarray,
        patch_height: int,
        patch_width: int,
        experiment_meta: SubjectMetadata,
    ) -> np.ndarray:

        with timed():
            points = transform_points_to_template_space(
                points=point_grid,
                input_volume_shape=self._volume.shape[2:],
                acquisition_axes=experiment_meta.axes,
                ls_template_info=self._template_parameters,
                registration_downsample=experiment_meta.registration_downsample
            )

        with timed():
            template_points = apply_transforms_to_points(
                points=points,
                template_parameters=self._template_parameters,
                cached_affine=self._cached_affine,
                warp=self._warp,
            )

        with timed():
            template_points = template_points.reshape((patch_height, patch_width, 3))

        return template_points

    def _get_shape_after_resize(self, replay: dict, original_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Compute the image shape after resize transforms (Resample, LongestMaxSize).

        This is needed to properly scale crop coordinates when applying them
        to original-resolution targets.
        """
        h, w = original_shape

        for t in replay.get("transforms", []):
            t_name = t.get("__class_fullname__", "")
            params = t.get("params", {}) or {}

            if "Resample" in t_name:
                shape = params.get("shape")
                if shape:
                    h, w = shape[0], shape[1]

            elif "LongestMaxSize" in t_name:
                scale = params.get("scale", 1.0)
                new_h = params.get("height")
                new_w = params.get("width")
                if new_h is not None and new_w is not None:
                    h, w = new_h, new_w
                elif scale != 1.0:
                    h = int(round(h * scale))
                    w = int(round(w * scale))

            elif "Crop" in t_name:
                break

        return (h, w)

    def _ensure_subject_loaded(self, metadata: SubjectMetadata):
        subject_id = metadata.subject_id
        if self._loaded_subject_id == subject_id:
            return
        logger.debug(f"Loading full volume for subject {subject_id}")
        self._volume = self._load_full_volume(metadata)
        logger.debug(f"Loading warp for subject {subject_id}")
        self._warp = self._load_warp(metadata=metadata)
        self._cached_affine = Affine.from_ants_file(metadata.ls_to_template_affine_matrix_path)
        self._loaded_subject_id = subject_id

    def _load_full_volume(self, metadata: SubjectMetadata) -> np.ndarray:
        volume_dir = self._scratch_path / 'volumes'
        volume_dir.mkdir(parents=True, exist_ok=True)
        npy_path = volume_dir / f'{metadata.subject_id}.npy'

        if npy_path.exists():
            logger.debug(f'Loading volume from numpy cache: {npy_path}')
            return np.load(str(npy_path), mmap_mode='r')

        # First time: load from tensorstore
        store = tensorstore.open(
            spec={
                "driver": "auto",
                "kvstore": create_kvstore(
                    path=f"{metadata.stitched_volume_path}/{self._registration_downsample_factor}",
                    aws_credentials_method="anonymous",
                ),
            },
            read=True,
        ).result()
        data = np.array(store[...].read().result())

        np.save(str(npy_path), data)
        logger.debug(f'Saved volume numpy cache: {npy_path}')

        return np.load(str(npy_path), mmap_mode='r')

    def _load_warp(self, metadata: SubjectMetadata) -> np.ndarray:
        warp_dir = self._scratch_path / 'warps'
        warp_dir.mkdir(parents=True, exist_ok=True)

        # Check for .npy cache first (fast path, already transposed)
        npy_cache_path = warp_dir / f'{metadata.subject_id}_warp.npy'
        if npy_cache_path.exists():
            logger.debug(f'Loading warp from numpy cache: {npy_cache_path}')
            return np.load(str(npy_cache_path))

        # Download/copy NIfTI if needed
        nifti_local_path = warp_dir / f'{metadata.subject_id}_{metadata.ls_to_template_inverse_warp_path_original.name}'
        if not nifti_local_path.exists():
            logger.debug(
                f'Copying {metadata.ls_to_template_inverse_warp_path_original} to {nifti_local_path}')
            if str(metadata.ls_to_template_inverse_warp_path_original).startswith('/data/aind_open_data'):
                s3 = boto3.client(
                    's3',
                    config=Config(signature_version=UNSIGNED),
                    region_name='us-west-2')
                s3.download_file('aind-open-data',
                                 str(metadata.ls_to_template_inverse_warp_path_original.relative_to('/data/aind_open_data')),
                                 str(nifti_local_path))
            else:
                shutil.copy(metadata.ls_to_template_inverse_warp_path_original, nifti_local_path)

        # Read via ANTs, transpose, save as .npy
        warp_raw = ants.image_read(str(nifti_local_path)).numpy()
        warp_transposed = np.ascontiguousarray(warp_raw.transpose(3, 0, 1, 2))

        np.save(npy_cache_path, warp_transposed)
        logger.debug(f'Saved warp numpy cache: {npy_cache_path}')

        return warp_transposed

    def _get_coordinate_grid(
        self,
        experiment_meta: SubjectMetadata,
        start_x: int,
        start_y: int,
        height: int,
        width: int,
        fixed_index_value: int,
        slice_axis: AcquisitionAxis,
        orientation: Optional[SliceOrientation] = None,
        subject_bboxes: Optional[list] = None,

    ):
        if self._rotate_slices:
            if orientation is None:
                raise ValueError('provide orientation')
            slice_rotation_ranges = self._get_slice_rotation_ranges(
                metadata=experiment_meta,
                orientation=orientation
            )

            # Compute tissue depth range from bounding boxes
            tissue_slices = [i for i, x in enumerate(subject_bboxes) if x is not None]
            tissue_min = min(tissue_slices)
            tissue_max = max(tissue_slices)
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
        return points

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


# Also expose as standalone for tests
def get_slice_rotation_ranges(
    metadata: SubjectMetadata,
    orientation: SliceOrientation,
    rotation_angles: Optional[RotationAngles] = None,
) -> SliceRotationRanges:
    """Standalone version of _get_slice_rotation_ranges for testing.

    When rotation_angles is not provided, uses hardcoded default ranges.
    """
    if rotation_angles is None:
        # Default ranges for testing
        AP_rot_range = (-10, 10)
        SI_rot_range = (-10, 10)
        ML_rot_range = (-10, 10)
    else:
        from deep_ccf_registration.datasets.transforms import get_subject_rotation_range as _get_range
        subject_rotation = rotation_angles.rotation_angles[metadata.subject_id]
        AP_rot_range = _get_range(subject_angle=subject_rotation.AP_rot, valid_range=rotation_angles.AP_range)
        SI_rot_range = _get_range(subject_angle=subject_rotation.SI_rot, valid_range=rotation_angles.SI_range)
        ML_rot_range = _get_range(subject_angle=subject_rotation.ML_rot, valid_range=rotation_angles.ML_range)

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

    bounded_x = scale_range(desired_x_rot_range, max_abs_x_bounded)
    bounded_y = scale_range(desired_y_rot_range, max_abs_y_bounded)

    return bounded_x, bounded_y
