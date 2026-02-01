import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import ants
import numpy as np
import tensorstore
from loguru import logger
from scipy.ndimage import map_coordinates
from torch.utils.data import IterableDataset, get_worker_info

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import map_points_to_right_hemisphere
from deep_ccf_registration.datasets.utils.template_points import create_coordinate_grid, \
    transform_points_to_template_space, apply_transforms_to_points, Affine
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation, TissueBoundingBoxes, \
    AcquisitionAxis
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
    worker_id: int = -1
    orientation: str = ""
    subject_id: str = ""
    tissue_mask: Optional[np.ndarray] = None
    eval_template_points: Optional[np.ndarray] = None
    eval_tissue_mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for batch collation."""
        return {
            "slice_idx": self.slice_idx,
            "start_y": self.start_y,
            "start_x": self.start_x,
            "data": self.data,
            "template_points": self.template_points,
            "dataset_idx": self.dataset_idx,
            "worker_id": self.worker_id,
            "orientation": self.orientation,
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True)
class SliceSampleSpec:
    """Slice to sample"""

    metadata: SubjectMetadata
    slice_idx: int
    orientation: SliceOrientation
    start_y: Optional[int] = None
    start_x: Optional[int] = None

class IterableSubjectSliceDataset(IterableDataset):
    """Iterable dataset that generates slices on-the-fly from a generator function."""

    def __init__(
        self,
        slice_generator: Callable[[], Iterator[SliceSampleSpec]],
        template_parameters: TemplateParameters,
        tensorstore_aws_credentials_method: str,
        is_train: bool,
        tissue_bboxes: TissueBoundingBoxes,
        crop_size: Optional[tuple[int, int]] = None,
        registration_downsample_factor: int = 3,
        transform: Optional[callable] = None,
        target_eval_transform: Optional[callable] = None,
        include_tissue_mask: bool = False,
        ccf_annotations: Optional[np.ndarray] = None,
        scratch_path: Path = Path('/tmp')
    ):
        if include_tissue_mask and ccf_annotations is None:
            raise ValueError("include_tissue_mask=True requires ccf_annotations")

        self._slice_generator = slice_generator
        self._template_parameters = template_parameters
        self._aws_credentials_method = tensorstore_aws_credentials_method
        self._registration_downsample_factor = registration_downsample_factor
        self._transform = transform
        self._target_eval_transform = target_eval_transform
        self._include_tissue_mask = include_tissue_mask
        self._ccf_annotations = ccf_annotations

        self._loaded_subject_id: Optional[str] = None
        self._volume: Optional[np.ndarray] = None
        self._warp: Optional[np.ndarray] = None
        self._tissue_bboxes = tissue_bboxes.bounding_boxes
        self._crop_size = crop_size
        self._is_train = is_train
        self._scratch_path = scratch_path

    def __iter__(self) -> Iterator[PatchSample]:
        for spec in self._slice_generator():
            self._ensure_subject_loaded(spec.metadata)
            yield self._load_slice(spec=spec)

    def _load_slice(self, spec: SliceSampleSpec) -> PatchSample:
        metadata = spec.metadata
        slice_axis = metadata.get_slice_axis(spec.orientation)
        axes = sorted(metadata.axes, key=lambda axis: axis.dimension)
        in_plane_axes = [ax for ax in axes if ax.dimension != slice_axis.dimension]
        if len(in_plane_axes) != 2:
            raise ValueError("Expected exactly two in-plane axes")
        y_axis, x_axis = in_plane_axes

        bbox = self._tissue_bboxes[metadata.subject_id][spec.slice_idx]
        if self._crop_size is None:
            start_y = bbox.y
            start_x = bbox.x
            patch_height = bbox.height
            patch_width = bbox.width
        else:
            if spec.start_y is not None and spec.start_x is not None:
                start_y, start_x = spec.start_y, spec.start_x
            else:
                start_y = random.randint(bbox.y, max(bbox.y, bbox.y + bbox.height - self._crop_size[0]))
                start_x = random.randint(bbox.x, max(bbox.x, bbox.x + bbox.width - self._crop_size[1]))
            patch_height = min(self._crop_size[0], bbox.y + bbox.height - start_y)
            patch_width = min(self._crop_size[1], bbox.x + bbox.width - start_x)
        
        spatial_slices = [0, 0, slice(None), slice(None), slice(None)]
        spatial_slices[2 + slice_axis.dimension] = spec.slice_idx
        spatial_slices[2 + y_axis.dimension] = slice(start_y, start_y + patch_height)
        spatial_slices[2 + x_axis.dimension] = slice(start_x, start_x + patch_width)

        data_patch = self._volume[tuple(spatial_slices)].astype("float32")
        template_patch = self._get_template_points(
            patch_height=patch_height,
            patch_width=patch_width,
            start_x=start_x,
            start_y=start_y,
            slice_axis=slice_axis,
            fixed_index_value=spec.slice_idx,
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

        eval_template_patch = None
        eval_tissue_mask = None
        if self._target_eval_transform is not None:
            eval_template_patch = template_patch.copy()
            eval_transform = self._target_eval_transform(
                image=data_patch,
                keypoints=eval_template_patch,
                mask=tissue_mask,
                slice_axis=slice_axis,
                acquisition_axes=metadata.axes,
                orientation=spec.orientation,
            )
            eval_template_patch = eval_transform["keypoints"]
            if self._ccf_annotations is not None:
                eval_tissue_mask = _get_tissue_mask(
                    annotations=self._ccf_annotations,
                    template_patch=eval_template_patch,
                    template_parameters=self._template_parameters,
                )

        if self._transform is not None:
            transforms = self._transform(
                image=data_patch,
                keypoints=template_patch,
                mask=tissue_mask,
                slice_axis=slice_axis,
                acquisition_axes=metadata.axes,
                orientation=spec.orientation,
            )
            data_patch = transforms["image"]
            template_patch = transforms["keypoints"]
            tissue_mask = transforms["mask"]

        worker_ctx = self._worker_context()

        return PatchSample(
            slice_idx=spec.slice_idx,
            start_y=start_y,
            start_x=start_x,
            data=data_patch,
            template_points=template_patch,
            dataset_idx=metadata.subject_id,
            worker_id=worker_ctx.worker_id,
            orientation=spec.orientation.value,
            subject_id=metadata.subject_id,
            tissue_mask=tissue_mask,
            eval_template_points=eval_template_patch,
            eval_tissue_mask=eval_tissue_mask,
        )

    @timed_func
    def _get_template_points(
        self,
        patch_height: int,
        patch_width: int,
        start_x: int,
        start_y: int,
        fixed_index_value: int,
        slice_axis: AcquisitionAxis,
        experiment_meta: SubjectMetadata,
    ) -> np.ndarray:

        with timed():
            point_grid = create_coordinate_grid(
                patch_height=patch_height,
                patch_width=patch_width,
                start_x=start_x,
                start_y=start_y,
                fixed_index_value=fixed_index_value,
                axes=experiment_meta.axes,
                slice_axis=slice_axis
            )

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


    @dataclass(frozen=True)
    class _WorkerContext:
        worker_id: int
        num_workers: int

    def _worker_context(self) -> _WorkerContext:
        worker_info = get_worker_info()
        if worker_info is None:
            return IterableSubjectSliceDataset._WorkerContext(worker_id=0, num_workers=1)
        return IterableSubjectSliceDataset._WorkerContext(
            worker_id=worker_info.id,
            num_workers=worker_info.num_workers,
        )

    def _ensure_subject_loaded(self, metadata: SubjectMetadata):
        subject_id = metadata.subject_id
        if self._loaded_subject_id == subject_id:
            return
        logger.info(f"Loading full volume for subject {subject_id}")
        self._volume = self._load_full_volume(metadata)
        logger.info(f"Loading warp for subject {subject_id}")
        warp = ants.image_read(str(metadata.ls_to_template_inverse_warp_path_original)).numpy()
        # This apparently improves efficiency when map_coordinates is called on each x,y,z offset dimension
        self._warp = np.ascontiguousarray(warp.transpose(3, 0, 1, 2))
        self._cached_affine = Affine.from_ants_file(metadata.ls_to_template_affine_matrix_path)
        self._loaded_subject_id = subject_id

    def _load_full_volume(self, metadata: SubjectMetadata) -> np.ndarray:
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
        data = store[...].read().result()
        return np.array(data)

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