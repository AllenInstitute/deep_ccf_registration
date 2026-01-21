"""Dataset caches for extracting random tissue-aligned patches from volumes."""
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Callable

import numpy as np
import tensorstore
import torch
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from deep_ccf_registration.datasets.transforms import map_points_to_right_hemisphere
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation, TissueBoundingBox
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


@dataclass
class CachedRegion:
    """Metadata about a cached chunk region."""
    dataset_idx: str
    worker_id: int
    # per-volume-axis slices (z, y, x) defining the cached subvolume
    spatial_slices: tuple[slice, slice, slice]
    chunk_id: int = -1


class SliceDatasetCache(IterableDataset):
    """Cache and stream patches for a single SmartSPIM volume."""

    def __init__(
        self,
        dataset_meta: SubjectMetadata,
        tissue_bboxes: Sequence[Optional[TissueBoundingBox]],
        template_parameters: TemplateParameters,
        is_train: bool,
        sample_fraction: float = 0.25,
        orientation: Optional[SliceOrientation] = None,
        tensorstore_aws_credentials_method: str = "anonymous",
        registration_downsample_factor: int = 3,
        patch_size: int = 512,
        chunk_size: int = 128,
        transform: Optional[Callable] = None,
        max_chunks_per_dataset: Optional[int] = 1,
        forced_coords: Optional[tuple[str, int, str, int, int]] = None,
    ):
        super().__init__()

        if not tissue_bboxes:
            raise ValueError("SliceDatasetCache requires at least one TissueBoundingBox entry")
        if all(bbox is None for bbox in tissue_bboxes):
            raise ValueError("SliceDatasetCache requires at least one non-empty TissueBoundingBox")

        self._metadata = dataset_meta
        self._dataset_idx = dataset_meta.subject_id

        self._is_train = is_train
        self._tissue_bboxes: list[Optional[TissueBoundingBox]] = list(tissue_bboxes)
        self._sample_fraction = sample_fraction
        self._orientation = orientation or SliceOrientation.SAGITTAL
        self._registration_downsample_factor = registration_downsample_factor
        self._aws_credentials_method = tensorstore_aws_credentials_method
        self._patch_size = patch_size
        self._chunk_size = chunk_size
        self._forced_coords = forced_coords

        self._volume: Optional[tensorstore.TensorStore] = None
        self._template_points: Optional[tensorstore.TensorStore] = None
        self._valid_slices = self._build_valid_slices()
        if not self._valid_slices:
            raise ValueError(
                f"No valid tissue slices found for dataset {self._dataset_idx}"
            )

        self._cached_input_chunks: Optional[np.ndarray] = None
        self._cached_template_points: Optional[np.ndarray] = None
        self._cached_region: Optional[CachedRegion] = None
        self._samples_per_chunk = max(1, int(self._chunk_size * self._sample_fraction))

        self._worker_id = 0
        self._num_workers = 1

        self._patch_buffer: list[PatchSample] = []
        self._chunk_counter = 0
        self._transform = transform
        if max_chunks_per_dataset is not None and max_chunks_per_dataset < 1:
            raise ValueError("max_chunks_per_dataset must be >= 1 or None")
        self._max_chunks_per_dataset = max_chunks_per_dataset
        self._template_parameters = template_parameters

    def _get_volume(self) -> tensorstore.TensorStore:
        if self._volume is None:
            logger.debug(
                f"Worker {self._worker_id} opening volume for dataset {self._dataset_idx}"
            )
            meta = self._metadata
            self._volume = tensorstore.open(
                spec={
                    "driver": "auto",
                    "kvstore": create_kvstore(
                        path=str(meta.stitched_volume_path)
                        + f"/{self._registration_downsample_factor}",
                        aws_credentials_method="anonymous",
                    ),
                },
                read=True,
            ).result()
        return self._volume

    def _get_template_points_volume(self) -> Optional[tensorstore.TensorStore]:
        if self._template_points is None:
            logger.debug(
                f"Worker {self._worker_id} opening template points for dataset {self._dataset_idx}"
            )
            meta = self._metadata
            self._template_points = tensorstore.open(
                spec={
                    "driver": "auto",
                    "kvstore": create_kvstore(
                        path=str(meta.get_template_points_path()),
                        aws_credentials_method=self._aws_credentials_method,
                    ),
                },
                read=True,
            ).result()
        return self._template_points

    def _build_valid_slices(self) -> list[tuple[int, TissueBoundingBox]]:
        subject_bboxes = self._tissue_bboxes
        valid_slices: list[tuple[int, TissueBoundingBox]] = []
        for slice_idx, bbox in enumerate(subject_bboxes):
            if bbox is None:
                continue
            valid_slices.append((slice_idx, bbox))
        return valid_slices

    def load_chunk_region(self) -> CachedRegion:
        """Load a random chunk from the volume, positioned to overlap with tissue."""
        meta = self._metadata
        volume = self._get_volume()
        template_points_volume = self._get_template_points_volume()
        slice_axis = meta.get_slice_axis(self._orientation)

        valid_slices = self._valid_slices
        if not valid_slices:
            raise ValueError(f"No valid slices for dataset {self._dataset_idx}")

        vol_shape = volume.shape[2:]
        y_axis = [i for i in range(3) if i != slice_axis.dimension][0]
        x_axis = [i for i in range(3) if i != slice_axis.dimension][1]

        vol_h = vol_shape[y_axis]
        vol_w = vol_shape[x_axis]
        vol_d = vol_shape[slice_axis.dimension]

        chunk_h = min(self._patch_size, vol_h)
        chunk_w = min(self._patch_size, vol_w)
        chunk_d = min(self._chunk_size, vol_d)

        # If forced coords are provided for this dataset, lock chunk to contain that slice/patch
        if self._forced_coords is not None and self._forced_coords[0] == self._dataset_idx:
            target_slice = self._forced_coords[1]
            target_y = self._forced_coords[3]
            target_x = self._forced_coords[4]

            start_slice = max(0, min(target_slice, vol_d - 1))
            end_slice = min(start_slice + 1, vol_d)

            start_y = max(0, min(target_y, vol_h - chunk_h))
            end_y = min(start_y + chunk_h, vol_h)

            start_x = max(0, min(target_x, vol_w - chunk_w))
            end_x = min(start_x + chunk_w, vol_w)
        else:
            # Pick a random slice range that has valid tissue slices
            slice_indices = sorted([s[0] for s in valid_slices])
            min_valid_slice = min(slice_indices)
            max_valid_slice = max(slice_indices)

            start_slice = random.randint(min_valid_slice, min(vol_d - chunk_d, max_valid_slice))

            end_slice = start_slice + chunk_d

        # Get bboxes for slices that will be in this chunk
        slices_in_chunk = [
            (idx, bbox) for idx, bbox in valid_slices
            if start_slice <= idx < end_slice
        ]

        if not slices_in_chunk:
            raise ValueError(
                "No tissue slices fell within the chunk boundaries. "
            )

        if self._forced_coords is None or self._forced_coords[0] != self._dataset_idx:
            # Position chunk to overlap with tissue - use union of bboxes
            union_min_y = min(bbox.y for _, bbox in slices_in_chunk)
            union_min_x = min(bbox.x for _, bbox in slices_in_chunk)
            union_max_y = max(bbox.y + bbox.height for _, bbox in slices_in_chunk)
            union_max_x = max(bbox.x + bbox.width for _, bbox in slices_in_chunk)

            if union_max_y <= chunk_h:
                start_y = 0
            else:
                start_y = random.randint(union_min_y, max(union_max_y - chunk_h, union_min_y + 1))

            if union_max_x <= chunk_w:
                start_x = 0
            else:
                start_x = random.randint(union_min_x, max(union_max_x - chunk_w, union_min_x + 1))

            end_y = min(start_y + chunk_h, vol_h)
            end_x = min(start_x + chunk_w, vol_w)

        # Build slice for volume indexing
        slices = [0, 0, None, None, None]
        slices[slice_axis.dimension + 2] = slice(start_slice, end_slice)
        slices[y_axis + 2] = slice(start_y, end_y)
        slices[x_axis + 2] = slice(start_x, end_x)

        logger.debug(
            f"Worker {self._worker_id} loading chunk for dataset {self._dataset_idx} "
            f"(slice_axis={slice_axis.dimension}, chunk_d={chunk_d})"
        )

        chunk_data = volume[tuple(slices)].read().result()
        self._cached_input_chunks = np.array(chunk_data)

        template_chunk = template_points_volume[tuple(slices)].read().result()
        self._cached_template_points = np.array(template_chunk)

        logger.debug(
            f"Worker {self._worker_id} cached chunk for dataset {self._dataset_idx}: "
            f"slices {start_slice}-{end_slice}, y {start_y}-{end_y}, x {start_x}-{end_x}"
        )

        self._cached_region = CachedRegion(
            dataset_idx=self._dataset_idx,
            worker_id=self._worker_id,
            spatial_slices=tuple(slices[2:]),
        )

        return self._cached_region

    def _extract_slice(self, array: np.ndarray, local_idx: int, slice_axis: int) -> np.ndarray:
        """Extract a 2D slice from a cached 3D chunk."""
        if slice_axis == 0:
            arr = array[local_idx, :, :]
        elif slice_axis == 1:
            arr = array[:, local_idx, :]
        else:
            arr = array[:, :, local_idx]
        return arr.astype('float32')


    def sample_patches_from_cache(self) -> list[PatchSample]:
        """
        Sample 2D patches from the cached chunk.

        Returns a list of ``PatchSample`` objects that capture the global slice index,
        absolute top-left coordinates, pixel data, and (optionally) the chunk metadata
        used to generate the patch.
        """
        if self._cached_input_chunks is None or self._cached_template_points is None:
            return []

        region = self._cached_region
        if region is None:
            return []

        slice_axis = self._metadata.get_slice_axis(self._orientation)
        plane_axes = [dim for dim in range(3) if dim != slice_axis.dimension]
        plane_axis_a, plane_axis_b = plane_axes

        spatial_slices = region.spatial_slices

        slice_span = spatial_slices[slice_axis.dimension]
        plane_span_a = spatial_slices[plane_axis_a]
        plane_span_b = spatial_slices[plane_axis_b]

        slice_start = slice_span.start
        start_y = plane_span_a.start
        start_x = plane_span_b.start

        if self._forced_coords is not None and self._forced_coords[0] == self._dataset_idx:
            # Respect the requested top-left coords when the chunk was forced
            start_y = self._forced_coords[3]
            start_x = self._forced_coords[4]

        slice_range = range(slice_span.stop - slice_span.start)

        # If forced coords are provided for this dataset, extract that exact slice deterministically
        if self._forced_coords is not None and self._forced_coords[0] == self._dataset_idx:
            target_slice_global = self._forced_coords[1]
            local_idx = target_slice_global - slice_start
            if local_idx < 0 or local_idx >= len(slice_range):
                return []
            target_indices = [local_idx]
        else:
            # Sample a fraction of the available slices
            target_samples = max(1, int(len(slice_range) * self._sample_fraction))
            target_samples = min(target_samples, len(slice_range))

            # Shuffle slice order so we try different slices each call
            shuffled_slices = list(slice_range)
            random.shuffle(shuffled_slices)
            target_indices = shuffled_slices[:target_samples]

        patches: list[PatchSample] = []
        for local_idx in target_indices:
            patch = self._extract_slice(
                array=self._cached_input_chunks,
                local_idx=local_idx,
                slice_axis=slice_axis.dimension,
            )
            template_patch = self._extract_slice(
                array=self._cached_template_points,
                local_idx=local_idx,
                slice_axis=slice_axis.dimension,
            )

            template_patch = map_points_to_right_hemisphere(
                template_points=template_patch,
                template_parameters=self._template_parameters
            )

            if self._transform is not None:
                transforms = self._transform(image=patch, keypoints=template_patch, slice_axis=slice_axis, acquisition_axes=self._metadata.axes, orientation=self._orientation)
                patch = transforms['image']
                template_patch = transforms['keypoints']

            global_slice_idx = slice_start + local_idx
            patches.append(
                PatchSample(
                            slice_idx=global_slice_idx,
                            start_y=start_y,
                            start_x=start_x,
                            data=patch,
                            template_points=template_patch,
                            dataset_idx=self._dataset_idx,
                            worker_id=self._worker_id,
                            orientation=self._orientation.value,
                            subject_id=self._metadata.subject_id,
                )
            )

        return patches

    def clear_cache(self):
        self._cached_input_chunks = None
        self._cached_template_points = None
        self._cached_region = None

    def _refill_patch_buffer(self) -> bool:
        """Load a new chunk and fill the patch buffer. Returns True if patches were loaded."""
        if self._max_chunks_per_dataset is not None and self._chunk_counter >= self._max_chunks_per_dataset:
            logger.debug(
                f"Worker {self._worker_id} reached chunk limit for dataset {self._dataset_idx}; stopping."
            )
            return False

        start_time = time.perf_counter()
        self.load_chunk_region()

        self._chunk_counter += 1
        if self._cached_region is not None:
            self._cached_region.chunk_id = self._chunk_counter
        patches = self.sample_patches_from_cache()
        self.clear_cache()

        if not patches:
            logger.warning(
                f"Worker {self._worker_id} found no patches for dataset {self._dataset_idx}; skipping."
            )
            return False

        # Copy patches to avoid issues with cleared cache
        for patch in patches:
            copied = PatchSample(
                slice_idx=patch.slice_idx,
                start_y=patch.start_y,
                start_x=patch.start_x,
                data=np.array(patch.data, copy=True),
                template_points=
                    np.array(patch.template_points, copy=True)
                    if patch.template_points is not None
                    else None,
                dataset_idx=patch.dataset_idx,
                worker_id=patch.worker_id,
                orientation=patch.orientation,
                subject_id=patch.subject_id,
            )
            self._patch_buffer.append(copied)

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"Worker {self._worker_id} generated {len(patches)} patches "
            f"for dataset {self._dataset_idx} ({elapsed:.3f}s)"
        )
        return True

    def next_patch(self) -> Optional[PatchSample]:
        """Get the next patch, loading a new chunk if the buffer is empty."""
        if not self._patch_buffer and not self._refill_patch_buffer():
            return None

        if not self._patch_buffer:
            return None

        patch = self._patch_buffer.pop(0)
        logger.debug(
            f"Worker {self._worker_id} yielded slice {patch.slice_idx}, ds {self._metadata.subject_id} from cache "
            f"(buffer_remaining={len(self._patch_buffer)})"
        )
        return patch

    def bind_worker(self, worker_id: int, num_workers: int):
        self._worker_id = worker_id
        self._num_workers = max(1, num_workers)

    def __iter__(self) -> Iterator[PatchSample]:
        if not self._valid_slices:
            return

        while True:
            patch = self.next_patch()
            if patch is None:
                break
            yield patch

    @property
    def cached_region_info(self) -> Optional[dict[str, Any]]:
        return self._cached_region

    @property
    def has_cache(self) -> bool:
        return self._cached_input_chunks is not None and self._cached_template_points is not None

    @property
    def dataset_idx(self) -> str:
        return self._dataset_idx

    @property
    def metadata(self) -> SubjectMetadata:
        return self._metadata


class ShardedMultiDatasetCache(IterableDataset):
    """
    Wraps multiple SliceDatasetCache instances and shards them across workers.

    Each DataLoader worker will only iterate over a subset of the datasets,
    ensuring different workers load different subjects.
    """

    def __init__(self, datasets: Sequence[SliceDatasetCache]):
        super().__init__()
        self._datasets = list(datasets)

    def __iter__(self) -> Iterator[PatchSample]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Shard datasets across workers
        worker_datasets = self._datasets[worker_id::num_workers]

        logger.debug(
            f"Worker {worker_id}/{num_workers} assigned {len(worker_datasets)} datasets "
            f"(indices {list(range(worker_id, len(self._datasets), num_workers))})"
        )

        for ds in worker_datasets:
            ds.bind_worker(worker_id, num_workers)
            yield from ds


def collate_patch_samples(samples: list[PatchSample], patch_size: int, is_train: bool) -> dict:
    """
    Collate a list of PatchSample into a batched dictionary.

    Returns a dict with:
        - input_images: (B, 1, H, W) tensor
        - dataset_indices: list of dataset identifiers (strings)
        - slice_indices: (B,) tensor
        - patch_ys: (B,) tensor
        - patch_xs: (B,) tensor
        - orientations: list of str
        - subject_ids: list of str
        - pad_masks: (B, H, W) tensor indicating valid (non-padded) pixels
        - pad_mask_heights: (B,) tensor of valid pixel counts along H
        - pad_mask_widths: (B,) tensor of valid pixel counts along W
    """
    batch_size = len(samples)
    heights = [s.data.shape[0] for s in samples]
    widths = [s.data.shape[1] for s in samples]

    target_h, target_w = (max(max(heights, widths)), max(max(heights, widths)))

    image_dtype = samples[0].data.dtype
    template_dtype = samples[0].template_points.dtype if samples[0].template_points is not None else np.float32

    images = np.zeros((batch_size, 1, target_h, target_w), dtype=image_dtype)

    # in train mode we resize the target. For inference we don't modify this size
    template_points = np.zeros((batch_size, 3, target_h if is_train else patch_size, target_w if is_train else patch_size), dtype=template_dtype)
    pad_masks = np.zeros((batch_size, target_h if is_train else patch_size, target_w if is_train else patch_size), dtype=np.uint8)
    pad_mask_heights = np.zeros(batch_size, dtype=np.int32)
    pad_mask_widths = np.zeros(batch_size, dtype=np.int32)

    for idx, sample in enumerate(samples):
        img = sample.data
        img_h, img_w = img.shape
        template_points_h, template_points_w = sample.template_points.shape[:-1]
        images[idx, 0, :img_h, :img_w] = img

        if sample.template_points is None:
            raise ValueError("PatchSample missing template_points; cannot collate")
        sample_template_points = np.transpose(sample.template_points, (2, 0, 1))
        template_points[idx, :, :sample_template_points.shape[1], :sample_template_points.shape[2]] = sample_template_points

        pad_masks[idx, :template_points_h, :template_points_w] = 1
        pad_mask_heights[idx] = template_points_h
        pad_mask_widths[idx] = template_points_w

    return {
        "input_images": torch.from_numpy(images),
        "target_template_points": torch.from_numpy(template_points),
        "pad_masks": torch.from_numpy(pad_masks.astype(bool)),
        "pad_mask_heights": torch.from_numpy(pad_mask_heights),
        "pad_mask_widths": torch.from_numpy(pad_mask_widths),
        "dataset_indices": [s.dataset_idx for s in samples],
        "slice_indices": torch.tensor([s.slice_idx for s in samples]),
        "patch_ys": torch.tensor([s.start_y for s in samples]),
        "patch_xs": torch.tensor([s.start_x for s in samples]),
        "orientations": [s.orientation for s in samples],
        "subject_ids": [s.subject_id for s in samples],
    }


class ShuffledBatchIterator:
    """
    Wraps a DataLoader to provide shuffled batches from multiple workers.

    Collects samples into a buffer and yields shuffled batches, ensuring
    samples from different workers are mixed together.
    """

    def __init__(
        self,
        dataloader,
        batch_size: int,
        buffer_batches: int = 4,
    ):
        self._dataloader = dataloader
        self._batch_size = batch_size
        self._buffer_size = batch_size * buffer_batches

    def __iter__(self) -> Iterator[list[PatchSample]]:
        buffer: list[PatchSample] = []

        for batch in self._dataloader:
            buffer.extend(batch)

            while len(buffer) >= self._buffer_size:
                random.shuffle(buffer)
                yield buffer[:self._batch_size]
                buffer = buffer[self._batch_size:]

        # Yield remaining samples
        while len(buffer) >= self._batch_size:
            random.shuffle(buffer)
            yield buffer[:self._batch_size]
            buffer = buffer[self._batch_size:]

        # Yield final partial batch if any
        if buffer:
            random.shuffle(buffer)
            yield buffer
