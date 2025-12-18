"""Dataset caches for extracting random tissue-aligned patches from volumes."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Callable

import numpy as np
import tensorstore
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from deep_ccf_registration.datasets.slice_dataset import (
    TissueBoundingBox,
)
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore

@dataclass(frozen=True)
class PatchSample:
    slice_idx: int
    start_y: int
    start_x: int
    data: np.ndarray
    dataset_idx: int = -1
    worker_id: int = -1


class SliceDatasetCache(IterableDataset):
    """Cache and stream patches for a single SmartSPIM volume."""

    def __init__(
        self,
        dataset_meta: SubjectMetadata,
        tissue_bboxes: Sequence[Optional[TissueBoundingBox]],
        sample_fraction: float = 0.25,
        orientation: Optional[SliceOrientation] = None,
        tensorstore_aws_credentials_method: str = "anonymous",
        registration_downsample_factor: int = 3,
        patch_size: int = 512,
        chunk_size: int = 128,
        debug_fixed_slice_idx: Optional[int] = None,
        transform: Optional[Callable] = None,
        max_chunks_per_dataset: Optional[int] = None,
    ):
        super().__init__()

        if not tissue_bboxes:
            raise ValueError("SliceDatasetCache requires at least one TissueBoundingBox entry")
        if all(bbox is None for bbox in tissue_bboxes):
            raise ValueError("SliceDatasetCache requires at least one non-empty TissueBoundingBox")

        self._metadata = dataset_meta
        self._dataset_idx = dataset_meta.subject_id

        self._tissue_bboxes: list[Optional[TissueBoundingBox]] = list(tissue_bboxes)
        self._sample_fraction = sample_fraction
        self._orientation = orientation or SliceOrientation.SAGITTAL
        self._registration_downsample_factor = registration_downsample_factor
        self._tensorstore_aws_credentials_method = tensorstore_aws_credentials_method
        self._patch_size = patch_size
        self._chunk_size = chunk_size
        self._debug_fixed_slice_idx = debug_fixed_slice_idx

        self._volume: Optional[tensorstore.TensorStore] = None
        self._valid_slices = self._build_valid_slices()
        if not self._valid_slices:
            raise ValueError(
                f"No valid tissue slices found for dataset {self._dataset_idx}"
            )

        self._cached_chunks: Optional[np.ndarray] = None
        self._cached_region: Optional[dict[str, Any]] = None
        self._samples_per_chunk = max(1, int(self._chunk_size * self._sample_fraction))

        self._worker_id = 0
        self._num_workers = 1

        self._patch_buffer: list[PatchSample] = []
        self._chunk_counter = 0
        self._transform = transform
        if max_chunks_per_dataset is not None and max_chunks_per_dataset < 1:
            raise ValueError("max_chunks_per_dataset must be >= 1 or None")
        self._max_chunks_per_dataset = max_chunks_per_dataset

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
                        aws_credentials_method=self._tensorstore_aws_credentials_method,
                    ),
                },
                read=True,
            ).result()
        return self._volume

    def _build_valid_slices(self) -> list[tuple[int, TissueBoundingBox]]:
        subject_bboxes = self._tissue_bboxes
        valid_slices: list[tuple[int, TissueBoundingBox]] = []
        for slice_idx, bbox in enumerate(subject_bboxes):
            if bbox is None:
                continue
            valid_slices.append((slice_idx, bbox))
        return valid_slices

    def load_chunk_region(self) -> dict[str, Any]:
        """Load a random chunk from the volume, positioned to overlap with tissue."""
        meta = self._metadata
        volume = self._get_volume()
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

        # Pick a random slice range that has valid tissue slices
        slice_indices = sorted([s[0] for s in valid_slices])
        min_valid_slice = min(slice_indices)
        max_valid_slice = max(slice_indices)

        debug_slice = self._debug_fixed_slice_idx
        if debug_slice is not None:
            clamped_target = max(0, min(vol_d - 1, debug_slice))
            start_slice = min(clamped_target, vol_d - chunk_d)
            start_slice = max(0, start_slice)
        else:
            start_slice = random.randint(min_valid_slice, min(vol_d - chunk_d, max_valid_slice))

        # Get bboxes for slices that will be in this chunk
        slices_in_chunk = [
            (idx, bbox) for idx, bbox in valid_slices
            if start_slice <= idx < start_slice + chunk_d
        ]

        if not slices_in_chunk:
            raise ValueError(
                "No tissue slices fell within the chunk boundaries. "
            )

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
        end_slice = start_slice + chunk_d

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
        self._cached_chunks = np.array(chunk_data)

        logger.debug(
            f"Worker {self._worker_id} cached chunk for dataset {self._dataset_idx}: "
            f"slices {start_slice}-{end_slice}, y {start_y}-{end_y}, x {start_x}-{end_x}"
        )

        # Get bboxes for slices in this range
        slice_bboxes = {
            idx - start_slice: bbox
            for idx, bbox in valid_slices
            if start_slice <= idx < end_slice
        }

        self._cached_region = {
            "dataset_idx": self._dataset_idx,
            "worker_id": self._worker_id,
            "start_slice": start_slice,
            "end_slice": end_slice,
            "start_y": start_y,
            "end_y": end_y,
            "start_x": start_x,
            "end_x": end_x,
            "chunk_h": chunk_h,
            "chunk_w": chunk_w,
            "chunk_d": chunk_d,
            "slice_axis": slice_axis.dimension,
            "slice_bboxes": dict(slice_bboxes),
            "chunk_id": -1,
        }

        return self._cached_region

    def _extract_slice(self, local_idx: int, slice_axis: int) -> np.ndarray:
        """Extract a 2D slice from the cached 3D chunk."""
        if slice_axis == 0:
            arr = self._cached_chunks[local_idx, :, :]
        elif slice_axis == 1:
            arr = self._cached_chunks[:, local_idx, :]
        else:
            arr = self._cached_chunks[:, :, local_idx]
        arr = arr.astype('float32')
        return arr

    def _build_patch_output(
        self, global_slice_idx: int, start_y: int, start_x: int, patch: np.ndarray
    ) -> PatchSample:
        """Build a structured patch sample."""
        return PatchSample(
            slice_idx=global_slice_idx,
            start_y=start_y,
            start_x=start_x,
            data=patch,
            dataset_idx=self._dataset_idx,
            worker_id=self._worker_id,
        )

    def sample_patches_from_cache(self) -> list[PatchSample]:
        """
        Sample 2D patches from the cached chunk.

        Returns a list of ``PatchSample`` objects that capture the global slice index,
        absolute top-left coordinates, pixel data, and (optionally) the chunk metadata
        used to generate the patch.
        """
        if self._cached_chunks is None:
            return []

        info = self._cached_region
        if info is None:
            return []

        slice_bboxes = info["slice_bboxes"]
        chunk_start_y = info["start_y"]
        chunk_start_x = info["start_x"]
        start_slice = info["start_slice"]
        slice_axis = info["slice_axis"]

        if self._debug_fixed_slice_idx is not None:
            return self._sample_debug_patch(slice_bboxes, info)

        slice_range = range(info["chunk_d"])

        # Sample a fraction of the available slices
        target_samples = max(1, int(len(slice_range) * self._sample_fraction))
        target_samples = min(target_samples, len(slice_range))

        # Shuffle slice order so we try different slices each call
        shuffled_slices = list(slice_range)
        random.shuffle(shuffled_slices)

        patches: list[PatchSample] = []
        for local_idx in shuffled_slices[:target_samples]:
            patch = self._extract_slice(local_idx, slice_axis)
            if self._transform is not None:
                patch = self._transform(image=patch)['image']
            global_slice_idx = start_slice + local_idx
            patches.append(self._build_patch_output(global_slice_idx, chunk_start_y, chunk_start_x, patch))

        return patches

    def _sample_debug_patch(
        self,
        slice_bboxes: dict[int, TissueBoundingBox],
        info: dict[str, Any],
    ) -> list[PatchSample]:
        target_slice = max(0, min(info["end_slice"] - 1, self._debug_fixed_slice_idx))
        start_slice = info["start_slice"]
        local_idx = target_slice - start_slice

        if local_idx not in slice_bboxes:
            raise ValueError(
                f"Debug slice {self._debug_fixed_slice_idx} is not cached in the current chunk."
            )

        patch = self._extract_slice(local_idx, info["slice_axis"])
        if patch.size == 0:
            raise ValueError("Debug slice produced an empty patch; reload chunk or adjust slice index.")

        global_slice_idx = start_slice + local_idx
        return [self._build_patch_output(global_slice_idx, info["start_y"], info["start_x"], patch)]

    def clear_cache(self):
        self._cached_chunks = None
        self._cached_region = None

    def _refill_patch_buffer(self) -> bool:
        """Load a new chunk and fill the patch buffer. Returns True if patches were loaded."""
        if self._max_chunks_per_dataset is not None and self._chunk_counter >= self._max_chunks_per_dataset:
            logger.debug(
                f"Worker {self._worker_id} reached chunk limit for dataset {self._dataset_idx}; stopping."
            )
            return False

        start_time = time.perf_counter()
        try:
            self.load_chunk_region()
        except ValueError:
            logger.warning(
                f"Worker {self._worker_id} found no patches for dataset {self._dataset_idx}; skipping."
            )
            return False

        self._chunk_counter += 1
        if self._cached_region is not None:
            self._cached_region["chunk_id"] = self._chunk_counter
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
                dataset_idx=patch.dataset_idx,
                worker_id=patch.worker_id,
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
        return self._cached_chunks is not None

    @property
    def dataset_idx(self) -> int:
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
