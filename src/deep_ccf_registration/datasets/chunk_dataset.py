"""
ChunkDataset: Load a random 512x512x128 chunk, cache it, and sample 2D patches
that respect tissue bounding boxes.
"""
import random
from typing import Optional

import numpy as np
import tensorstore
from torch.utils.data import Dataset

from deep_ccf_registration.datasets.slice_dataset import TissueBoundingBox, TissueBoundingBoxes
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore


class SliceDatasetCache(Dataset):
    def __init__(
        self,
        dataset_meta: list[SubjectMetadata],
        tissue_bboxes: TissueBoundingBoxes,
        sample_fraction: float = 0.25,
        orientation: Optional[SliceOrientation] = None,
        tensorstore_aws_credentials_method: str = "anonymous",
        registration_downsample_factor: int = 3,
        patch_size: int = 512,
        chunk_size: int = 128,
        debug_fixed_slice_idx: Optional[int] = None,
    ):
        super().__init__()
        self._dataset_meta = dataset_meta
        self._tissue_bboxes = tissue_bboxes.bounding_boxes
        self._sample_fraction = sample_fraction
        self._orientation = orientation or SliceOrientation.SAGITTAL
        self._registration_downsample_factor = registration_downsample_factor
        self._tensorstore_aws_credentials_method = tensorstore_aws_credentials_method
        self._patch_size = patch_size
        self._chunk_size = chunk_size
        self._debug_fixed_slice_idx = debug_fixed_slice_idx

        self._volumes: dict[int, tensorstore.TensorStore] = {}
        self._valid_locations = self._build_dataset_tissue_slices_map()

        self._cached_chunks: Optional[np.ndarray] = None
        self._cached_region: Optional[dict] = None

    def _get_volume(self, dataset_idx: int) -> tensorstore.TensorStore:
        if dataset_idx not in self._volumes:
            meta = self._dataset_meta[dataset_idx]
            volume = tensorstore.open(
                spec={
                    'driver': 'auto',
                    'kvstore': create_kvstore(
                        path=str(meta.stitched_volume_path) + f'/{self._registration_downsample_factor}',
                        aws_credentials_method=self._tensorstore_aws_credentials_method
                    )
                },
                read=True
            ).result()
            self._volumes[dataset_idx] = volume
        return self._volumes[dataset_idx]

    def _build_dataset_tissue_slices_map(self) -> list[tuple[int, list[tuple[int, TissueBoundingBox]]]]:
        """
        returns map between dataset_idx and slices in dataset that contain tissue
        :return:
        """
        locations = []
        for dataset_idx, meta in enumerate(self._dataset_meta):
            subject_bboxes = self._tissue_bboxes.get(meta.subject_id, [])
            valid_slices = [
                (slice_idx, TissueBoundingBox(**bbox) if isinstance(bbox, dict) else bbox)
                for slice_idx, bbox in enumerate(subject_bboxes)
                if bbox is not None
            ]
            if valid_slices:
                locations.append((dataset_idx, valid_slices))
        return locations

    def load_chunk_region(self, dataset_idx: int) -> dict:
        """Load a random 512x512x128 chunk from the volume, positioned to overlap with tissue."""
        meta = self._dataset_meta[dataset_idx]
        volume = self._get_volume(dataset_idx)
        slice_axis = meta.get_slice_axis(self._orientation)

        _, valid_slices = next(
            (loc for loc in self._valid_locations if loc[0] == dataset_idx),
            (None, [])
        )

        if not valid_slices:
            raise ValueError(f"No valid slices for dataset {dataset_idx}")

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

        chunk_data = volume[tuple(slices)].read().result()
        self._cached_chunks = np.array(chunk_data)

        # Get bboxes for slices in this range
        slice_bboxes = {
            idx - start_slice: bbox
            for idx, bbox in valid_slices
            if start_slice <= idx < end_slice
        }

        self._cached_region = {
            'dataset_idx': dataset_idx,
            'start_slice': start_slice,
            'end_slice': end_slice,
            'start_y': start_y,
            'end_y': end_y,
            'start_x': start_x,
            'end_x': end_x,
            'chunk_h': chunk_h,
            'chunk_w': chunk_w,
            'chunk_d': chunk_d,
            'slice_axis': slice_axis.dimension,
            'slice_bboxes': slice_bboxes,
        }

        return self._cached_region

    def sample_patches_from_cache(self) -> list[tuple[int, int, int, np.ndarray]]:
        """
        Sample 2D patches from the cached chunk.

        Returns list of (global_slice_idx, patch_start_y, patch_start_x, patch) tuples.
        """
        if self._cached_chunks is None:
            return []

        info = self._cached_region
        slice_bboxes = info['slice_bboxes']
        chunk_start_y = info['start_y']
        chunk_start_x = info['start_x']
        chunk_h = info['chunk_h']
        chunk_w = info['chunk_w']
        start_slice = info['start_slice']
        slice_axis = info['slice_axis']

        patch_h = min(self._patch_size, chunk_h)
        patch_w = min(self._patch_size, chunk_w)

        if self._debug_fixed_slice_idx is not None:
            return self._sample_debug_patch(slice_bboxes, info, patch_h, patch_w)

        slice_range = range(info['chunk_d'])

        # Sample a fraction of the available slices, respecting sparsity in the cache
        target_samples = max(1, int(len(slice_range) * self._sample_fraction))
        target_samples = min(target_samples, len(slice_range))

        # Shuffle slice order so we try different slices each call, but keep sampling until
        # we reach the target or exhaust valid slices.
        shuffled_slices = list(slice_range)
        random.shuffle(shuffled_slices)

        patches: list[tuple[int, int, int, np.ndarray]] = []
        for local_idx in shuffled_slices:
            if slice_axis == 0:
                patch = self._cached_chunks[local_idx, :, :]
            elif slice_axis == 1:
                patch = self._cached_chunks[:, local_idx, :]
            else:
                patch = self._cached_chunks[:, :, local_idx]

            global_slice_idx = start_slice + local_idx
            patches.append((global_slice_idx, chunk_start_y, chunk_start_x, patch))

            if len(patches) >= target_samples:
                break

        return patches

    def _sample_debug_patch(
        self,
        slice_bboxes: dict[int, TissueBoundingBox],
        info: dict,
        patch_h: int,
        patch_w: int,
    ) -> list[tuple[int, int, int, np.ndarray]]:
        target_slice = max(0, min(info['end_slice'] - 1, self._debug_fixed_slice_idx))
        start_slice = info['start_slice']
        local_idx = target_slice - start_slice

        if local_idx not in slice_bboxes:
            raise ValueError(
                f"Debug slice {self._debug_fixed_slice_idx} is not cached in the current chunk."
            )

        slice_axis = info['slice_axis']

        if slice_axis == 0:
            patch = self._cached_chunks[local_idx, :, :]
        elif slice_axis == 1:
            patch = self._cached_chunks[:, local_idx, :]
        else:
            patch = self._cached_chunks[:, :, local_idx]

        if patch.size == 0:
            raise ValueError("Debug slice produced an empty patch; reload chunk or adjust slice index.")

        global_slice_idx = start_slice + local_idx
        return [(global_slice_idx, info['start_y'], info['start_x'], patch)]

    def clear_cache(self):
        self._cached_chunks = None
        self._cached_region = None

    def __len__(self) -> int:
        return len(self._valid_locations)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx, _ = self._valid_locations[idx]
        return self.load_chunk_region(dataset_idx)

    @property
    def cached_region_info(self) -> Optional[dict]:
        return self._cached_region

    @property
    def has_cache(self) -> bool:
        return self._cached_chunks is not None
