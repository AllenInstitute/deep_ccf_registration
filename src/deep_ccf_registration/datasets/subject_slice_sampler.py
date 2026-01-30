"""Sampler for generating slice specs with contiguous subject assignment across workers."""
import math
import random
from typing import Iterator, Optional, Sequence

from torch.utils.data import get_worker_info

from deep_ccf_registration.datasets.iterable_slice_dataset import SliceSampleSpec
from deep_ccf_registration.metadata import SliceOrientation, SubjectMetadata, TissueBoundingBoxes


class SubjectSliceSampler:
    """Sampler that generates slice specs with contiguous subject assignment across workers."""

    def __init__(
        self,
        subjects: Sequence[SubjectMetadata],
        tissue_bboxes: TissueBoundingBoxes,
        orientations: Sequence[SliceOrientation] = (SliceOrientation.SAGITTAL,),
        slice_fraction: float = 1.0,
        shuffle_subjects: bool = True,
        shuffle_slices_within_subject: bool = True,
        seed: Optional[int] = None,
        is_debug: bool = False,
        debug_slice_idx: Optional[int] = None,
        debug_start_y: Optional[int] = 0,
        debug_start_x: Optional[int] = 0,
    ):
        """
        Args:
            subjects: List of subject metadata
            orientations: Which orientations to sample (currently only SAGITTAL)
            slice_fraction: Fraction of slices to sample per subject (0.0 to 1.0)
            shuffle_subjects: Whether to shuffle subject order each epoch
            shuffle_slices_within_subject: Whether to shuffle slice order within each subject
            seed: Random seed for reproducibility
            debug_slice_idx: If set, only sample this specific slice index (debug mode)
        """
        self.subjects = subjects
        self.orientations = orientations
        self.slice_fraction = slice_fraction
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_slices_within_subject = shuffle_slices_within_subject
        self.seed = seed
        self.debug_slice_idx = debug_slice_idx
        self._debug_start_y = debug_start_y
        self._debug_start_x = debug_start_x
        self._epoch = 0
        self._is_debug = is_debug
        self._tissue_bboxes = tissue_bboxes.bounding_boxes

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic shuffling."""
        self._epoch = epoch

    def __call__(self) -> Iterator[SliceSampleSpec]:
        """Generate slice specs, handling worker distribution."""
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Determine subject order
        rng = random.Random(self.seed + self._epoch if self.seed is not None else None)
        subject_list = list(self.subjects)
        if self.shuffle_subjects:
            rng.shuffle(subject_list)

        # Calculate which subjects this worker handles
        total_subjects = len(subject_list)
        subjects_per_worker = math.ceil(total_subjects / num_workers)
        start_subject = worker_id * subjects_per_worker
        end_subject = min(start_subject + subjects_per_worker, total_subjects)

        worker_subjects = subject_list[start_subject:end_subject]

        # Generate specs for this worker's subjects
        for metadata in worker_subjects:
            # Build all (slice_idx, orientation) pairs for this subject
            subject_bboxes = self._tissue_bboxes[metadata.subject_id]
            slices = [i for i, x in enumerate(subject_bboxes) if x is not None]
            total_slices = len(slices)

            if self._is_debug:
                if self.debug_slice_idx is not None:
                    slice_indices = [self.debug_slice_idx]
                else:
                    slice_indices = [slices[int(len(slices)/2)]]
            else:
                # Sample slice indices
                num_slices = int(total_slices * self.slice_fraction)
                slice_indices = list(range(slices[0], slices[-1] + 1))
                
                if self.slice_fraction < 1.0:
                    # Sample a subset
                    slice_indices = rng.sample(slice_indices, num_slices)
            
            # Create all (slice_idx, orientation) pairs
            slice_orientation_pairs = [
                (slice_idx, orientation)
                for slice_idx in slice_indices
                for orientation in self.orientations
            ]
            
            # Shuffle pairs to interleave slices and orientations
            if self.shuffle_slices_within_subject:
                rng.shuffle(slice_orientation_pairs)
            else:
                # Sort by slice_idx first, then orientation
                slice_orientation_pairs.sort(key=lambda x: (x[0], x[1].value))
            
            # Generate specs for each (slice, orientation) pair
            for slice_idx, orientation in slice_orientation_pairs:
                yield SliceSampleSpec(
                    metadata=metadata,
                    slice_idx=slice_idx,
                    orientation=orientation,
                    start_x=self._debug_start_x if self._is_debug else None,
                    start_y=self._debug_start_y if self._is_debug else None,
                )
