import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import tensorstore
from loguru import logger

from deep_ccf_registration.utils.logging_utils import timed


class BatchPrefetcher:
    """
    Prefetches subject batches in a background thread to avoid blocking the main training loop.

    This implements a producer-consumer pattern where:
    - Producer thread: loads arrays in the background and writes them to memmap files on disk
    - Consumer (main loop): gets already-written batches from the queue

    Parameters
    ----------
    volumes : list
        List of volume tensorstore arrays
    warps : list
        List of warp tensorstore arrays
    subject_metadata : list
        List of subject metadata
    n_subjects_per_batch : int
        Number of subjects per batch
    maxsize : int, optional
        Maximum number of batches to prefetch (default=1)
    memmap_dir : Path, optional
        Directory to store memmap files (default='/results')
    """

    def __init__(
        self,
        volumes: list[tensorstore.TensorStore],
        warps: list[tensorstore.TensorStore],
        subject_metadata: list,
        n_subjects_per_batch: int,
        maxsize: int = 1,
        memmap_dir: Path = Path('/results'),
    ):
        self._maxsize = maxsize
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.volumes = volumes
        self.warps = warps
        self.n_subjects_per_batch = n_subjects_per_batch
        self.subject_idx_batches = self._get_subject_batches(subject_metadata, n_subjects_per_batch)
        self.thread: Optional[threading.Thread] = None
        os.makedirs(memmap_dir, exist_ok=True)
        self._memmap_dir = memmap_dir
        self._subject_metadata = subject_metadata

    @staticmethod
    def _get_subject_batches(subject_metadata: list, n_subjects_per_batch: int) -> list[list[int]]:
        """
        Create batches of subject indices.
        
        Parameters
        ----------
        subject_metadata : list
            List of subject metadata
        n_subjects_per_batch : int
            Number of subjects per batch
            
        Returns
        -------
        list[list[int]]
            List of batches, each containing subject indices
        """
        subject_idxs = np.arange(len(subject_metadata))
        np.random.shuffle(subject_idxs)
        subject_idxs = subject_idxs.tolist()
        subject_idx_batches = [
            subject_idxs[i:i + n_subjects_per_batch] 
            for i in range(0, len(subject_idxs), n_subjects_per_batch)
        ]
        return subject_idx_batches

    def _load_arrays(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load volume/warp into RAM (only once) before writing memmaps."""
        logger.debug(f'Loading tensors for subject {idx}')

        with timed():
            volume = self.volumes[idx][:].read().result()
        with timed():
            warp = self.warps[idx][:].read().result()
        return volume, warp

    def _write_single_memmap(self, idx: int, volume: np.ndarray, warp: np.ndarray) -> tuple[Path, Path]:
        """Write a single volume and warp array to memmap files (if not already on disk)."""
        vol_path = self._memmap_dir / f'vol_{idx}.dat'
        if not vol_path.exists():
            logger.debug(f'writing volume memmap to {vol_path}')
            vol_mmap = np.memmap(vol_path, dtype=volume.dtype, mode='w+', shape=volume.shape)
            with timed():
                vol_mmap[:] = volume
            vol_mmap.flush()

        warp_path = self._memmap_dir / f'warp_{idx}.dat'
        if not warp_path.exists():
            logger.debug(f'writing warp memmap to {vol_path}')
            warp_mmap = np.memmap(warp_path, dtype=warp.dtype, mode='w+', shape=warp.shape)
            with timed():
                warp_mmap[:] = warp
            warp_mmap.flush()

        return vol_path, warp_path

    def cleanup(self):
        logger.info(f'Cleaning up {self._memmap_dir}')
        shutil.rmtree(self._memmap_dir)
        os.makedirs(self._memmap_dir, exist_ok=True)

    def _producer(self):
        """Producer function that loads batches and writes memmaps in the background."""
        for i, subject_idx_batch in enumerate(self.subject_idx_batches):
            self.cache_data(subject_idx_batch=subject_idx_batch)
            logger.info(f'Batch {i} has been cached')
            self.queue.put(subject_idx_batch)

    def cache_data(self, subject_idx_batch: list[int]):
        logger.debug(f'caching {subject_idx_batch}')
        for i, idx in enumerate(subject_idx_batch):
            vol_path = self._memmap_dir / f'vol_{idx}.dat'
            warp_path = self._memmap_dir / f'warp_{idx}.dat'
            if not (vol_path.exists() and warp_path.exists()):
                logger.info(f'Caching {i}/{len(subject_idx_batch)}')
                volume, warp = self._load_arrays(idx)
                self._write_single_memmap(idx=idx, volume=volume, warp=warp)
            else:
                logger.debug(f'{idx} already cached')

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def stop(self):
        self.cleanup()
        if self.thread:
            self.thread.join(timeout=5.0)

    def reset(self):
        """Reset the queue/thread so the same subject batches can be iterated again."""
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.queue = queue.Queue(maxsize=self._maxsize)
        self.thread = None
        self.start()

    def __iter__(self) -> Iterator[list[int]]:
        if not self.thread or not self.thread.is_alive():
            self.start()
        for _ in range(len(self.subject_idx_batches)):
            yield self.queue.get()
