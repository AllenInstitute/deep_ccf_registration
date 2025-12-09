import atexit
import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import tensorstore
from loguru import logger


class MemmapCache:
    def __init__(self):
        self._paths: dict[int, tuple[Path, Path]] = {}

    def set(self, idx: int, vol_path: Path, warp_path: Path):
        self._paths[idx] = (vol_path, warp_path)

    def get(self, idx: int) -> Optional[tuple[Path, Path]]:
        return self._paths.get(idx)

    def clear(self):
        self._paths.clear()


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
        memmap_cache: MemmapCache,
        n_subjects_per_batch: int,
        maxsize: int = 1,
        memmap_dir: Path = Path('/results'),
        clean_on_exit: bool = True
    ):
        self._maxsize = maxsize
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.volumes = volumes
        self.warps = warps
        self.n_subjects_per_batch = n_subjects_per_batch
        self.subject_idx_batches = self._get_subject_batches(subject_metadata, n_subjects_per_batch)
        self.thread: Optional[threading.Thread] = None
        self.exception: Optional[Exception] = None
        os.makedirs(memmap_dir, exist_ok=True)
        self._memmap_dir = memmap_dir
        self.memmap_cache = memmap_cache
        self._subject_metadata = subject_metadata
        self._register_cleanup()
        self._clean_on_exit = clean_on_exit

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
        volume = self.volumes[idx][:].read().result()
        warp = self.warps[idx][:].read().result()
        return volume, warp

    def _write_single_memmap(self, idx: int, volume: np.ndarray, warp: np.ndarray) -> tuple[Path, Path]:
        """Write a single volume and warp array to memmap files (if not already on disk)."""
        vol_path = self._memmap_dir / f'vol_{idx}.dat'
        if not vol_path.exists():
            logger.debug(f'writing volume memmap to {vol_path}')
            vol_mmap = np.memmap(vol_path, dtype=volume.dtype, mode='w+', shape=volume.shape)
            vol_mmap[:] = volume
            vol_mmap.flush()

        warp_path = self._memmap_dir / f'warp_{idx}.dat'
        if not warp_path.exists():
            logger.debug(f'writing warp memmap to {vol_path}')
            warp_mmap = np.memmap(warp_path, dtype=warp.dtype, mode='w+', shape=warp.shape)
            warp_mmap[:] = warp
            warp_mmap.flush()

        self.memmap_cache.set(idx, vol_path, warp_path)
        return vol_path, warp_path

    def _register_cleanup(self):
        atexit.register(self.cleanup)

    def cleanup(self):
        logger.info(f'Cleaning up {self._memmap_dir}')
        shutil.rmtree(self._memmap_dir)
        os.makedirs(self._memmap_dir, exist_ok=True)
        self.memmap_cache.clear()

    def _producer(self):
        """Producer function that loads batches and writes memmaps in the background."""
        try:
            for subject_idx_batch in self.subject_idx_batches:
                logger.info(f'Loading subject data {subject_idx_batch}')
                for idx in subject_idx_batch:
                    if self.memmap_cache.get(idx) is None:
                        volume, warp = self._load_arrays(idx)
                        self._write_single_memmap(idx=idx, volume=volume, warp=warp)
                    else:
                        logger.debug(f'Memmap already exists for subject {idx}')
                logger.debug(f'Enqueuing subjects {subject_idx_batch}')
                self.queue.put(subject_idx_batch)
            self.queue.put(None)
        except Exception as e:
            self.exception = e
            self.queue.put(None)

    def _reset_iteration_state(self):
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.queue = queue.Queue(maxsize=self._maxsize)
        self.subject_idx_batches = self._get_subject_batches(self._subject_metadata, self.n_subjects_per_batch)
        self.exception = None

    def _start_producer(self):
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def iter_batches(self) -> Iterator[list[int]]:
        self._reset_iteration_state()
        self._start_producer()
        while True:
            item = self.queue.get()
            if item is None:
                if self.exception:
                    raise self.exception
                break
            logger.debug(f'Loaded {item} from queue')
            yield item
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def __iter__(self) -> Iterator[list[int]]:
        return self.iter_batches()

    def start(self):
        self._reset_iteration_state()
        self._start_producer()
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        if self._clean_on_exit:
            self.cleanup()