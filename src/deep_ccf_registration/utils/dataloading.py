import queue
import threading
from typing import Iterator, Tuple, Optional, Any

import numpy as np

from deep_ccf_registration.datasets.slice_dataset import SliceDataset


class BatchPrefetcher:
    """
    Prefetches subject batches in a background thread to avoid blocking the main training loop.

    This implements a producer-consumer pattern where:
    - Producer thread: loads batches by calling dataset.get_arrays() in the background
    - Consumer (main loop): gets already-loaded batches from the queue

    While training on batch N, batch N+1 is being loaded in parallel.

    Parameters
    ----------
    dataset : SliceDataset
        The dataset to load batches from
    subject_idx_batches : list
        List of subject index batches to load
    maxsize : int, optional
        Maximum number of batches to prefetch (default=1)
    """

    def __init__(self, dataset: SliceDataset, subject_idx_batches: list[list[int]], maxsize: int = 1):
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dataset = dataset
        self.subject_idx_batches = subject_idx_batches
        self.thread: Optional[threading.Thread] = None
        self.exception: Optional[Exception] = None

    def _producer(self):
        """Producer function that loads batches in the background."""
        try:
            for subject_idx_batch in self.subject_idx_batches:
                batch_volumes, batch_warps = self.dataset.get_arrays(idxs=subject_idx_batch)
                self.queue.put((subject_idx_batch, batch_volumes, batch_warps))
            # Put sentinel value to signal completion
            self.queue.put(None)
        except Exception as e:
            self.exception = e
            # Put sentinel to unblock consumer
            self.queue.put(None)

    def start(self):
        """Start the background prefetching thread."""
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def __iter__(self) -> Iterator[Tuple[list[int], list[np.ndarray], list[np.ndarray]]]:
        """
        Iterate over prefetched batches.

        Yields
        ------
        tuple
            (subject_idx_batch, batch_volumes, batch_warps)
        """
        while True:
            item = self.queue.get()
            if item is None:
                # Check if we stopped due to an exception
                if self.exception:
                    raise self.exception
                break
            yield item

    def __enter__(self):
        """Context manager entry - starts prefetching."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - waits for thread to finish."""
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)