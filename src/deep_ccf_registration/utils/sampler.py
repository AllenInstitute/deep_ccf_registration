import random

from torch.utils.data import Sampler

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.utils.dataloading import BatchPrefetcher


class SliceSampler(Sampler):
    def __init__(
            self,
            dataset: SliceDataset,
            subject_batch_iter: BatchPrefetcher,
            max_iters_per_subject_batch: int,
            batch_size: int,
            is_debug: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.subject_batch_iter = subject_batch_iter
        self.max_samples_per_subject_batch = max_iters_per_subject_batch * batch_size
        self.current_subject_batch_idx = 0
        self.current_subject_batch = None
        self._is_debug = is_debug

    def __iter__(self):
        for subject_batch_idx, subject_idx_batch in enumerate(self.subject_batch_iter):
            self.current_subject_batch_idx = subject_batch_idx
            self.current_subject_batch = subject_idx_batch

            if self._is_debug:
                indices = [1000]
            else:
                indices = self.dataset.get_subject_sample_idxs(subject_idx_batch)
            random.shuffle(indices)

            # Yield up to max samples from this subject batch
            for i, idx in enumerate(indices):
                if i >= self.max_samples_per_subject_batch:
                    break
                yield idx

    def __len__(self):
        # Approximate
        return len(self.dataset)