import random

from torch.utils.data import Sampler


class SliceSampler(Sampler):
    """
    This is so that we can call `set_indices` for a given subject batch
    """
    def __init__(self, indices: list[int]):
        super().__init__()
        self._indices = indices

    def set_indices(self, indices: list[int]):
        self._indices = indices

    def __iter__(self):
        # Shuffle each time we iterate
        indices = self._indices.copy()
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self._indices)