import os

import torch
from torch import distributed as dist


def is_main_process() -> bool:
    """Check if this is the main process (rank 0) for logging/mlflow."""
    return int(os.environ.get('RANK', 0)) == 0


def reduce_mean(val: float, device: str) -> float:
    """Average a scalar across all DDP ranks. No-op if not distributed."""
    if not dist.is_initialized():
        return val
    t = torch.tensor(val, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()
