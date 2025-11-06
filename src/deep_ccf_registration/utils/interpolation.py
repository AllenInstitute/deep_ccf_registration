import numpy as np
import torch
import torch.nn.functional as F


def interpolate(array: np.ndarray, grid: np.ndarray, mode: str):
    array, grid = _prepare_grid_sample(
        array=array,
        voxels=grid,
    )

    interpolated = F.grid_sample(
        input=array,
        grid=grid,
        mode=mode,
        padding_mode='border',
        align_corners=True
    )
    return interpolated


def _prepare_grid_sample(
    array: np.ndarray,
    voxels: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes the voxels and array and converts them to a format
    suitable for PyTorch's grid_sample function.

    Parameters
    ----------
    array : np.ndarray
        array to interpolate values from
    voxels : np.ndarray
        Voxel coordinates to use for interpolation

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (array, voxels) ready for grid_sample.
    """
    array_shape = array.shape

    # Convert array to torch tensor with shape (1, 3, D, H, W)
    # grid_sample expects (batch, channels, depth, height, width)
    array = torch.from_numpy(array)
    if len(array.shape) == 4:
        array = array.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, D, H, W)
    elif len(array.shape) == 3:
        array = array.permute(0, 1, 2).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    else:
        raise ValueError(f'cannot handle array of shape {array_shape}')

    # Convert voxel indices to normalized coordinates [-1, 1]
    # grid_sample expects coordinates in (x, y, z) order for the last dimension
    array_shape = np.array(array_shape[:3])
    normalized_voxels = 2.0 * voxels / (array_shape - 1) - 1.0

    # grid_sample expects coordinates in (W, H, D) order, but our voxels are in (D, H, W)
    # So we need to reorder: [D, H, W] -> [W, H, D]
    normalized_voxels = normalized_voxels[:, [2, 1, 0]]  # Reorder to (W, H, D)

    # Reshape for grid_sample: (1, N, 1, 1, 3) for 3D sampling
    # where N is the number of points
    n_points = len(voxels)
    normalized_voxels = torch.from_numpy(normalized_voxels)

    if array.dtype == torch.float16:
        normalized_voxels = normalized_voxels.half()
    elif array.dtype == torch.float32:
        normalized_voxels = normalized_voxels.float()

    normalized_voxels = normalized_voxels.reshape(1, n_points, 1, 1, 3)

    return array, normalized_voxels
