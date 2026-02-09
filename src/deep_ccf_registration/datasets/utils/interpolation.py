import time

import numpy as np
from loguru import logger
from scipy.ndimage import map_coordinates


def map_coordinates_cropped(
    volume: np.ndarray,
    coords: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    pad: int = 1,
) -> np.ndarray:
    """Interpolate with map_coordinates on a cropped subvolume to reduce IO.

    Args:
        volume: 3D volume to sample.
        coords: array of shape (3, n_points) in volume index space.
        order: interpolation order (use 1 for linear).
        mode: out-of-bounds handling.
        cval: constant value for mode='constant'.
        pad: extra voxel padding around min/max coords.
    """
    mins = np.floor(coords.min(axis=1) - pad).astype(int)
    maxs = np.ceil(coords.max(axis=1) + pad).astype(int)

    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, np.array(volume.shape) - 1)

    z0, y0, x0 = mins
    z1, y1, x1 = maxs + 1  # slice end is exclusive

    # Force the cropped subvolume into contiguous RAM.  When ``volume``
    # is memory-mapped, the slice creates a *view* — ``map_coordinates``
    # then triggers scattered page faults across the mmap.  Reading the
    # subvolume into a contiguous array first converts those scattered
    # faults into a single sequential read (fast on local disk).
    t0 = time.perf_counter()
    subvol = np.ascontiguousarray(volume[z0:z1, y0:y1, x0:x1])
    t1 = time.perf_counter()

    coords_local = coords.copy()
    coords_local[0] -= z0
    coords_local[1] -= y0
    coords_local[2] -= x0

    t2 = time.perf_counter()
    result = map_coordinates(
        input=subvol,
        coordinates=coords_local,
        order=order,
        mode=mode,
        cval=cval,
    )
    t3 = time.perf_counter()

    subvol_shape = (z1 - z0, y1 - y0, x1 - x0)
    subvol_mb = subvol.nbytes / (1024 * 1024)
    is_mmap = hasattr(volume, '_mmap') or not volume.flags['OWNDATA']
    logger.debug(
        f"map_coordinates_cropped: subvol={subvol_shape} ({subvol_mb:.1f}MB) "
        f"mmap={is_mmap} copy={t1-t0:.4f}s interp={t3-t2:.4f}s "
        f"n_points={coords.shape[1]} vol_shape={volume.shape}"
    )

    return result
