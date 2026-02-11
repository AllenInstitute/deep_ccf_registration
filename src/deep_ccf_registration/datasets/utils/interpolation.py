from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorstore
from loguru import logger
from scipy.ndimage import map_coordinates

from deep_ccf_registration.utils.logging_utils import timed


def map_coordinates_cropped(
    volume: tensorstore.TensorStore,
    coords: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Interpolate with map_coordinates on a cropped subvolume to reduce IO.

    Args:
        volume: 3D volume to sample.
        coords: array of shape (3, n_points) in volume index space.
        order: interpolation order (use 1 for linear).
        mode: out-of-bounds handling.
        cval: constant value for mode='constant'.
    """
    mins = np.floor(coords.min(axis=1)).astype(int)
    maxs = np.ceil(coords.max(axis=1)).astype(int)

    mins = np.maximum(mins, 0)

    if len(volume.shape) == 4:
        maxs = np.minimum(maxs, np.array(volume.shape[1:]) - 1)
    else:
        maxs = np.minimum(maxs, np.array(volume.shape) - 1)

    z0, y0, x0 = mins
    z1, y1, x1 = maxs + 1  # slice end is exclusive

    with timed():
        logger.debug(f'reading warp with bbox {slice(z0,z1), slice(y0,y1), slice(x0,x1)}')
        if len(volume.shape) == 4:
            subvol = volume[:, z0:z1, y0:y1, x0:x1].read().result()
        else:
            subvol = volume[z0:z1, y0:y1, x0:x1].read().result()

    coords_local = coords.copy()
    coords_local[0] -= z0
    coords_local[1] -= y0
    coords_local[2] -= x0

    if len(subvol.shape) == 4:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    map_coordinates,
                    subvol[i],
                    coords,
                    order=order,
                    mode=mode,
                    cval=cval,
                )
                for i in range(3)
            ]
        res = np.stack([f.result() for f in futures], axis=-1)
    else:
        res = map_coordinates(
        input=subvol,
        coordinates=coords_local,
        order=order,
        mode=mode,
        cval=cval,
    )
    return res
