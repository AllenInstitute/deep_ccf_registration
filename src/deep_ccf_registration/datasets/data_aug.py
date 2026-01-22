from dataclasses import dataclass

import numpy as np


@dataclass
class ObliqueSliceParams:
    x_rot: float
    y_rot: float
    z_rot: float


class ObliqueSliceSampler:
    """
    Sample oblique slices from the volume.
    """
    def __init__(
        self,
        slice_range: range,
        vol_shape: tuple[int, int, int],
        slice_axis: int,
        crop_size: tuple[int, int],
        crop_center_yx: tuple[int, int],
        x_range: tuple[float, float] = (0, 0),
        y_range: tuple[float, float] = (0, 0),
        z_range: tuple[float, float] = (0, 0),
    ):
        """
        Args:
            slice_range: Range of slice indices to extract
            vol_shape: (dim0, dim1, dim2) shape of volume
            slice_axis: Which axis is perpendicular to slices
            crop_size: (height, width) of each output crop
            crop_center_yx: (in_plane_0, in_plane_1) center of crop in volume coords
            x_range, y_range, z_range: Rotation ranges in degrees
        """
        self._slice_range = slice_range
        self._vol_shape = vol_shape
        self._slice_axis = slice_axis
        self._crop_size = crop_size
        self._crop_center_yx = crop_center_yx

        # Sample params
        self._params = ObliqueSliceParams(
            x_rot=np.random.uniform(*x_range),
            y_rot=np.random.uniform(*y_range),
            z_rot=np.random.uniform(*z_range),
        )

        # Compute chunk bounds
        self._chunk_slices = self._compute_chunk_bounds()

    @property
    def slice_range(self) -> range:
        return self._slice_range

    @property
    def chunk_slices(self) -> tuple[slice, slice, slice]:
        return self._chunk_slices

    @property
    def params(self) -> ObliqueSliceParams:
        return self._params

    @property
    def crop_size(self) -> tuple[int, int]:
        return self._crop_size

    @property
    def crop_center_yx(self) -> tuple[int, int]:
        return self._crop_center_yx

    def _compute_chunk_bounds(self) -> tuple[slice, slice, slice]:
        """Compute minimal chunk bounds to contain all rotated samples."""
        in_plane_axes = [i for i in range(3) if i != self._slice_axis]
        crop_h, crop_w = self._crop_size

        # In-plane extent due to z-rotation
        theta_z = np.radians(abs(self._params.z_rot))
        half_extent_0 = (crop_h * np.cos(theta_z) + crop_w * np.sin(theta_z)) / 2
        half_extent_1 = (crop_w * np.cos(theta_z) + crop_h * np.sin(theta_z)) / 2

        # Slice axis offset due to x/y rotation
        tan_x = np.tan(np.radians(self._params.x_rot))
        tan_y = np.tan(np.radians(self._params.y_rot))
        max_slice_offset = half_extent_0 * abs(tan_y) + half_extent_1 * abs(tan_x)

        min_slice = min(self._slice_range)
        max_slice = max(self._slice_range)

        slices = [None, None, None]

        slices[self._slice_axis] = slice(
            max(0, int(np.floor(min_slice - max_slice_offset))),
            min(self._vol_shape[self._slice_axis], int(np.ceil(max_slice + max_slice_offset)) + 1)
        )

        slices[in_plane_axes[0]] = slice(
            max(0, int(np.floor(self._crop_center_yx[0] - half_extent_0))),
            min(self._vol_shape[in_plane_axes[0]], int(np.ceil(self._crop_center_yx[0] + half_extent_0)) + 1)
        )

        slices[in_plane_axes[1]] = slice(
            max(0, int(np.floor(self._crop_center_yx[1] - half_extent_1))),
            min(self._vol_shape[in_plane_axes[1]], int(np.ceil(self._crop_center_yx[1] + half_extent_1)) + 1)
        )

        return tuple(slices)

    def extract_oblique_coords(self, slice_idx: int):
        """
        Extract coordinates for a single oblique slice.

        Args:
            slice_idx: Slice index in VOLUME coordinates

        Returns:
            List of coordinate arrays [coords_0, coords_1, coords_2] in VOLUME coordinates.
        """
        crop_h, crop_w = self._crop_size
        in_plane_axes = [i for i in range(3) if i != self._slice_axis]

        # Build output grid centered at origin
        grid_0 = np.arange(crop_h) - (crop_h - 1) / 2.0
        grid_1 = np.arange(crop_w) - (crop_w - 1) / 2.0
        centered_0, centered_1 = np.meshgrid(grid_0, grid_1, indexing='ij')

        # Apply z-rotation
        theta_z = np.radians(-self._params.z_rot)
        cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)

        rotated_0 = cos_z * centered_0 - sin_z * centered_1
        rotated_1 = sin_z * centered_0 + cos_z * centered_1

        # Compute slice offset from x/y rotation
        slice_offset = rotated_0 * np.tan(np.radians(self._params.y_rot)) \
                       + rotated_1 * np.tan(np.radians(self._params.x_rot))

        # Map to volume coordinates
        coords = [None, None, None]
        coords[self._slice_axis] = slice_idx + slice_offset
        coords[in_plane_axes[0]] = self._crop_center_yx[0] + rotated_0
        coords[in_plane_axes[1]] = self._crop_center_yx[1] + rotated_1

        return coords