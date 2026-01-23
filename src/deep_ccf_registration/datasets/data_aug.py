from dataclasses import dataclass

import numpy as np

from deep_ccf_registration.metadata import TissueBoundingBox


@dataclass
class ObliqueSliceParams:
    x_rot: float
    y_rot: float
    z_rot: float


class ObliqueSliceSampler:
    def __init__(
            self,
            slice_range: range,
            vol_shape: tuple[int, int, int],
            slice_axis: int,
            crop_size: tuple[int, int],
            bbox: TissueBoundingBox,
            x_range: tuple[float, float] = (0, 0),
            y_range: tuple[float, float] = (0, 0),
            z_range: tuple[float, float] = (0, 0),
    ):
        self._slice_range = slice_range
        self._vol_shape = vol_shape
        self._slice_axis = slice_axis
        self._crop_size = crop_size
        self._bbox = bbox

        self._params = ObliqueSliceParams(
            x_rot=np.random.uniform(*x_range),
            y_rot=np.random.uniform(*y_range),
            z_rot=np.random.uniform(*z_range),
        )

        self._chunk_slices = self._compute_chunk_bounds()

    @property
    def bbox(self) -> TissueBoundingBox:
        return self._bbox

    @property
    def chunk_slices(self) -> tuple[slice, slice, slice]:
        return self._chunk_slices

    @property
    def slice_range(self) -> range:
        return self._slice_range

    @property
    def params(self) -> ObliqueSliceParams:
        return self._params

    def _compute_chunk_bounds(self) -> tuple[slice, slice, slice]:
        """Compute minimal chunk bounds to contain all rotated samples within bbox."""
        in_plane_axes = [i for i in range(3) if i != self._slice_axis]

        bbox_h = self._bbox.height
        bbox_w = self._bbox.width

        theta_z = np.radians(abs(self._params.z_rot))
        crop_h, crop_w = self._crop_size
        half_extent_0 = (crop_h * np.cos(theta_z) + crop_w * np.sin(theta_z)) / 2
        half_extent_1 = (crop_w * np.cos(theta_z) + crop_h * np.sin(theta_z)) / 2

        # Only add the EXTRA extent due to rotation (beyond original crop size)
        margin_0 = max(0, half_extent_0 - crop_h / 2)
        margin_1 = max(0, half_extent_1 - crop_w / 2)

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
            max(0, int(np.floor(self._bbox.y - margin_0))),
            min(self._vol_shape[in_plane_axes[0]],
                int(np.ceil(self._bbox.y + bbox_h + margin_0)) + 1)
        )

        slices[in_plane_axes[1]] = slice(
            max(0, int(np.floor(self._bbox.x - margin_1))),
            min(self._vol_shape[in_plane_axes[1]],
                int(np.ceil(self._bbox.x + bbox_w + margin_1)) + 1)
        )

        return tuple(slices)

    def extract_oblique_coords(self, slice_idx: int, start_yx: tuple[int, int]):
        """
        Extract coordinates for a single oblique slice.

        Args:
            slice_idx: Slice index in VOLUME coordinates
            start_yx: (y, x) top-left corner of crop

        Returns:
            List of coordinate arrays [coords_0, coords_1, coords_2] in VOLUME coordinates.
        """
        crop_h, crop_w = self._crop_size
        in_plane_axes = [i for i in range(3) if i != self._slice_axis]

        crop_center_yx = (
            start_yx[0] + crop_h / 2,
            start_yx[1] + crop_w / 2,
        )

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
        coords[in_plane_axes[0]] = crop_center_yx[0] + rotated_0
        coords[in_plane_axes[1]] = crop_center_yx[1] + rotated_1

        return coords
