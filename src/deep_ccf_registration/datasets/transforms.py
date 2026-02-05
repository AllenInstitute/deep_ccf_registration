import os
from typing import Any

import albumentations
import cv2
import numpy as np
import torch
from loguru import logger
from skimage.exposure import rescale_intensity

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SliceOrientation, AcquisitionAxis


def _restore_grayscale_channel_last(original: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    """Ensure grayscale images keep a trailing channel dim.

    Rule: if the *input* looked grayscale (HxW or HxWx1) but the transform returned HxW,
    restore to HxWx1.

    This intentionally does *not* touch masks.
    """

    is_grayscale_input = original.ndim == 2 or (original.ndim == 3 and original.shape[-1] == 1)
    if is_grayscale_input and transformed.ndim == 2:
        return transformed[..., None]
    return transformed


class Rotate(albumentations.Rotate):
    """
    Subclassing, so that template_coords can be treated as an image
    """
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply
        return targets

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)


class RandomCrop(albumentations.RandomCrop):
    """
    Subclassing, so that template_coords can be treated as an image
    """
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply
        return targets

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)


class Crop(albumentations.Crop):
    """
    Subclassing, so that template_coords can be treated as an image
    """
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply
        return targets

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)

class SquareSymmetry(albumentations.SquareSymmetry):
    """
    Subclassing, so that template_coords can be treated as an image
    """
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply
        return targets

    def apply(self, img: np.ndarray, *args, **params: Any) -> np.ndarray:
        transformed = super().apply(img, *args, **params)
        return _restore_grayscale_channel_last(img, transformed)

class PadIfNeeded(albumentations.PadIfNeeded):
    """
    Subclassing, so that template_coords can be treated as an image
    """
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply
        return targets

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)


class Resample(albumentations.DualTransform):
    """Resample to fixed resolution"""
    def __init__(
            self,
            fixed_resolution: int = 48,
            resize_target: bool = True,
            resize_mask: bool = True

    ):
        super().__init__(p=1.0)
        self._fixed_resolution = fixed_resolution
        self._resize_target = resize_target
        self._resize_mask = resize_mask

    @property
    def targets(self) -> dict[str, Any]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "template_coords": self.apply_to_template_coords,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        return {
            "acquisition_axes": data["acquisition_axes"],
            "slice_axis": data["slice_axis"],
            "orientation": data["orientation"],
        }

    def apply(self, img: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation,  **params: Any) -> np.ndarray:
        transformed = self._resample(x=img, acquisition_axes=acquisition_axes, slice_axis=slice_axis)
        return _restore_grayscale_channel_last(img, transformed)

    def apply_to_mask(self, mask: np.ndarray, *args: Any, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation,  **params: Any) -> np.ndarray:
        if not self._resize_mask:
            return mask
        return self._resample(x=mask, acquisition_axes=acquisition_axes, slice_axis=slice_axis)

    def apply_to_template_coords(self, img: np.ndarray, *args: Any, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation,  **params: Any) -> np.ndarray:
        if not self._resize_target:
            return img
        return self._resample(x=img, acquisition_axes=acquisition_axes, slice_axis=slice_axis)

    def _resample(self, x: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis):
        axes = sorted(acquisition_axes, key=lambda x: x.dimension)
        axes = [x for x in axes if x.dimension != slice_axis.dimension]
        scale_y = axes[0].resolution * 2**3 / self._fixed_resolution
        scale_x = axes[1].resolution * 2**3 / self._fixed_resolution
        resampled = cv2.resize(x, None, fx=scale_x, fy=scale_y)
        return resampled

class LongestMaxSize(albumentations.DualTransform):
    """Resize so the longest side matches ``max_size`` while keeping aspect."""

    def __init__(self, max_size: int, interpolation: int = cv2.INTER_LINEAR, resize_target: bool = True, resize_mask: bool = True):
        super().__init__(p=1.0)
        self.max_size = max_size
        self.interpolation = interpolation
        self.resize_target = resize_target
        self.resize_mask = resize_mask

    @property
    def targets(self) -> dict[str, Any]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "template_coords": self.apply_to_template_coords,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        img = data["image"]
        height, width = img.shape[:2]
        max_dim = max(height, width)
        if max_dim <= self.max_size:
            scale = 1.0
        else:
            scale = self.max_size / max_dim

        new_height = int(round(height * scale)) or 1
        new_width = int(round(width * scale)) or 1
        return {"scale": scale, "height": new_height, "width": new_width}

    def apply(self, img: np.ndarray, scale: float, height: int, width: int, **params: Any) -> np.ndarray:
        if scale == 1.0:
            return img
        resized = cv2.resize(img, (width, height), interpolation=self.interpolation)
        return _restore_grayscale_channel_last(img, resized)

    def apply_to_mask(self, mask: np.ndarray, *args: Any, scale: float, height: int, width: int, **params: Any) -> np.ndarray:
        if not self.resize_mask or scale == 1.0:
            return mask
        return self._resize_array(mask, height=height, width=width)

    def apply_to_template_coords(self, img: np.ndarray, scale: float, height: int, width: int, **params: Any) -> np.ndarray:
        if not self.resize_target or scale == 1.0:
            return img
        return self._resize_array(img, height=height, width=width)

    def _resize_array(self, array: np.ndarray, height: int, width: int) -> np.ndarray:
        if array.ndim != 3:
            raise ValueError("array must be HxWxC")
        channels = [
            cv2.resize(array[..., idx], (width, height), interpolation=self.interpolation)
            for idx in range(array.shape[2])
        ]
        return np.stack(channels, axis=-1)


class TemplatePointsNormalization(albumentations.DualTransform):
    """
    Normalizes template points to [-1,1] (background points will be outside this range)
    """
    def __init__(self, origin: tuple[float, ...], scale: tuple[float, ...], direction: tuple[float, ...], shape: tuple[int, ...]):
        super().__init__(p=1.0)
        self._physical_extent = get_physical_extent(
            origin=origin,
            scale=scale,
            direction=direction,
            shape=shape
        )

    @property
    def targets(self) -> dict[str, Any]:
        return {
            "image": self.apply,
            "template_coords": self.apply_to_template_coords,
        }

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        # passthrough for image, but keep a trailing channel dim for grayscale
        return _restore_grayscale_channel_last(img, img)

    def apply_to_template_coords(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        # normalize template coordinates to [-1, 1]
        extent_min, extent_max = self._physical_extent
        return 2 * (img - extent_min) / (extent_max - extent_min) - 1

    @property
    def physical_extent(self) -> tuple[np.ndarray, np.ndarray]:
        return self._physical_extent

def get_template_point_normalization_inverse(x: np.ndarray | torch.Tensor, template_parameters: TemplateParameters):
    """
    Inverse transform: converts normalized coordinates [-1,1] back to physical coordinates.
    """
    extent_min, extent_max = get_physical_extent(
        origin=template_parameters.origin,
        scale=template_parameters.scale,
        direction=template_parameters.direction,
        shape=template_parameters.shape
    )

    # Convert to same type as input
    if isinstance(x, torch.Tensor):
        extent_min = torch.tensor(extent_min, dtype=x.dtype, device=x.device)
        extent_max = torch.tensor(extent_max, dtype=x.dtype, device=x.device)
        # Reshape for broadcasting with (B, C, H, W) tensors
        if x.dim() == 4:
            extent_min = extent_min.view(1, 3, 1, 1)
            extent_max = extent_max.view(1, 3, 1, 1)

    return ((x + 1.0) / 2.0) * (extent_max - extent_min) + extent_min

class ImageNormalization(albumentations.ImageOnlyTransform):
    def __init__(self):
        super().__init__(p=1.0)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        img = data['image']
        low, high = np.percentile(img, (1, 99))
        return {'low': low, 'high': high}

    def apply(self, img: np.ndarray, low: float, high: float, **params: Any) -> np.ndarray:
        return rescale_intensity(
            img,
            in_range=tuple((low, high)),
            out_range=(0, 1)
        )


def get_physical_extent(origin, scale, direction, shape):
    origin = np.array(origin)
    scale = np.array(scale)
    direction = np.array(direction)
    shape = np.array(shape)

    extent = shape * scale
    extent_min = np.where(direction > 0, origin, origin + direction * extent)
    extent_max = np.where(direction > 0, origin + extent, origin)

    return extent_min, extent_max

def build_transform(
    config: TrainConfig,
    template_parameters: TemplateParameters,
    square_symmetry: bool = False,
    resample_to_fixed_resolution: bool = False,
    rotate_slices: bool = False,
    normalize_template_points: bool = False,
    longest_max_size: bool = False,
    pad_if_needed: bool = True
):
    transforms: list[Any] = [ImageNormalization()]

    if square_symmetry:
        transforms.append(SquareSymmetry())

    if resample_to_fixed_resolution:
        assert config.resample_to_fixed_resolution is not None
        transforms.append(Resample(fixed_resolution=config.resample_to_fixed_resolution))

    if rotate_slices:
        # range obtained from smartSPIM data
        transforms.append(Rotate(limit=(-20, 20), border_mode=cv2.BORDER_REPLICATE))

    if normalize_template_points:
        transforms.append(TemplatePointsNormalization(
            origin=template_parameters.origin,
            scale=template_parameters.scale,
            direction=template_parameters.direction,
            shape=template_parameters.shape
        ))


    if config.patch_size is not None:
        if config.debug and (config.debug_start_y is not None and config.debug_start_x is not None):
            transforms.append(Crop(
                y_min=config.debug_start_y,
                x_min=config.debug_start_x,
                y_max=config.debug_start_y+config.patch_size[0],
                x_max=config.debug_start_x+config.patch_size[1],
                pad_if_needed=True,
                pad_position='top_left'
            ))
        else:
            transforms.append(RandomCrop(
                height=config.patch_size[0],
                width=config.patch_size[1],
                pad_if_needed=True,
                pad_position='top_left'
            ))

    if longest_max_size:
        assert config.longest_max_size is not None
        transforms.append(LongestMaxSize(max_size=config.longest_max_size))

    if pad_if_needed:
        transforms.append(PadIfNeeded(min_height=config.pad_dim, min_width=config.pad_dim, position='top_left'))

    if len(transforms) > 0:
        return albumentations.ReplayCompose(transforms, seed=config.seed)
    return None

def apply_crop_pad_to_original(
    template_coords: np.ndarray,
    replay: dict,
    original_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
    """
    Apply crop and pad transforms to original-resolution coordinates.

    Scales crop coordinates from resized space to original space, then applies
    crop and pad. This allows evaluation at original resolution without
    interpolating the target coordinates.

    Args:
        template_coords: Original template coordinates (H_orig, W_orig, 3)
        replay: Transform replay dict from albumentations
        original_shape: (H_orig, W_orig) original image shape
        resized_shape: (H_resized, W_resized) shape after resize transforms
        mask: Optional mask array (H_orig, W_orig)

    Returns:
        Tuple of (transformed_coords, transformed_mask, eval_shape)
        where eval_shape is the spatial dimensions of the eval targets
    """
    orig_h, orig_w = original_shape
    resized_h, resized_w = resized_shape

    # Compute scale factors
    scale_h = orig_h / resized_h if resized_h > 0 else 1.0
    scale_w = orig_w / resized_w if resized_w > 0 else 1.0

    result_coords = template_coords.copy()
    result_mask = mask.copy() if mask is not None else None

    # Process transforms in order
    for t in replay.get("transforms", []):
        t_name = t.get("__class_fullname__", "")
        params = t.get("params", {}) or {}

        if "RandomCrop" in t_name:
            # Scale crop coordinates to original resolution
            h_start = int(params.get("h_start", 0) * scale_h)
            w_start = int(params.get("w_start", 0) * scale_w)
            # Get crop size from the transform (height/width attributes)
            crop_h = int(params.get("shape", (0, 0))[0] * scale_h) if "shape" in params else int(100 * scale_h)
            crop_w = int(params.get("shape", (0, 0))[1] * scale_w) if "shape" in params else int(100 * scale_w)

            # Actually, RandomCrop params have crop_coords: (y_min, x_min, y_max, x_max)
            crop_coords = params.get("crop_coords")
            if crop_coords:
                y_min, x_min, y_max, x_max = crop_coords
                h_start = int(y_min * scale_h)
                w_start = int(x_min * scale_w)
                crop_h = int((y_max - y_min) * scale_h)
                crop_w = int((x_max - x_min) * scale_w)

            # Clamp to valid range
            h_start = max(0, min(h_start, result_coords.shape[0] - 1))
            w_start = max(0, min(w_start, result_coords.shape[1] - 1))
            h_end = min(h_start + crop_h, result_coords.shape[0])
            w_end = min(w_start + crop_w, result_coords.shape[1])

            result_coords = result_coords[h_start:h_end, w_start:w_end]
            if result_mask is not None:
                result_mask = result_mask[h_start:h_end, w_start:w_end]

        elif "Crop" in t_name and "RandomCrop" not in t_name:
            # Fixed crop - scale coordinates
            y_min = int(t.get("y_min", params.get("y_min", 0)) * scale_h)
            x_min = int(t.get("x_min", params.get("x_min", 0)) * scale_w)
            y_max = int(t.get("y_max", params.get("y_max", result_coords.shape[0])) * scale_h)
            x_max = int(t.get("x_max", params.get("x_max", result_coords.shape[1])) * scale_w)

            result_coords = result_coords[y_min:y_max, x_min:x_max]
            if result_mask is not None:
                result_mask = result_mask[y_min:y_max, x_min:x_max]

        elif "PadIfNeeded" in t_name:
            # Scale pad amounts
            pad_top = int(params.get("pad_top", 0) * scale_h)
            pad_bottom = int(params.get("pad_bottom", 0) * scale_h)
            pad_left = int(params.get("pad_left", 0) * scale_w)
            pad_right = int(params.get("pad_right", 0) * scale_w)

            # Pad coordinates - use 0 for padding (will be masked out)
            new_h = result_coords.shape[0] + pad_top + pad_bottom
            new_w = result_coords.shape[1] + pad_left + pad_right

            padded_coords = np.zeros((new_h, new_w, 3), dtype=result_coords.dtype)
            padded_coords[pad_top:pad_top + result_coords.shape[0],
                         pad_left:pad_left + result_coords.shape[1]] = result_coords
            result_coords = padded_coords

            if result_mask is not None:
                padded_mask = np.zeros((new_h, new_w), dtype=result_mask.dtype)
                padded_mask[pad_top:pad_top + result_mask.shape[0],
                           pad_left:pad_left + result_mask.shape[1]] = result_mask
                result_mask = padded_mask

    eval_shape = (result_coords.shape[0], result_coords.shape[1])
    return result_coords, result_mask, eval_shape


def physical_to_index_space(physical_pts: torch.Tensor | np.ndarray, template_parameters: TemplateParameters):
    points = physical_pts.clone() if isinstance(physical_pts, torch.Tensor) else physical_pts.copy()
    for dim in range(template_parameters.dims):
        points[:, :, dim] -= template_parameters.origin[dim]
        points[:, :, dim] *= template_parameters.direction[dim]
        points[:, :, dim] /= template_parameters.scale[dim]
    return points

def map_points_to_right_hemisphere(
        template_points: np.ndarray,
        template_parameters: TemplateParameters,
):
    """
    Because which hemisphere a slice is in is ambiguous, this maps the ground truth template points
    to the right hemisphere.

    :param template_points:
    :param template_parameters:
    :return:
    """
    template_points_index_space = physical_to_index_space(
        physical_pts=template_points,
        template_parameters=template_parameters,
    )

    # checks whether the ML points are < halfway in index space. the LS template iS RAS.
    # therefore this checks whether points are in left hemisphere
    need_mirror = (template_points_index_space[:, :, 0] < template_parameters.shape[0] / 2).all()
    if os.environ.get('LOG_LEVEL') == 'DEBUG':
        logger.debug(
            f"ML index mean: {template_points[:, :, 0].mean():.1f}, "
            f"ML index range: {template_points[:, :, 0].min():.1f} - {template_points[:, :, 0].max():.1f} "
            f"midpoint: {template_parameters.shape[0] / 2:.1f}, "
            f"need_mirror: {need_mirror}")
    if need_mirror:
        # map to right hemisphere
        template_points = mirror_points(points=template_points, template_parameters=template_parameters)
    return template_points

def mirror_points(points: torch.Tensor | np.ndarray, template_parameters: TemplateParameters):
    points = points.clone() if isinstance(points, torch.Tensor) else points.copy()

    # 1. Convert to index space
    points = physical_to_index_space(physical_pts=points, template_parameters=template_parameters)

    # 2. Flip ML in index space
    points[:, :, 0] = template_parameters.shape[0]-1 - points[:, :, 0]

    # 3. Convert back to physical
    for dim in range(template_parameters.dims):
        points[:, :, dim] *= template_parameters.scale[dim]
        points[:, :, dim] *= template_parameters.direction[dim]
        points[:, :, dim] += template_parameters.origin[dim]

    return points