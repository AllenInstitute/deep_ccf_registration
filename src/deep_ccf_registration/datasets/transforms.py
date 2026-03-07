import os
from typing import Any

import albumentations
import cv2
import numpy as np
import torch
from albumentations import Normalize, GaussNoise, CoarseDropout, RandomBrightnessContrast
from loguru import logger
from skimage.exposure import rescale_intensity

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SliceOrientation, AcquisitionAxis, SubjectRotationAngle, \
    RotationAngles


def _restore_grayscale_channel_last(original: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    """Ensure grayscale images keep a trailing channel dim.
    """

    is_grayscale_input = original.ndim == 2 or (original.ndim == 3 and original.shape[-1] == 1)
    if is_grayscale_input and transformed.ndim == 2:
        return transformed[..., None]
    return transformed

class Rotate(albumentations.Rotate):

    def __init__(
            self,
            rotation_angles: RotationAngles,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._rotation_angles = rotation_angles

    @property
    def targets(self) -> dict[str, Any]:
        return {
            "image": self.apply,
            "mask": self.apply,
            "template_coords": self.apply_to_template_coords,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        # Extract custom data from params
        orientation: SliceOrientation = data["orientation"]

        # Determine which axis is the in-plane rotation axis based on slice orientation
        if orientation == SliceOrientation.SAGITTAL:
            valid_range = self._rotation_angles.ML_range
        else:
            raise ValueError(f"Unknown orientation: {orientation}")


        # Sample rotation angle within valid range
        angle = float(np.random.uniform(valid_range[0], valid_range[1]))

        logger.debug(f'z_rot={angle:.3f}')

        # Compute symmetric padding needed for angle so rotation won't crop.
        img = data["image"]
        height, width = img.shape[:2]

        theta = np.deg2rad(abs(angle))
        cos_t = abs(float(np.cos(theta)))
        sin_t = abs(float(np.sin(theta)))

        new_w = width * cos_t + height * sin_t
        new_h = width * sin_t + height * cos_t

        pad_x = int(np.ceil(max(0.0, (new_w - width) / 2.0)))
        pad_y = int(np.ceil(max(0.0, (new_h - height) / 2.0)))

        pad_top = pad_y
        pad_bottom = pad_y
        pad_left = pad_x
        pad_right = pad_x

        # Use padded shape so parent computes rotation matrix with correct center
        padded_h = height + pad_top + pad_bottom
        padded_w = width + pad_left + pad_right

        params_copy = params.copy()
        params_copy["shape"] = (padded_h, padded_w) + params_copy["shape"][2:]

        # Force parent to use our sampled angle (it ignores params["angle"]
        # and instead samples from self.limit)
        original_limit = self.limit
        self.limit = (angle, angle)
        try:
            rotate_params = super().get_params_dependent_on_data(params=params_copy, data=data)
        finally:
            self.limit = original_limit
        rotate_params.update(
            {
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
            }
        )
        return rotate_params

    def apply(
            self,
            img: np.ndarray,
            pad_top: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
            pad_right: int = 0,
            border_mode: str = 'constant',
            **params: Any
    ) -> np.ndarray:
        if pad_top or pad_bottom or pad_left or pad_right:
            pad_width = (
                ((pad_top, pad_bottom), (pad_left, pad_right))
                if img.ndim == 2
                else ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            )

            if border_mode == "constant":
                padded = np.pad(img, pad_width=pad_width, mode=border_mode, constant_values=0)
            else:
                padded = np.pad(img, pad_width=pad_width, mode=border_mode)
            img = _restore_grayscale_channel_last(original=img, transformed=padded)
        params['shape'] = img.shape
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)

    def apply_to_template_coords(self, *args, **kwargs) -> np.ndarray:
        return self.apply(
            *args,
            **kwargs,
            border_mode='edge',
        )

class GridDistortion(albumentations.GridDistortion):
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply
        return targets

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        try:
            return super().get_params_dependent_on_data(params, data)
        except ZeroDivisionError:
            logger.warning(
                f"Skipping GridDistortion due to ZeroDivisionError: "
                f"image shape={params['shape'][:2]}, "
                f"subject_id={data.get('subject_id')}, slice_idx={data.get('slice_idx')}"
            )
            return {"map_x": None, "map_y": None}

    def apply(self, img: np.ndarray, map_x: np.ndarray | None = None, **params: Any) -> np.ndarray:
        if map_x is None:
            return _restore_grayscale_channel_last(img, img)
        transformed = super().apply(img, map_x=map_x, **params)
        return _restore_grayscale_channel_last(img, transformed)


class RandomCrop(albumentations.RandomCrop):
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply_to_template_coords
        return targets

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, -1)
        transformed = super().apply_to_mask(mask, **params)
        return _restore_grayscale_channel_last(mask, transformed)

    def apply_to_template_coords(self, img: np.ndarray, **params: Any) -> np.ndarray:
        original_border_mode = self.border_mode
        self.border_mode = cv2.BORDER_REPLICATE
        try:
            transformed = super().apply(img, **params)
        finally:
            self.border_mode = original_border_mode
        return _restore_grayscale_channel_last(img, transformed)

class Crop(albumentations.Crop):
    @property
    def targets(self) -> dict[str, Any]:
        targets = dict(super().targets)
        targets["template_coords"] = self.apply_to_template_coords
        return targets

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        transformed = super().apply(img, **params)
        return _restore_grayscale_channel_last(img, transformed)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, -1)
        transformed = super().apply_to_mask(mask, **params)
        return _restore_grayscale_channel_last(mask, transformed)

    def apply_to_template_coords(self, img: np.ndarray, **params: Any) -> np.ndarray:
        original_border_mode = self.border_mode
        self.border_mode = cv2.BORDER_REPLICATE
        try:
            transformed = super().apply(img, **params)
        finally:
            self.border_mode = original_border_mode
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
        acquisition_axes = data["acquisition_axes"]
        slice_axis = data["slice_axis"]

        # Compute output shape
        img = data["image"]
        h, w = img.shape[:2]
        axes = sorted(acquisition_axes, key=lambda a: a.dimension)
        axes = [a for a in axes if a.dimension != slice_axis.dimension]
        scale_y = axes[0].resolution * 2**3 / self._fixed_resolution
        scale_x = axes[1].resolution * 2**3 / self._fixed_resolution
        new_h = int(round(h * scale_y))
        new_w = int(round(w * scale_x))

        return {
            "acquisition_axes": acquisition_axes,
            "slice_axis": slice_axis,
            "orientation": data["orientation"],
            "shape": (new_h, new_w),
            "subject_id": data.get("subject_id"),
            "slice_idx": data.get("slice_idx"),
        }

    def apply(self, img: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation, subject_id: str = "", slice_idx: int = -1, **params: Any) -> np.ndarray:
        transformed = self._resample(x=img, acquisition_axes=acquisition_axes, slice_axis=slice_axis, subject_id=subject_id, slice_idx=slice_idx)
        return _restore_grayscale_channel_last(img, transformed)

    def apply_to_mask(self, mask: np.ndarray, *args: Any, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation, subject_id: str = "", slice_idx: int = -1, **params: Any) -> np.ndarray:
        if not self._resize_mask:
            return mask
        return self._resample(x=mask, acquisition_axes=acquisition_axes, slice_axis=slice_axis, subject_id=subject_id, slice_idx=slice_idx)

    def apply_to_template_coords(self, img: np.ndarray, *args: Any, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation, subject_id: str = "", slice_idx: int = -1, **params: Any) -> np.ndarray:
        if not self._resize_target:
            return img
        return self._resample(x=img, acquisition_axes=acquisition_axes, slice_axis=slice_axis, subject_id=subject_id, slice_idx=slice_idx)

    def _resample(self, x: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, subject_id: str = "", slice_idx: int = -1):
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
    rotation_angles: RotationAngles,
    square_symmetry: bool = False,
    resample_to_fixed_resolution: bool = False,
    rotate_slices: bool = False,
    normalize_template_points: bool = False,
    apply_grid_distortion: bool = False,
    apply_gaussian_noise: bool = False,
    apply_coarse_dropout: bool = False,
    apply_random_brightness_contrast: bool = False
):
    transforms: list[Any] = [ImageNormalization()]

    if apply_gaussian_noise:
        transforms.append(GaussNoise())

    if apply_coarse_dropout:
        transforms.append(CoarseDropout())

    if apply_random_brightness_contrast:
        transforms.append(RandomBrightnessContrast())

    if rotate_slices:
        transforms.append(Rotate(rotation_angles=rotation_angles, border_mode=cv2.BORDER_REPLICATE))

    if square_symmetry:
        transforms.append(SquareSymmetry())

    if apply_grid_distortion:
        transforms.append(GridDistortion(border_mode=cv2.BORDER_REPLICATE))

    if resample_to_fixed_resolution:
        assert config.resample_to_fixed_resolution is not None
        transforms.append(Resample(fixed_resolution=config.resample_to_fixed_resolution))

    if normalize_template_points:
        transforms.append(TemplatePointsNormalization(
            origin=template_parameters.origin,
            scale=template_parameters.scale,
            direction=template_parameters.direction,
            shape=template_parameters.shape
        ))

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

    if len(transforms) > 0:
        return albumentations.Compose(transforms)
    return None


def physical_to_index_space(
        physical_pts: torch.Tensor | np.ndarray,
        template_parameters: TemplateParameters,
        channel_dim: int = 2
):
    points = physical_pts.clone() if isinstance(physical_pts, torch.Tensor) else physical_pts.copy()
    for dim in range(template_parameters.dims):
        if channel_dim == 2:
            assert points.shape[2] == 3
            points[:, :, dim] -= template_parameters.origin[dim]
            points[:, :, dim] *= template_parameters.direction[dim]
            points[:, :, dim] /= template_parameters.scale[dim]
        elif channel_dim == 1:
            assert points.shape[1] == 3
            points[:, dim] -= template_parameters.origin[dim]
            points[:, dim] *= template_parameters.direction[dim]
            points[:, dim] /= template_parameters.scale[dim]
        else:
            raise NotImplementedError

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
            f"ML index mean: {template_points_index_space[:, :, 0].mean():.1f}, "
            f"ML index range: {template_points_index_space[:, :, 0].min():.1f} - {template_points_index_space[:, :, 0].max():.1f} "
            f"midpoint: {template_parameters.shape[0] / 2:.1f}, "
            f"need_mirror: {need_mirror}")
    if need_mirror:
        # map to right hemisphere
        template_points = mirror_points(points=template_points, template_parameters=template_parameters)
    return template_points

def mirror_points(
        points: torch.Tensor | np.ndarray,
        template_parameters: TemplateParameters
):
    """
    Mirrors points over the SI axis (only relevant for sagittal slices)

    :param points: BxCxHxW
    :param template_parameters:
    :return:
    """
    assert len(points.shape) == 4 and points.shape[1] == 3
    points = points.clone() if isinstance(points, torch.Tensor) else points.copy()

    # 1. Convert to index space
    points = physical_to_index_space(
        physical_pts=points,
        template_parameters=template_parameters,
        channel_dim=1
    )

    # 2. Flip ML in index space
    points[:, 0] = template_parameters.shape[0]-1 - points[:, 0]

    # 3. Convert back to physical
    for dim in range(template_parameters.dims):
        points[:, dim] *= template_parameters.scale[dim]
        points[:, dim] *= template_parameters.direction[dim]
        points[:, dim] += template_parameters.origin[dim]

    return points