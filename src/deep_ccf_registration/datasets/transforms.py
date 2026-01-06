from dataclasses import dataclass
from enum import Enum
from typing import Any

import albumentations
import cv2
import numpy as np
import torch
from skimage.exposure import rescale_intensity

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.metadata import SliceOrientation, AcquisitionAxis


class AcquisitionDirection(Enum):
    LEFT_TO_RIGHT = 'Left_to_right'
    RIGHT_TO_LEFT = 'Right_to_left'
    POSTERIOR_TO_ANTERIOR = 'Posterior_to_anterior'
    ANTERIOR_TO_POSTERIOR = 'Anterior_to_posterior'
    SUPERIOR_TO_INFERIOR = 'Superior_to_inferior'
    INFERIOR_TO_SUPERIOR = 'Inferior_to_superior'


@dataclass
class TemplateParameters:
    origin: tuple
    scale: tuple
    direction: tuple
    shape: tuple
    dims: int = 3


class LongestMaxSize(albumentations.DualTransform):
    """Resize so the longest side matches ``max_size`` while keeping aspect."""

    def __init__(self, max_size: int, interpolation: int = cv2.INTER_LINEAR, is_train: bool = True):
        super().__init__(p=1.0)
        self.max_size = max_size
        self.interpolation = interpolation
        self.is_train = is_train

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
        return cv2.resize(img, (width, height), interpolation=self.interpolation)

    def apply_to_keypoints(self, keypoints: np.ndarray, scale: float, height: int, width: int, **params: Any) -> np.ndarray:
        if not self.is_train or scale == 1.0:
            return keypoints
        return self._resize_array(keypoints, height=height, width=width)

    def _resize_array(self, array: np.ndarray, height: int, width: int) -> np.ndarray:
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Template keypoints must be HxWx3")
        channels = [
            cv2.resize(array[..., idx], (width, height), interpolation=self.interpolation)
            for idx in range(3)
        ]
        return np.stack(channels, axis=-1)


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


class OrientationNormalization(albumentations.DualTransform):
    def __init__(self):
        super().__init__(p=1.0)
        self._normalize_orientation_map = {
            SliceOrientation.SAGITTAL: [AcquisitionDirection.SUPERIOR_TO_INFERIOR,
                                        AcquisitionDirection.ANTERIOR_TO_POSTERIOR]
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        return {
            "acquisition_axes": data["acquisition_axes"],
            "slice_axis": data["slice_axis"],
            "orientation": data["orientation"],
        }

    def apply(self, img: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation,  **params: Any) -> np.ndarray:
        return self._normalize_orientation(x=img, acquisition_axes=acquisition_axes, slice_axis=slice_axis, orientation=orientation)

    def apply_to_keypoints(self, keypoints: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation,  **params: Any) -> np.ndarray:
        return self._normalize_orientation(x=keypoints, acquisition_axes=acquisition_axes, slice_axis=slice_axis, orientation=orientation)
    
    def _normalize_orientation(self, x: np.ndarray, acquisition_axes: list[AcquisitionAxis], slice_axis: AcquisitionAxis, orientation: SliceOrientation) -> np.ndarray:
        desired_orientation: list[AcquisitionDirection] = self._normalize_orientation_map[orientation]
        acquisition_axes = sorted(acquisition_axes, key=lambda x: x.dimension)

        desired_orientation = desired_orientation.copy()
        desired_orientation.insert(slice_axis.dimension, slice_axis.direction)

        _, swapped, mat = _get_orientation_transform(
            orientation_in=''.join([x.direction.value.lower()[0] for x in acquisition_axes]),
            orientation_out=''.join([x.value.lower()[0] for x in desired_orientation])
        )

        # exclude the slice axis, since just dealing with 2d slices
        mat = mat[[x for x in range(3) if x != slice_axis.dimension]]
        mat = mat[:, [x for x in range(3) if x != slice_axis.dimension]]

        # flip axis to desired orientation
        for idx, dim_orient in enumerate(mat.sum(axis=1)):
            if dim_orient < 0:
                if idx == 0:
                    x = np.flipud(x)
                else:
                    x = np.fliplr(x)

        if swapped.tolist() != list(range(3)):
            x = np.swapaxes(x, 0, 1)

        return x


def _get_orientation_transform(
    orientation_in: str, orientation_out: str
) -> tuple:
    """
    Takes orientation acronyms (i.e. spr) and creates a convertion matrix for
    converting from one to another

    Parameters
    ----------
    orientation_in : str
        the current orientation of image or cells (i.e. spr)
    orientation_out : str
        the orientation that you want to convert the image or
        cells to (i.e. ras)

    Returns
    -------
    tuple
        the location of the values in the identity matrix with values
        (original, swapped)
    """

    reverse_dict = {"r": "l", "l": "r", "a": "p", "p": "a", "s": "i", "i": "s"}

    input_dict = {dim.lower(): c for c, dim in enumerate(orientation_in)}
    output_dict = {dim.lower(): c for c, dim in enumerate(orientation_out)}

    transform_matrix = np.zeros((3, 3))
    for k, v in input_dict.items():
        if k in output_dict.keys():
            transform_matrix[v, output_dict[k]] = 1
        else:
            k_reverse = reverse_dict[k]
            transform_matrix[v, output_dict[k_reverse]] = -1

    if orientation_in.lower() == "spl" or orientation_out.lower() == "spl":
        transform_matrix = abs(transform_matrix)

    original, swapped = np.where(transform_matrix.T)

    return original, swapped, transform_matrix

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

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        # just a passthrough
        return img

    def apply_to_keypoints(self, keypoints: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        # repurposing "keypoints" transform for the slice template points
        extent_min, extent_max = self._physical_extent
        return 2 * (keypoints - extent_min) / (extent_max - extent_min) - 1

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

def get_physical_extent(origin, scale, direction, shape):
    origin = np.array(origin)
    scale = np.array(scale)
    direction = np.array(direction)
    shape = np.array(shape)

    extent = shape * scale
    extent_min = np.where(direction > 0, origin, origin + direction * extent)
    extent_max = np.where(direction > 0, origin + extent, origin)

    return extent_min, extent_max

def build_transform(config: TrainConfig, is_train: bool, template_parameters: TemplateParameters):
    transforms = []
    transforms.append(LongestMaxSize(max_size=512, is_train=is_train))

    if config.normalize_input_image:
        transforms.append(ImageNormalization())
    if config.normalize_orientation:
        transforms.append(OrientationNormalization())
    if is_train:
        transforms.append(TemplatePointsNormalization(
            origin=template_parameters.origin,
            scale=template_parameters.scale,
            direction=template_parameters.direction,
            shape=template_parameters.shape
        ))
    if len(transforms) > 0:
        transforms = albumentations.Compose(transforms, seed=config.seed)
    else:
        transforms = None
    return transforms

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
    points_index_space = template_points.copy()
    for dim in range(template_parameters.dims):
        points_index_space[:, :, dim] -= template_parameters.origin[dim]
        points_index_space[:, :, dim] *= template_parameters.direction[dim]
        points_index_space[:, :, dim] /= template_parameters.scale[dim]

    # checks whether the ML points are > halfway in index space. the LS template iS RAS.
    # therefore this checks whether points are in left hemisphere
    need_mirror = (points_index_space[:, :, 0] > template_parameters.shape[0] / 2).all()
    if need_mirror:
        # map to right hemisphere
        template_points = mirror_points(points=template_points, template_parameters=template_parameters)
    return template_points


def mirror_points(points: torch.Tensor | np.ndarray, template_parameters: TemplateParameters):
    points = points.clone() if isinstance(points, torch.Tensor) else points.copy()

    # 1. Convert to index space
    for dim in range(template_parameters.dims):
        points[:, :, dim] -= template_parameters.origin[dim]
        points[:, :, dim] *= template_parameters.direction[dim]
        points[:, :, dim] /= template_parameters.scale[dim]

    # 2. Flip ML in index space
    points[:, :, 0] = template_parameters.shape[0]-1 - points[:, :, 0]

    # 3. Convert back to physical
    for dim in range(template_parameters.dims):
        points[:, :, dim] *= template_parameters.scale[dim]
        points[:, :, dim] *= template_parameters.direction[dim]
        points[:, :, dim] += template_parameters.origin[dim]

    return points