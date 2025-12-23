from dataclasses import dataclass
from enum import Enum
from typing import Any

import albumentations
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

class TemplatePointsNormalization:
    """
    Normalizes template points to [-1,1] (background points will be outside this range)
    """
    def __init__(self, origin: tuple[float, ...], scale: tuple[float, ...], direction: tuple[float, ...], shape: tuple[int, ...]):
        self._physical_extent = get_physical_extent(
            origin=origin,
            scale=scale,
            direction=direction,
            shape=shape
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        extent_min, extent_max = self._physical_extent
        return 2 * (x - extent_min) / (extent_max - extent_min) - 1

    def inverse(self, x):
        """
        Inverse transform: converts normalized coordinates [-1,1] back to physical coordinates.

        Works with both numpy arrays and torch tensors.
        """
        extent_min, extent_max = self._physical_extent

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

def build_transform(config: TrainConfig):
    transforms = []
    if config.patch_size[0] > 512:
        transforms.append(albumentations.LongestMaxSize(max_size=512))

    if config.normalize_input_image:
        transforms.append(ImageNormalization())
    if config.normalize_orientation:
        transforms.append(OrientationNormalization())
    if len(transforms) > 0:
        transforms = albumentations.Compose(transforms, seed=config.seed)
    else:
        transforms = None
    return transforms


@dataclass
class TemplateParameters:
    origin: tuple
    scale: tuple
    direction: tuple
    shape: tuple
    dims: int = 3

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
    flipped = points.clone() if isinstance(points, torch.Tensor) else points.copy()

    # 1. Convert to index space
    for dim in range(template_parameters.dims):
        flipped[:, :, dim] -= template_parameters.origin[dim]
        flipped[:, :, dim] *= template_parameters.direction[dim]
        flipped[:, :, dim] /= template_parameters.scale[dim]

    # 2. Flip ML in index space
    flipped[:, :, 0] = template_parameters.shape[0]-1 - flipped[:, 0]

    # 3. Convert back to physical
    for dim in range(template_parameters.dims):
        flipped[:, :, dim] *= template_parameters.scale[dim]
        flipped[:, :, dim] *= template_parameters.direction[dim]
        flipped[:, :, dim] += template_parameters.origin[dim]

    return flipped