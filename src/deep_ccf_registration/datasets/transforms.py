from typing import Any

import albumentations
import numpy as np
from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection
from skimage.exposure import rescale_intensity

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.metadata import SliceOrientation, AcquisitionAxis


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