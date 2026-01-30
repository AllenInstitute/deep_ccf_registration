from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from deep_ccf_registration.datasets.aquisition_meta import AcquisitionDirection


class AcqusitionAxesName(Enum):
    X = 'X'
    Y = 'Y'
    Z = 'Z'

class AcquisitionAxis(BaseModel):
    dimension: int
    direction: AcquisitionDirection
    name: AcqusitionAxesName
    unit: str
    resolution: float

class SliceOrientation(Enum):
    SAGITTAL = 'sagittal'
    CORONAL = 'coronal'
    HORIZONTAL = 'horizontal'

class SubjectMetadata(BaseModel):
    subject_id: str
    stitched_volume_path: str | Path
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registration_downsample: int
    # The index that splits the 2 hemispheres in voxels the same dim as the sagittal axis in the registered volume
    # obtained via `get_input_space_midline.py`
    sagittal_midline: Optional[int] = None
    ls_to_template_inverse_warp_path_original: Path
    ls_to_template_affine_matrix_path: Path

    def get_slice_shape(self, orientation: SliceOrientation):
        axes_except_slice = [ax for ax in self.axes if ax != self.get_slice_axis(orientation=orientation)]
        slice_shape = tuple(
            self.registered_shape[ax.dimension] for ax in axes_except_slice)
        return slice_shape

    def get_slice_axis(self, orientation: SliceOrientation) -> AcquisitionAxis:
        """
        Get the acquisition axis corresponding to the slice orientation.

        Parameters
        ----------
        orientation : SliceOrientation
            Desired slice orientation.

        Returns
        -------
        AcquisitionAxis
            The axis along which slicing occurs.

        Raises
        ------
        ValueError
            If no unique axis matches the orientation.
        NotImplementedError
            If orientation other than SAGITTAL is requested.
        """
        if orientation == SliceOrientation.SAGITTAL:
            slice_axis = [i for i in range(len(self.axes)) if
                          self.axes[i].direction in (AcquisitionDirection.LEFT_TO_RIGHT,
                                                AcquisitionDirection.RIGHT_TO_LEFT)]
            if len(slice_axis) != 1:
                raise ValueError(f'expected to find 1 sagittal axis but found {len(slice_axis)}')
            slice_axis = self.axes[slice_axis[0]]
        else:
            raise NotImplementedError(f'{orientation} not supported')
        return slice_axis


class TissueBoundingBox(BaseModel):
    """
    start y, x and width, height of tissue bounding boxes, obtained via
    `get_tissue_bounding_box.py`
    """
    y: int
    x: int
    width: int
    height: int


class TissueBoundingBoxes(BaseModel):
    bounding_boxes: dict[str, list[Optional[TissueBoundingBox]]]
