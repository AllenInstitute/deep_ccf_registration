from enum import Enum
from pathlib import Path
from typing import Optional

from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection, AcqusitionAxesName
from pydantic import BaseModel


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
    template_points_path: str
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registration_downsample: int
    # The index that splits the 2 hemispheres in voxels the same dim as the sagittal axis in the registered volume
    # obtained via `get_input_space_midline.py`
    sagittal_midline: Optional[int] = None

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
