from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class AcquisitionDirection(Enum):
    LEFT_TO_RIGHT = 'Left_to_right'
    RIGHT_TO_LEFT = 'Right_to_left'
    POSTERIOR_TO_ANTERIOR = 'Posterior_to_anterior'
    ANTERIOR_TO_POSTERIOR = 'Anterior_to_posterior'
    SUPERIOR_TO_INFERIOR = 'Superior_to_inferior'
    INFERIOR_TO_SUPERIOR = 'Inferior_to_superior'

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
    template_points_path: Optional[str] = None
    axes: list[AcquisitionAxis]
    registered_shape: tuple[int, int, int]
    registration_downsample: int
    # The index that splits the 2 hemispheres in voxels the same dim as the sagittal axis in the registered volume
    # obtained via `get_input_space_midline.py`
    sagittal_midline: Optional[int] = None

    def get_template_points_path(self) -> str:
        """
        If template_points_path is not passed, infer it from the stitched_path root prefix

        :return:
        """
        if self.template_points_path is not None:
            template_points_path = self.template_points_path
        else:
            # Extract the first prefix from stitched_volume_path
            # e.g., s3://aind-open-data/SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr
            # becomes s3://aind-open-data/SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08
            if self.stitched_volume_path.startswith('s3://'):
                # Split by '/' and take the first 4 parts: s3://bucket/prefix
                parts = self.stitched_volume_path.split('/')
                root_prefix = '/'.join(parts[:4])  # s3://aind-open-data/SmartSPIM_...
            else:
                raise ValueError('template_points_path not passed and stitched_volume_path is not an s3 uri. Cannot infer template_points_path')
            template_points_path = f"{root_prefix}/{self.subject_id}_template_points.zarr"
        return template_points_path

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
