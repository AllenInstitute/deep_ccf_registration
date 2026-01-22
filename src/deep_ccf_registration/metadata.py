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
            # Extract the prefix from stitched_volume_path
            # e.g., s3://aind-open-data/SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr
            # extract: SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08
            if self.stitched_volume_path.startswith('s3://'):
                # Split by '/' and get the 4th part (index 3) which is the prefix
                parts = self.stitched_volume_path.split('/')
                prefix = parts[3]  # SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08
            else:
                raise ValueError('template_points_path not passed and stitched_volume_path is not an s3 uri. Cannot infer template_points_path')
            # the template points calculated by scripts/create_point_map.py
            template_points_path = f"s3://marmot-development-802451596237-us-west-2/smartspim-registration/{prefix}/{self.subject_id}_template_points.zarr"
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
