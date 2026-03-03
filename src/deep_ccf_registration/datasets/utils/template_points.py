import json
from dataclasses import dataclass
from pathlib import Path

import ants
import aind_smartspim_transform_utils
import numpy as np
import tensorstore
from aind_smartspim_transform_utils.utils.utils import get_orientation, \
    convert_to_ants_space, convert_from_ants_space
from concurrent.futures import ThreadPoolExecutor

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import AcquisitionAxis
from deep_ccf_registration.utils.logging_utils import timed
from deep_ccf_registration.datasets.utils.interpolation import map_coordinates_cropped


@dataclass
class Affine:
    """Pre-computed inverse affine transform from the ANTs .mat file.

    ANTs affine transform: y = A @ (x - c) + c + t
    where A is the matrix, t is translation, c is center.

    See: https://github.com/ANTsX/ANTsPy/wiki/ANTs-transform-concepts-and-file-formats
    """
    A_inv: np.ndarray  # (3, 3) inverse of matrix A
    c: np.ndarray  # (3,) center of rotation
    t: np.ndarray  # (3,) translation vector

    @classmethod
    def from_ants_file(cls, affine_path: Path) -> "Affine":
        """Load ANTs affine and precompute inverse."""
        tx = ants.read_transform(str(affine_path))
        params = np.array(tx.parameters)
        A = params[:9].reshape(3, 3)
        t = params[9:12]
        c = np.array(tx.fixed_parameters)
        A_inv = np.linalg.inv(A)
        return cls(A_inv=A_inv, c=c, t=t)

    def apply_inverse(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse affine transform to points.

        ANTs forward:
        --------------
        y = A@x + t + c - A@c
        y = A@x - A@c + c + t
        y = A@(x-c) + c + t

        ANTs inverse:
        --------------
        y = A@(x - c) + c + t
        y - c - t = A@(x - c)
        A_inv @ (y - c - t) = x - c
        x = A_inv @ (y - c - t) + c
        """
        return (x - self.c - self.t) @ self.A_inv.T + self.c


def transform_points_to_template_space(
    acquisition_axes: list[AcquisitionAxis],
    ls_template_info: TemplateParameters,
    points: np.ndarray,
    input_volume_shape: tuple[int, int, int],
    template_resolution: int = 25,
    registration_downsample: float = 3.0,
) -> np.ndarray:
    """
    Transform points from input index space to physical template ANTs space.

    Performs orientation alignment, scaling, and coordinate system conversion
    to map points from acquisition space to template space.

    Parameters
    ----------
    acquisition_axes : list[AcquisitionAxis]
        Acquisition axes defining the input volume orientation.
    ls_template_info : AntsImageParameters
        Template image parameters.
    points : np.ndarray
        Points in input volume coordinates.
    input_volume_shape : tuple[int, int, int]
        Shape of the input volume.
    template_resolution : int, default=25
        Resolution of the template in micrometers.
    registration_downsample : float, default=3.0
        Downsample factor used during registration.

    Returns
    -------
    np.ndarray
        Points in ANTs template space.
    """
    acquisition_axes = sorted(acquisition_axes, key=lambda x: x.dimension)

    orient = get_orientation([json.loads(x.model_dump_json()) for x in acquisition_axes])

    _, swapped, mat = aind_smartspim_transform_utils.utils.utils.get_orientation_transform(
        orient, ls_template_info.orientation
    )

    # flip axis based on the template orientation relative to input image
    for idx, dim_orient in enumerate(mat.sum(axis=1)):
        if dim_orient < 0:
            points[:, idx] = input_volume_shape[idx] - points[:, idx]

    # scale points
    points_resolution = [x.resolution * 2 ** registration_downsample for x in acquisition_axes]
    scaling = [res_1 / res_2 for res_1, res_2 in zip(points_resolution, [template_resolution] * 3)]
    scaled_pts = scale_points(points=points, scaling=scaling)

    # orient axes to template
    orient_pts = scaled_pts[:, swapped]

    # convert points into ccf space
    ants_pts = convert_to_ants_space(
        ls_template_info, orient_pts
    )

    return ants_pts

def scale_points(points: np.ndarray, scaling: list[float]) -> np.ndarray:
    assert points.shape[1] == 3 and len(points.shape) == 2
    assert len(scaling) == 3
    scale = np.array([scaling])
    points *= scale
    return points

def apply_transforms_to_points(
    points: np.ndarray,
    affine: Affine,
    warp: tensorstore.TensorStore,
    template_parameters: TemplateParameters,
) -> np.ndarray:
    """
    Apply affine and non-linear transformations to points

    Transforms points from input space to template space by applying inverse affine
    transformation followed by displacement field warping.

    Parameters
    ----------
    points : np.ndarray
        Points in physical input space to be transformed.
    affine : Affine
        Pre-computed inverse affine transform.
    warp : CxHxWxD displacement vector for each voxel in HxWxD template
    template_parameters : AntsImageParameters
        Template image parameters.

    Returns
    -------
    np array of shape n points x 3. The 2nd dim is ordered ["ML", "AP", "DV"] according to light sheet template orientation.
    The points are in physical space.
    """
    # apply inverse affine to points in input space
    # this returns points in physical space

    with timed():
        affine_transformed_points = affine.apply_inverse(points)

    # convert physical points to voxels,
    # so we can index into the displacement field
    with timed():
        affine_transformed_voxels = convert_from_ants_space(
            template_parameters=template_parameters,
            physical_pts=affine_transformed_points
        )

    coords = affine_transformed_voxels.T
    with timed():
        displacements = map_coordinates_cropped(
            volume=warp,
            coords=coords,
            order=1,
            mode='nearest'
        )

    with timed():
        # apply displacement vector to affine transformed points
        transformed_points = affine_transformed_points + displacements

    return transformed_points