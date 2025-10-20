"""
This script applies the transforms to a given slice and plots the resulting mapping
template points
"""
import json
from pathlib import Path
from aind_smartspim_transform_utils.CoordinateTransform import CoordinateTransform
import ants
import click
from deep_ccf_registration.datasets.slice_dataset import AcquisitionDirection, SliceDataset, \
    SliceOrientation, SubjectMetadata
from deep_ccf_registration.utils.utils import visualize_alignment
from loguru import logger
import tensorstore
import pandas as pd
import torch


# class AcqusitionAxesName(Enum):
#     X = 'X'
#     Y = 'Y'
#     Z = 'Z'

# class AcquisitionAxis(BaseModel):
#     dimension: int
#     direction: AcquisitionDirection
#     name: AcqusitionAxesName
#     unit: str


# def _create_coordinate_dataframe(height: int, width: int, fixed_index_value: int, slice_axis: AcquisitionAxis, axes: list[AcquisitionAxis]) -> pd.DataFrame:
#     axis1_coords, axis2_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

#     # Flatten the coordinate arrays
#     axis1_flat = axis1_coords.flatten()
#     axis2_flat = axis2_coords.flatten()

#     n_points = len(axis1_flat)

#     slice_index = np.full(n_points, fixed_index_value)

#     axes = sorted(axes, key=lambda x: x.dimension)

#     points = np.zeros((n_points, 3))

#     points[:, slice_axis.dimension] = slice_index
#     points[:, [x for x in axes if x != slice_axis][0].dimension] = axis1_flat
#     points[:, [x for x in axes if x != slice_axis][1].dimension] = axis2_flat

#     df = pd.DataFrame(data=points, columns=[x.name.value.lower() for x in axes]).astype(float)

#     return df

# def forward_transform_points(stitched_path: Path, acquisition_path: Path, affine_path: Path, inverse_warp_path: Path, slice_index: int):

#     base_volume = tensorstore.open(
#         {'driver': 'zarr',
#          'kvstore': {'driver': 'file', 'path': str(stitched_path / '0')}
#         }, read=True).result()

#     with open(acquisition_path) as f:
#         acquisition = json.load(f)

#     acquisition_axes = [AcquisitionAxis.model_validate(x) for x in acquisition['axes']]


#     slice_axis = [i for i in range(len(acquisition_axes)) if
#                   acquisition_axes[i].direction in (AcquisitionDirection.LEFT_TO_RIGHT,
#                                         AcquisitionDirection.RIGHT_TO_LEFT)]
#     slice_axis = acquisition_axes[slice_axis[0]]

#     spatial_dims = sorted([x for x in acquisition_axes if x != slice_axis], key=lambda x: x.dimension)

#     slice_height, slice_width = [base_volume.shape[spatial_dims[0].dimension + 2], base_volume.shape[spatial_dims[1].dimension + 2]]

#     coord_transform = CoordinateTransform(
#         name='smartspim_lca',
#         dataset_transforms={
#             'points_to_ccf': [
#                 str(affine_path),
#                 str(inverse_warp_path),
#             ]
#         },
#         acquisition=acquisition,
#         image_metadata={'shape': base_volume.shape[2:]},
#         ccf_transforms={
#             'points_to_ccf': [
#                 '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_0GenericAffine_25.mat',
#                 '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_1InverseWarp_25.nii.gz'
#             ],
#             'points_from_ccf': [
#                 '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_1Warp_25.nii.gz',
#                 '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_0GenericAffine_25.mat'
#             ]
#         },
#         ccf_template_path='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/ccf_average_template_25.nii.gz',
#         ls_template_path='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz'
#     )
#     point_grid = _create_coordinate_dataframe(
#         height=slice_height,
#         width=slice_width,
#         fixed_index_value=slice_index,
#         axes=acquisition_axes,
#         slice_axis=slice_axis
#     )

#     ccf_template_points = coord_transform.forward_transform(
#         points=point_grid,
#     )

#     return point_grid, ccf_template_points, slice_axis, slice_height, slice_width

def reverse_transform_points(
        affine_path, warp_path, template_points: pd.DataFrame, stitched_path, acquisition_path
):
    base_volume = tensorstore.open(
        {'driver': 'zarr',
         'kvstore': {'driver': 'file', 'path': str(stitched_path / '0')}
         }, read=True).result()

    with open(acquisition_path) as f:
        acquisition = json.load(f)

    coord_transform = CoordinateTransform(
        name='smartspim_lca',
        dataset_transforms={
            'points_from_ccf': [
                str(warp_path),
                str(affine_path),
            ]
        },
        acquisition=acquisition,
        image_metadata={'shape': base_volume.shape[2:]},
        ccf_transforms={
            'points_to_ccf': [
                '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_0GenericAffine_25.mat',
                '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_1InverseWarp_25.nii.gz'
            ],
            'points_from_ccf': [
                '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_1Warp_25.nii.gz',
                '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_0GenericAffine_25.mat'
            ]
        },
        ccf_template_path='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/ccf_average_template_25.nii.gz',
        ls_template_path='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz'
    )

    input_points = coord_transform.reverse_transform(
        points=template_points,
    )

    return input_points


# def visualize_alignment(registered_volume: tensorstore.TensorStore, template_points: np.ndarray, template: ants.ANTsImage, slice_index: int, slice_axis):
#     volume_slice = [0, 0, slice(None), slice(None), slice(None)]
#     volume_slice[slice_axis.dimension + 2] = slice_index
#     input_slice = registered_volume[tuple(volume_slice)].read().result()

#     template_points = resize(template_points, output_shape=(*input_slice.shape, 3), anti_aliasing=True)

#     raw_rgb = np.zeros_like(input_slice, shape=(*input_slice.shape, 3))
#     raw_rgb[:, :, 0] = input_slice

#     template_on_input = map_coordinates(
#         input=template.numpy(),
#         coordinates=template_points.reshape((-1, 3)).T
#     )
#     template_on_input_rgb = np.zeros_like(template_on_input,
#                                              shape=(*input_slice.shape, 3))
#     template_on_input_rgb[:, :, 2] = np.array(template_on_input).reshape(input_slice.shape)

#     height, width = input_slice.shape
#     if width > height:
#         figsize = (30, 15)
#     else:
#         figsize = (15, 30)

#     fig, ax = plt.subplots(figsize=figsize, ncols=3, dpi=100)
#     ax[0].imshow(rescale_intensity(raw_rgb, out_range=(0, 1)), alpha=0.8)
#     ax[0].imshow(rescale_intensity(template_on_input_rgb, out_range=(0, 1)), alpha=0.4)
#     ax[1].imshow(input_slice, cmap='gray')
#     ax[1].set_title('input')
#     ax[2].imshow(np.array(template_on_input).reshape(input_slice.shape), cmap='gray')
#     ax[2].set_title('Template')
#     return fig

@click.command()
@click.option('--subject-id', required=True)
@click.option('--dataset-meta-path', type=click.Path(path_type=Path))
def main(subject_id: str, dataset_meta_path: Path):
    with open(dataset_meta_path) as f:
        dataset_meta = json.load(f)
    dataset_meta: list[SubjectMetadata] = [SubjectMetadata.model_validate(x) for x in dataset_meta]

    subject_meta: SubjectMetadata = [x for x in dataset_meta if x.subject_id == subject_id][0]

    sagittal_axis = [i for i in range(len(subject_meta.axes)) if
                     subject_meta.axes[i].direction in (AcquisitionDirection.LEFT_TO_RIGHT,
                                                        AcquisitionDirection.RIGHT_TO_LEFT)]
    sagittal_axis = subject_meta.axes[sagittal_axis[0]]

    slice_index = int(subject_meta.registered_shape[sagittal_axis.dimension] / 2)

    ls_template = ants.image_read(
        '/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz')

    slice_dataset = SliceDataset(
        ls_template=ls_template,
        dataset_meta=[subject_meta],
        orientation=SliceOrientation.SAGITTAL,
    )

    logger.info('forward transform')
    input_slice, output_points, dataset_idx, slice_idx = slice_dataset[slice_index]

    logger.info('generating plot')
    fig = visualize_alignment(
        input_slice=torch.tensor(input_slice),
        template_points=torch.tensor(output_points),
        template=ls_template
    )
    fig.savefig(f'/results/{subject_meta.subject_id}.png')

    # logger.info('reverse transform')
    # input_points_from_template = reverse_transform_points(
    #     stitched_path=subject_meta.stitched_volume_path,
    #     acquisition_path=subject_meta.stitched_volume_path.parent.parent.parent / 'acquisition.json',
    #     affine_path=subject_meta.ls_to_template_affine_matrix_path,
    #     warp_path=subject_meta.ls_to_template_inverse_warp_path.parent / 'ls_to_template_SyN_1Warp.nii.gz',
    #     template_points=ccf_template_points
    # )

    # convert points to microns
    # axes = sorted(subject_meta.axes, key=lambda x: x.dimension)
    # axis_resolution = np.array([x.resolution for x in axes])
    # input_points_microns = input_points.values * axis_resolution
    # input_points_from_template_microns = input_points_from_template.values * axis_resolution

    # distance = np.mean(np.sqrt(((input_points_microns - input_points_from_template_microns) ** 2).sum(axis=1)))

    # registered_volume = tensorstore.open(
    #     {'driver': 'zarr',
    #      'kvstore': {'driver': 'file', 'path': str(subject_meta.stitched_volume_path / f'{subject_meta.registration_downsample}')}
    #     }, read=True).result()

    # fig = visualize_alignment(
    #     registered_volume=registered_volume,
    #     template_points=ccf_template_points.values.reshape((slice_height, slice_width, 3)),
    #     template=ants.image_read('/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/ccf_average_template_25.nii.gz'),
    #     slice_index=int(full_res_slice_index / 2 ** subject_meta.registration_downsample),
    #     slice_axis=sagittal_axis
    # )
    # fig.savefig(f'/results/{subject_meta.subject_id}.png')

    # with open(f'/results/{subject_meta.subject_id}_roundtrip_distance.json', 'w') as f:
    #     f.write(json.dumps({'roundtrip_distance': float(distance)}))


if __name__ == '__main__':
    main()
