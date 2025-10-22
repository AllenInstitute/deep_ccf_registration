"""
This script converts the inverse transforms needed for transforming a point in input space to
light sheet template space into ome-zarr. This serves 2 purposes:

1. Accessing the niftii warp file is slow compared to zarr
2. This serves as a prototype for implementing RFC-4 and RFC-5 under https://ngff.openmicroscopy.org/rfc/

"""
import json
import os
from pathlib import Path

import ants
import click
import zarr
import aind_smartspim_transform_utils.utils
from aind_smartspim_transform_utils.utils.utils import AntsImageParameters
import numpy as np
from loguru import logger
from obstore.store import from_url
from zarr.storage import ObjectStore, LocalStore

from deep_ccf_registration.datasets.slice_dataset import SubjectMetadata

LIGHT_SHEET_COORDINATE_SYSTEM = {
    "name": "light_sheet_template",
    "axes": [
        {
            "name": "x",
            "orientation": {"type": "anatomical", "value": "right-to-left"},
            "type": "space",
            "unit": "micrometer"
        },
        {
            "name": "y",
            "orientation": {"type": "anatomical", "value": "anterior-to-posterior"},
            "type": "space",
            "unit": "micrometer"
        },
        {
            "name": "z",
            "orientation": {"type": "anatomical", "value": "superior-to-inferior"},
            "type": "space",
            "unit": "micrometer"
        }
    ]
}

def _get_input_space_to_light_sheet_transform(
    experiment_meta: SubjectMetadata,
    light_sheet_template_path: Path,
    input_coordinate_system: dict,
    template_resolution: float = 25
):
    orient = aind_smartspim_transform_utils.utils.utils.get_orientation(
        [json.loads(x.model_dump_json()) for x in experiment_meta.axes]
    )

    ls_template = ants.image_read(str(light_sheet_template_path))
    ls_template_info = AntsImageParameters.from_ants_image(ls_template)
    _, permutation, mirror_transform_mat = aind_smartspim_transform_utils.utils.utils.get_orientation_transform(
        orient, ls_template_info.orientation
    )

    axes_require_mirror = np.where(mirror_transform_mat.sum(axis=1) < 0)[0]
    mirror_translation = np.zeros((3, 1))
    mirror_translation[axes_require_mirror, 0] = np.array(experiment_meta.registered_shape)[
        axes_require_mirror]

    coordinate_transformations = [
        {
            "name": "mirror axes (e.g. right to left -> left to right)",
            "type": "affine",
            "affine": np.hstack([mirror_transform_mat, mirror_translation]).tolist()
        },
        {
            "name": "light sheet raw -> light sheet template resolution",
            "type": "scale",
            "scale": (np.array([x.resolution * 2 ** experiment_meta.registration_downsample for x in sorted(experiment_meta.axes, key=lambda x: x.dimension)]) / template_resolution).tolist()
        },

        {
            "name": "axes permutation",
            "type": "mapAxis",
            "mapAxis": {LIGHT_SHEET_COORDINATE_SYSTEM["axes"][permutation[i]]["name"]:
                            experiment_meta.axes[i].direction.value for i in range(3)},
            "input": input_coordinate_system["name"],
            "output": LIGHT_SHEET_COORDINATE_SYSTEM["name"]
        }
    ]

    return coordinate_transformations


@click.command(help="Convert point transformations to ome zarr for an subject id")
@click.option(
    '--dataset-metadata-path',
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to file containing metadata that conforms with `list[SubjectMetadata]`",
    required=True
)
@click.option(
    '--light-sheet-template-path',
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to light sheet template",
    required=True
)
@click.option(
    '--subject-id',
    type=str,
    help="Experiment id to process",
    required=True
)
@click.option(
    '--output-path',
    type=str,
    help='Path to root folder of zarr store',
    required=True
)
@click.option(
    '--template-resolution',
    type=int,
    default=25,
    help='Template resolution in micrometers',
)
@click.option(
    '--chunk-size',
    type=int,
    default=256,
    help='Cubic chunk size for warp',
)
@click.option(
    '--warp-precision',
    type=str,
    default='float32',
    help='Warp precision',
)
@click.option(
    '--chunks-per-shard-dim',
    type=int,
    default=2,
    help='How many chunks per each spatial dim to group into a shard'
)
def main(dataset_metadata_path: Path, subject_id: str, output_path: str, template_resolution: int,
         light_sheet_template_path: Path,
         chunk_size: int = 256,
         warp_precision: str = 'float32',
         chunks_per_shard_dim: int =2):
    with open(dataset_metadata_path) as f:
        dataset_metadata = json.load(f)
    dataset_metadata = [SubjectMetadata.model_validate(x) for x in dataset_metadata]
    dataset_metadata = [x for x in dataset_metadata if x.subject_id == subject_id]
    if len(dataset_metadata) != 1:
        raise ValueError(
            f'expected 1 instance in dataset_metadata of exp id {subject_id} but got {len(dataset_metadata)}')
    experiment_meta = dataset_metadata[0]

    logger.info('Loading inverse warp')
    inverse_warp = ants.image_read(
        str('data' / Path(experiment_meta.ls_to_template_inverse_warp_path))).numpy()
    if inverse_warp.dtype != warp_precision:
        inverse_warp = inverse_warp.astype(warp_precision)
    affine = ants.read_transform(
        str('data' / experiment_meta.ls_to_template_affine_matrix_path)).parameters

    logger.info('creating zarr arrays')
    if output_path.startswith('s3://'):
        store = ObjectStore(store=from_url(url=output_path, region=os.environ['AWS_REGION']))
    else:
        store = LocalStore(root=output_path)
    root = zarr.create_group(store=store)

    chunks = (chunk_size, chunk_size, chunk_size, 3)

    zarr.create_array(store=store,
                      name='coordinateTransformations/ls_to_template_SyN_1InverseWarp',
                      data=inverse_warp,
                      chunks=chunks,
                      shards=(chunks[0] * chunks_per_shard_dim, chunks[1] * chunks_per_shard_dim, chunks[2] * chunks_per_shard_dim, chunks[-1]),
                      )
    zarr.create_array(store=store,
                      name='coordinateTransformations/ls_to_template_SyN_0GenericAffine',
                      data=affine.reshape((3, 4)))

    input_coordinate_system = {
        "name": "light_sheet_raw",
        "axes": [
            {
                "name": x.name.value,
                "orientation": {"type": "anatomical", "value": x.direction.value},
                "type": "space",
                "unit": "micrometer"
            }
            for x in sorted(experiment_meta.axes, key=lambda x: x.dimension)]
    }

    logger.info('getting input to light sheet transforms')
    input_to_ls_transform = _get_input_space_to_light_sheet_transform(
        experiment_meta=experiment_meta,
        light_sheet_template_path=light_sheet_template_path,
        input_coordinate_system=input_coordinate_system,
        template_resolution=template_resolution

    )

    logger.info('writing ome-zarr metadata')
    root.attrs['coordinateSystems'] = [
        input_coordinate_system,
        LIGHT_SHEET_COORDINATE_SYSTEM
    ]

    # Note: the bijection is not given in the forward direction as this direction
    # was not used

    root.attrs['coordinateTransformations'] = [
        {
            "type": "sequence",
            "transformations": input_to_ls_transform,
            "input": input_coordinate_system['name'],
            "output": LIGHT_SHEET_COORDINATE_SYSTEM['name']
        },
        {
            "type": "sequence",
            "transformations": [
                {
                    "type": "inverseOf",
                    "transformation": {
                        "type": "affine",
                        "path": "coordinateTransformations/ls_to_template_SyN_0GenericAffine"

                    },
                },
                {
                    "type": "displacements",
                    "path": "coordinateTransformations/ls_to_template_SyN_1InverseWarp",
                }
            ]
        }
    ]


if __name__ == '__main__':
    main()
