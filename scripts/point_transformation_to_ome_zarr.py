import json
from pathlib import Path

import ants
import click
import zarr
import aind_smartspim_transform_utils.utils
from aind_smartspim_transform_utils.utils.utils import AntsImageParameters
import numpy as np

from deep_ccf_registration.datasets.slice_dataset import ExperimentMetadata

LIGHT_SHEET_COORDINATE_SYSTEM = {
    "name": "light_sheet_template",
    "axes": [
        {
            "name": "Right_to_left",
            "type": "space",
            "unit": "micrometer"
        },
        {
            "name": "Anterior_to_posterior",
            "type": "space",
            "unit": "micrometer"
        },
        {
            "name": "Superior_to_inferior",
            "type": "space",
            "unit": "micrometer"
        }
    ]
}


def _get_input_space_to_light_sheet_transform(
    experiment_meta: ExperimentMetadata,
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
            "scale": (np.array(
                experiment_meta.registered_resolution) / template_resolution).tolist()
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


@click.command(help="Convert point transformations to ome zarr for an experiment id")
@click.option(
    '--dataset-metadata-path',
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to file containing metadata that conforms with `list[ExperimentMetadata]`",
    required=True
)
@click.option(
    '--light-sheet-template-path',
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to light sheet template",
    required=True
)
@click.option(
    '--experiment-id',
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
def main(dataset_metadata_path: Path, experiment_id: str, output_path: str, template_resolution: int,
         light_sheet_template_path: Path):
    with open(dataset_metadata_path) as f:
        dataset_metadata = json.load(f)
    dataset_metadata = [ExperimentMetadata.model_validate(x) for x in dataset_metadata]
    dataset_metadata = [x for x in dataset_metadata if x.experiment_id == experiment_id]
    if len(dataset_metadata) != 1:
        raise ValueError(
            f'expected 1 instance in dataset_metadata of exp id {experiment_id} but got {len(dataset_metadata)}')
    experiment_meta = dataset_metadata[0]

    inverse_warp = ants.image_read(
        str('data' / experiment_meta.ls_to_template_inverse_warp_path)).numpy()
    affine = ants.read_transform(
        str('data' / experiment_meta.ls_to_template_affine_matrix_path)).parameters

    root = zarr.create_group(store=output_path)
    zarr.create_array(store=output_path,
                      name='coordinateTransformations/ls_to_template_SyN_1InverseWarp',
                      data=inverse_warp)
    zarr.create_array(store=output_path,
                      name='coordinateTransformations/ls_to_template_SyN_0GenericAffine',
                      data=affine.reshape((3, 4)))

    input_coordinate_system = {
        "name": "light_sheet_raw",
        "axes": [
            {
                "name": x.direction.value,
                "type": "space",
                "unit": "micrometer"
            }
            for x in sorted(experiment_meta.axes, key=lambda x: x.dimension)]
    }

    input_to_ls_transform = _get_input_space_to_light_sheet_transform(
        experiment_meta=experiment_meta,
        light_sheet_template_path=light_sheet_template_path,
        input_coordinate_system=input_coordinate_system,
        template_resolution=template_resolution

    )

    root.attrs['coordinateSystems'] = [
        input_coordinate_system,
        LIGHT_SHEET_COORDINATE_SYSTEM
    ]

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
