import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tensorstore
import click
from aind_smartspim_transform_utils.CoordinateTransform import CoordinateTransform
from loguru import logger
from tqdm import tqdm


from deep_ccf_registration.metadata import SubjectMetadata, AcquisitionAxis
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)

def get_points(shape: tuple[int, ...], axes: list[AcquisitionAxis]) -> pd.DataFrame:
    axes = sorted(axes, key=lambda x: x.dimension)
    indices = np.indices(shape).reshape(len(shape), -1).T
    columns = [x.name.value.lower() for x in axes]
    return pd.DataFrame(indices, columns=columns)

def create_point_map(
    full_res_path: str,
    registered_volume_path: str,
    dataset_meta_path: Path,
    subject_id: str,
    warp_path: Path,
    affine_path: Path,
    acquisition_path: Path,
    ls_template_path: Path,
    output_zarr_path: Path,
    batch_size: int,
) -> None:
    full_res = tensorstore.open(
        spec={
            'driver': 'auto',
            'kvstore': create_kvstore(
                path=full_res_path,
                aws_credentials_method="anonymous"
            )
        },
        read=True
    ).result()

    registered_volume = tensorstore.open(
        spec={
            'driver': 'auto',
            'kvstore': create_kvstore(
                path=registered_volume_path,
                aws_credentials_method="anonymous"
            )
        },
        read=True
    ).result()

    with open(dataset_meta_path) as f:
        meta = json.load(f)

    meta = [SubjectMetadata.model_validate(x) for x in meta]
    meta = [x for x in meta if x.subject_id == subject_id][0]
    points = get_points(shape=registered_volume.shape[2:], axes=meta.axes)
    points = points * 2**3  # scale to full resolution indices

    with open(acquisition_path) as f:
        acquisition = json.load(f)

    coord_transform = CoordinateTransform(
        name='smartspim_lca',
        dataset_transforms={
            'points_to_ccf': [
                str(affine_path),
                str(warp_path),
            ]
        },
        acquisition=acquisition,
        image_metadata={'shape': full_res.shape[2:]},
        ls_template_path=str(ls_template_path)
    )
    batches = [points.iloc[i:i + batch_size] for i in range(0, len(points), batch_size)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        ls_template_points = list(tqdm(
            executor.map(lambda b: coord_transform.forward_transform(b, to_ccf=False, to_index_space=False), batches),
            total=len(batches), desc='forward_transform'
        ))
    ls_template_points = pd.concat(ls_template_points, ignore_index=True)
    ls_template_points = ls_template_points.values

    ls_template_points = ls_template_points.reshape((*registered_volume.shape, 3))

    spec = registered_volume.spec().to_json()
    spec['dtype'] = 'float32'

    output_zarr_path_str = str(output_zarr_path)
    spec['kvstore'] = create_kvstore(path=output_zarr_path_str)
    spec['metadata']['dtype'] = '<f4'
    spec['metadata']['chunks'] = [*spec['metadata']['chunks'], 3]
    spec['metadata']['shape'] = [*spec['metadata']['shape'], 3]
    spec['transform']['input_exclusive_max'] = [*spec['transform']['input_exclusive_max'], [3]]
    spec['transform']['input_inclusive_min'] = [*spec['transform']['input_inclusive_min'], 0]
    ls_template_points_ts = tensorstore.open(spec, create=True).result()
    ls_template_points_ts[:] = ls_template_points


@click.command()
@click.option(
    '--zarr-base-path',
    default='s3://aind-open-data/SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr',
    show_default=True,
    help='Base OME-Zarr path shared by the full-resolution (/0) and registered (/3) volumes.'
)
@click.option(
    '--dataset-meta-path',
    type=click.Path(path_type=Path),
    default=Path('/Users/adam.amster/smartspim-registration/dataset_meta-test.json'),
    show_default=True,
    help='Path to the dataset metadata JSON file.'
)
@click.option(
    '--subject-id',
    default='806624',
    show_default=True,
    help='Subject identifier to select from the metadata file.'
)
@click.option(
    '--warp-path',
    type=click.Path(path_type=Path),
    default=Path('/Users/adam.amster/smartspim-registration/806624_ls_to_template_SyN_1InverseWarp.nii.gz'),
    show_default=True,
    help='Path to the inverse warp transform file.'
)
@click.option(
    '--affine-path',
    type=click.Path(path_type=Path),
    default=Path('/Users/adam.amster/smartspim-registration/806624_ls_to_template_SyN_0GenericAffine.mat'),
    show_default=True,
    help='Path to the affine transform file.'
)
@click.option(
    '--acquisition-path',
    type=click.Path(path_type=Path),
    default=Path('/Users/adam.amster/smartspim-registration/806624_acquisition.json'),
    show_default=True,
    help='Path to the acquisition JSON metadata.'
)
@click.option(
    '--ls-template-path',
    type=click.Path(path_type=Path),
    default=Path('/Users/adam.amster/.transform_utils/transform_utils/smartspim_lca/template/smartspim_lca_template_25.nii.gz'),
    show_default=True,
    help='Path to the light-sheet template volume.'
)
@click.option(
    '--output-zarr-path',
    type=click.Path(path_type=Path),
    default=Path('/tmp/806624_template_points.zarr'),
    show_default=True,
    help='Destination path for the generated template point tensorstore.'
)
@click.option(
    '--batch-size',
    default=10_000_000,
    show_default=True,
    help='Number of points transformed per batch.'
)
def main(
    zarr_base_path: str,
    dataset_meta_path: Path,
    subject_id: str,
    warp_path: Path,
    affine_path: Path,
    acquisition_path: Path,
    ls_template_path: Path,
    output_zarr_path: Path,
    batch_size: int,
) -> None:
    """Generate a template point map aligned to CCF space."""
    zarr_base_path = zarr_base_path.rstrip('/')
    full_res_path = f"{zarr_base_path}/0"
    registered_volume_path = f"{zarr_base_path}/3"
    create_point_map(
        full_res_path=full_res_path,
        registered_volume_path=registered_volume_path,
        dataset_meta_path=dataset_meta_path,
        subject_id=subject_id,
        warp_path=warp_path,
        affine_path=affine_path,
        acquisition_path=acquisition_path,
        ls_template_path=ls_template_path,
        output_zarr_path=output_zarr_path,
        batch_size=batch_size,
    )


if __name__ == '__main__':
    main()
