import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorstore
import click
from aind_smartspim_transform_utils.CoordinateTransform import CoordinateTransform
from loguru import logger
from tqdm import tqdm


from deep_ccf_registration.metadata import AcquisitionAxis
from deep_ccf_registration.utils.tensorstore_utils import create_kvstore

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)

def iter_point_batches(
    shape: tuple[int, ...],
    axes: list[AcquisitionAxis],
    batch_size: int,
    scale: int,
):
    axes = sorted(axes, key=lambda x: x.dimension)
    columns = [x.name.value.lower() for x in axes]
    total_points = int(np.prod(shape))
    for start_idx in range(0, total_points, batch_size):
        end_idx = min(start_idx + batch_size, total_points)
        flat_indices = np.arange(start_idx, end_idx, dtype=np.int64)
        coords = np.column_stack(np.unravel_index(flat_indices, shape))
        if scale != 1:
            coords = coords * scale
        yield start_idx, pd.DataFrame(coords, columns=columns)

def create_point_map(
    full_res_path: str,
    registered_volume_path: str,
    warp_path: Path,
    affine_path: Path,
    acquisition_path: Path,
    ls_template_path: Path,
    output_zarr_path: str,
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

    spec = registered_volume.spec().to_json()
    spec['dtype'] = 'float32'

    spec['kvstore'] = create_kvstore(path=str(output_zarr_path))
    spec['metadata']['dtype'] = '<f4'
    spec['metadata']['chunks'] = [*spec['metadata']['chunks'], 3]
    spec['metadata']['shape'] = [*spec['metadata']['shape'], 3]
    spec['transform']['input_exclusive_max'] = [*spec['transform']['input_exclusive_max'], [3]]
    spec['transform']['input_inclusive_min'] = [*spec['transform']['input_inclusive_min'], 0]
    ls_template_points_ts = tensorstore.open(spec, create=True, open=True).result()

    tmp_warp_path = f'/results/{Path(warp_path).name}'
    logger.info(f'copying warp path from {warp_path} to {tmp_warp_path}')
    shutil.copy(warp_path, tmp_warp_path)

    with open(acquisition_path) as f:
        acquisition = json.load(f)

    resolution = acquisition["tiles"][0]["coordinate_transformations"][1]["scale"]
    resolution_ordering = {'X': 0, 'Y': 1, 'Z': 2}

    axes = [
        AcquisitionAxis(
            **x,
            resolution=resolution[resolution_ordering[x["name"]]]
        ) for x in acquisition['axes']]
    spatial_shape = registered_volume.shape[2:]
    if not spatial_shape:
        raise ValueError('registered volume must have spatial dimensions beyond the first two axes')
    total_points = int(np.prod(spatial_shape))
    point_batches = iter_point_batches(
        shape=spatial_shape,
        axes=axes,
        batch_size=batch_size,
        scale=2**3,
    )
    coord_transform = CoordinateTransform(
        name='smartspim_lca',
        dataset_transforms={
            'points_to_ccf': [
                str(affine_path),
                str(tmp_warp_path),
            ]
        },
        acquisition=acquisition,
        image_metadata={'shape': full_res.shape[2:]},
        ls_template_path=str(ls_template_path)
    )
    tmp_points_path = '/results/template_points.dat'
    total_batches = math.ceil(total_points / batch_size) if total_points else 0
    try:
        ls_template_points = np.memmap(
            tmp_points_path,
            dtype=np.float32,
            mode='w+',
            shape=(total_points, 3)
        )

        for start_idx, batch in tqdm(point_batches, total=total_batches, desc='forward_transform'):
            end_idx = start_idx + len(batch)
            transformed = coord_transform.forward_transform(batch, to_ccf=False, to_index_space=False)
            ls_template_points[start_idx:end_idx] = transformed.to_numpy(dtype=np.float32, copy=False)

        ls_template_points.flush()
        reshaped_points = ls_template_points.reshape((*registered_volume.shape, 3))
        ls_template_points_ts[:] = reshaped_points
    finally:
        if os.path.exists(tmp_points_path):
            os.remove(tmp_points_path)
        os.remove(tmp_warp_path)

@click.command()
@click.option(
    '--zarr-base-path',
    default='s3://aind-open-data/SmartSPIM_806624_2025-08-27_15-42-18_stitched_2025-08-29_22-47-08/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr',
    show_default=True,
    help='Base OME-Zarr path shared by the full-resolution (/0) and registered (/3) volumes.'
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
    default='/tmp/806624_template_points.zarr',
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
    warp_path: Path,
    affine_path: Path,
    acquisition_path: Path,
    ls_template_path: Path,
    output_zarr_path: str,
    batch_size: int,
) -> None:
    """Generate a template point map aligned to CCF space."""
    zarr_base_path = zarr_base_path.rstrip('/')
    full_res_path = f"{zarr_base_path}/0"
    registered_volume_path = f"{zarr_base_path}/3"
    create_point_map(
        full_res_path=full_res_path,
        registered_volume_path=registered_volume_path,
        warp_path=warp_path,
        affine_path=affine_path,
        acquisition_path=acquisition_path,
        ls_template_path=ls_template_path,
        output_zarr_path=output_zarr_path,
        batch_size=batch_size,
    )


if __name__ == '__main__':
    main()
