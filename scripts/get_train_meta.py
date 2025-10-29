from datetime import datetime
import json
from pathlib import Path
from typing import Any

import tensorstore
from loguru import logger
from tqdm import tqdm
from deep_ccf_registration.datasets.slice_dataset import SubjectMetadata, AcquisitionAxis
import click

from deep_ccf_registration.utils.tensorstore_utils import create_kvstore


def _get_subject_id_registration_channel_map(smartspim_raw_dirs: list[Path]) -> dict[
    str, list[str]]:
    subject_id_registration_channel_map: dict[str, list[str]] = {}
    for smartspim_raw_dir in tqdm(smartspim_raw_dirs,
                                  desc='Getting subject id registration channel map'):
        subject_id = smartspim_raw_dir.name.split('_')[1]

        if (smartspim_raw_dir / 'SPIM' / 'derivatives' / 'processing_manifest.json').exists():
            processing_manifest_path = smartspim_raw_dir / 'SPIM' / 'derivatives' / 'processing_manifest.json'
        elif (smartspim_raw_dir / 'derivatives' / 'processing_manifest.json').exists():
            processing_manifest_path = smartspim_raw_dir / 'derivatives' / 'processing_manifest.json'
        else:
            logger.warning(f'subject id {subject_id} cannot find processing_manifest. skipping')
            continue

        with open(processing_manifest_path) as f:
            processing_manifest: dict[str, Any] = json.load(f)

        try:
            registration_channels: list[str] = \
            processing_manifest['pipeline_processing']['registration']['channels']
        except KeyError:
            logger.warning(f'subject id {subject_id} does not have registration meta. Skipping...')
            continue

        if len(registration_channels) == 0:
            logger.warning(
                f'subject id {subject_id} has {len(registration_channels)} registration channels. skipping')

        subject_id_registration_channel_map[subject_id] = registration_channels

    return subject_id_registration_channel_map


@click.command()
@click.option('--aind-open-data-dir', type=click.Path(dir_okay=True, path_type=Path),
              help='Directory to aind-open-data')
@click.option('--output-path', type=click.Path(file_okay=True, writable=True, path_type=Path),
              help='Path to write json')
@click.option('--input-space-midline-path', type=click.Path(file_okay=True, writable=True, path_type=Path),
              help='Path to input space midline found via `get_input_space_midline.py`',
              default='/data/input_space_midline/midline.json')
def main(
    output_path: Path,
    aind_open_data_dir: Path,
    input_space_midline_path: Path
):
    smartspim_dirs = list(aind_open_data_dir.glob(pattern='SmartSPIM_*'))
    smartspim_stitched_dirs = [x for x in smartspim_dirs if 'stitched' in x.name]
    smartspim_raw_dirs = [x for x in smartspim_dirs if 'stitched' not in x.name]

    subject_id_channel_map = _get_subject_id_registration_channel_map(
        smartspim_raw_dirs=smartspim_raw_dirs
    )

    experiments: list[SubjectMetadata] = []
    for subject_id, channels in tqdm(subject_id_channel_map.items(), desc='Fetching subject meta'):
        subject_stitched_dirs = [x for x in smartspim_stitched_dirs if
                                 x.name.startswith(f'SmartSPIM_{subject_id}')]
        if len(subject_stitched_dirs) == 0:
            logger.warning(
                f'subject id {subject_id} has {len(subject_stitched_dirs)} stitched dirs. skipping...')
            continue
        stitched_dir_date_map = {
            datetime.strptime('_'.join(x.name.split('_')[-2:]), '%Y-%m-%d_%H-%M-%S'): x
            for x in subject_stitched_dirs
        }
        most_recent_stitch = sorted(stitched_dir_date_map)[-1]
        smartspim_stitched_dir = stitched_dir_date_map[most_recent_stitch]

        acquisition_meta_path = smartspim_stitched_dir / 'acquisition.json'
        if not acquisition_meta_path.exists():
            continue

        with open(acquisition_meta_path) as f:
            acquisition_meta = json.load(f)

        for channel in channels:
            image_atlas_alignment_dir = smartspim_stitched_dir / 'image_atlas_alignment' / f'{channel}'
            stitched_volume_path = smartspim_stitched_dir / 'image_tile_fusing' / 'OMEZarr' / f'{channel}.zarr'

            # Using the s3 uri is faster than using s3fs
            stitched_volume_path = f's3://aind-open-data/{str(stitched_volume_path).replace("/data/aind_open_data/", "")}'

            if not (image_atlas_alignment_dir / 'metadata' / 'processing.json').exists():
                logger.warning('no processing.json. skipping')
                continue

            with open(image_atlas_alignment_dir / 'metadata' / 'processing.json') as f:
                registration_processing_meta = json.load(f)

            try:
                alignment_step = \
                [x for x in registration_processing_meta['processing_pipeline']['data_processes'] if
                 x['name'] == 'Image atlas alignment'][0]
                alignment_downsample_factor = int(Path(alignment_step['input_location']).name)
            except KeyError:
                logger.warning('registration_processing_meta invalid. skipping...')
                continue

            try:
                volume = tensorstore.open(
                    spec={
                        'driver': 'auto',
                        'kvstore': create_kvstore(
                            path=f'{stitched_volume_path}/{alignment_downsample_factor}',
                            aws_credentials_method="anonymous"
                        )
                    },
                    read=True
                ).result()
            except ValueError:
                logger.warning(
                    f'subject id {subject_id} channel {channel} stitched ome zarr does not exist. Skipping')
                continue

            ls_to_template_inverse_warp_niftii = image_atlas_alignment_dir / 'ls_to_template_SyN_1InverseWarp.nii.gz'
            if not ls_to_template_inverse_warp_niftii.exists():
                continue

            # ordered x, y, z
            resolution = acquisition_meta["tiles"][0]["coordinate_transformations"][1]["scale"]
            resolution_ordering = {'X': 0, 'Y': 1, 'Z': 2}

            axes: list[AcquisitionAxis] = []
            for axis in acquisition_meta['axes']:
                axes.append(
                    AcquisitionAxis(
                        **axis,
                        resolution=resolution[resolution_ordering[axis["name"]]]
                    )
                )

            ls_to_template_affine_matrix_path = image_atlas_alignment_dir / 'ls_to_template_SyN_0GenericAffine.mat'
            ls_to_template_inverse_warp_path = image_atlas_alignment_dir / 'transforms.zarr/coordinateTransformations/ls_to_template_SyN_1InverseWarp'

            assert ls_to_template_affine_matrix_path.exists()

            # use the zarr array instead, since much faster
            ls_to_template_inverse_warp_path = f's3://marmot-development-802451596237-us-west-2/transforms/{str(ls_to_template_inverse_warp_path).replace("/data/aind_open_data/", "")}'

            try:
                # try to open zarr array to make sure it exists
                tensorstore.open(
                    spec={
                        'driver': 'zarr3',
                        'kvstore': create_kvstore(
                            path=str(ls_to_template_inverse_warp_path),
                            aws_credentials_method='ecs'
                        )
                    },
                    read=True
                ).result()
            except ValueError:
                logger.warning(f'{ls_to_template_inverse_warp_path} does not exist')
                continue

            with open(input_space_midline_path) as f:
                midlines = json.load(f)
            subject_midline = [x for x in midlines if x['subject_id'] == subject_id][0]['midline_mean']
            # convert to downsampled index
            subject_midline = int(subject_midline / 2 ** alignment_downsample_factor)

            experiment_meta = SubjectMetadata(
                subject_id=subject_id,
                stitched_volume_path=stitched_volume_path,
                axes=axes,
                registered_shape=volume.shape[2:],  # shape is T, C, ...
                registration_downsample=alignment_downsample_factor,
                ls_to_template_affine_matrix_path=ls_to_template_affine_matrix_path,
                ls_to_template_inverse_warp_path=ls_to_template_inverse_warp_path,
                ls_to_template_inverse_warp_path_original=ls_to_template_inverse_warp_niftii,
                sagittal_midline=subject_midline
            )
            experiments.append(experiment_meta)

    logger.info(f'{len(experiments)} experiments')

    with open(output_path, 'w') as f:
        f.write(json.dumps([json.loads(x.model_dump_json()) for x in experiments], indent=2))


if __name__ == '__main__':
    main()
