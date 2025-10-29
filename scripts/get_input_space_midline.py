import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tensorstore
from aind_smartspim_transform_utils.CoordinateTransform import CoordinateTransform

from deep_ccf_registration.datasets.slice_dataset import SubjectMetadata


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

@click.command()
@click.option('--subject-id', required=True)
@click.option('--dataset-meta-path', type=click.Path(path_type=Path), default='/data/smartspim_dataset/subject_metadata.json')
@click.option('--ccf-midline-path', type=click.Path(path_type=Path), default='/data/third_ventricle/third_ventricle.csv')
@click.option('--output-dir', type=click.Path(path_type=Path), default='/results')
def main(subject_id: str, dataset_meta_path: Path, ccf_midline_path: Path, output_dir: Path):
    with open(dataset_meta_path) as f:
        dataset_meta = json.load(f)
    dataset_meta: list[SubjectMetadata] = [SubjectMetadata.model_validate(x) for x in dataset_meta]

    subject_meta: SubjectMetadata = [x for x in dataset_meta if x.subject_id == subject_id][0]

    ccf_midline = pd.read_csv(ccf_midline_path)

    # sample some of the points
    rng = np.random.default_rng(1234)
    idxs = np.arange(ccf_midline.shape[0])
    rng.shuffle(idxs)
    idxs = idxs[:int(.1*len(idxs))]
    ccf_midline = ccf_midline.iloc[idxs]

    input_points_from_template = reverse_transform_points(
        stitched_path=subject_meta.stitched_volume_path,
        acquisition_path=subject_meta.stitched_volume_path.parent.parent.parent / 'acquisition.json',
        affine_path=subject_meta.ls_to_template_affine_matrix_path,
        warp_path=subject_meta.ls_to_template_inverse_warp_path_original.parent / 'ls_to_template_SyN_1Warp.nii.gz',
        template_points=ccf_midline
    )

    input_points_from_template.to_csv(output_dir / 'input_space_midline.csv', index=False)

    with open(output_dir / 'midline.json', 'w') as f:
        f.write(json.dumps({
            'subject_id': subject_id,
            'midline_mean': int(input_points_from_template['ML'].mean())
        }))

if __name__ == '__main__':
    main()
