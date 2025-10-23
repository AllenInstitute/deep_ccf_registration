"""
This script applies the transforms to a given slice and plots the resulting mapping
template points
"""
import json
import os
import sys
from pathlib import Path
import ants
import click
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from deep_ccf_registration.datasets.slice_dataset import AcquisitionDirection, SliceDataset, \
    SliceOrientation, SubjectMetadata
from deep_ccf_registration.utils.utils import visualize_alignment
from loguru import logger

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)

def _calc_dice_metric(input_slice: np.ndarray, template_on_input: np.ndarray):
    input_slice_mask = (input_slice != 0).astype(int)
    template_on_input_mask = (template_on_input != 0).astype(int)

    # Calculate Dice coefficient
    intersection = np.sum(input_slice_mask * template_on_input_mask)
    sum_masks = np.sum(input_slice_mask) + np.sum(template_on_input_mask)

    # Avoid division by zero
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / sum_masks
    return dice

@click.command()
@click.option('--subject-id', required=True)
@click.option('--dataset-meta-path', type=click.Path(path_type=Path))
@click.option('--light-sheet-template-path', type=click.Path(path_type=Path),
              default='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz')
@click.option('--output-dir', type=click.Path(path_type=Path),
              default='/results/')
def main(subject_id: str,
         dataset_meta_path: Path,
         light_sheet_template_path: Path,
         output_dir: Path,
         ):
    with open(dataset_meta_path) as f:
        dataset_meta = json.load(f)
    dataset_meta: list[SubjectMetadata] = [SubjectMetadata.model_validate(x) for x in dataset_meta]

    subject_meta: SubjectMetadata = [x for x in dataset_meta if x.subject_id == subject_id][0]

    sagittal_axis = [i for i in range(len(subject_meta.axes)) if
                     subject_meta.axes[i].direction in (AcquisitionDirection.LEFT_TO_RIGHT,
                                                        AcquisitionDirection.RIGHT_TO_LEFT)]
    sagittal_axis = subject_meta.axes[sagittal_axis[0]]

    slice_index = int(subject_meta.registered_shape[sagittal_axis.dimension] / 2)

    ls_template = ants.image_read(str(light_sheet_template_path))

    slice_dataset = SliceDataset(
        ls_template=ls_template,
        dataset_meta=[subject_meta],
        orientation=SliceOrientation.SAGITTAL,
        mode='inference',
        patch_size=None
    )

    logger.info('forward transform')
    input_slice, output_points, dataset_idx, slice_idx = slice_dataset[slice_index]

    logger.info('generating plot')

    template_on_input = map_coordinates(
        input=ls_template.numpy(),
        coordinates=output_points.reshape((-1, 3)).T
    )

    dice_metric = _calc_dice_metric(
        input_slice=input_slice,
        template_on_input=template_on_input.reshape(input_slice.shape)
    )

    fig = visualize_alignment(
        input_slice=input_slice,
        template_on_input=template_on_input
    )
    fig.savefig(f'{output_dir}/{subject_meta.subject_id}.png')

    with open(f'{output_dir}/dice_metric.json', 'w') as f:
        f.write(json.dumps({'dice_metric': dice_metric}, indent=2))

if __name__ == '__main__':
    main()
