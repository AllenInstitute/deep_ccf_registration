import json

import multiprocessing
import os
import sys
from pathlib import Path

import ants
import click
import cv2
import numpy as np
import tensorstore
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from loguru import logger
from tqdm import tqdm

from deep_ccf_registration.utils.utils import get_ccf_annotations

from deep_ccf_registration.datasets.slice_dataset import SliceDataset, TrainMode
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)


@click.command()
@click.option('--subject-id', required=True)
@click.option('--dataset-meta-path', type=click.Path(path_type=Path), default='/data/smartspim_dataset/subject_metadata.json')
@click.option('--light-sheet-template-path', type=click.Path(path_type=Path),
              default='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz')
@click.option("--out-dir", type=click.Path(path_type=Path, dir_okay=True, writable=True), default='/results')
@click.option('--ccf-annotations-path', type=click.Path(path_type=Path, exists=True),
              default='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/ccf_annotation_to_template_moved_25.nii.gz')
def main(
    dataset_meta_path: Path,
    subject_id,
    light_sheet_template_path: Path,
    out_dir: Path,
    ccf_annotations_path: Path,
    padding: int = 10
):
    with open(dataset_meta_path) as f:
        dataset_meta = json.load(f)
    dataset_meta = [SubjectMetadata.model_validate(x) for x in dataset_meta]
    subject_meta = [x for x in dataset_meta if x.subject_id == subject_id]

    ls_template = ants.image_read(
        str(light_sheet_template_path))
    ls_template_parameters = AntsImageParameters.from_ants_image(ls_template)

    slice_dataset = SliceDataset(
        dataset_meta=subject_meta,
        orientation=SliceOrientation.SAGITTAL,
        mode=TrainMode.TEST,
        ls_template_parameters=ls_template_parameters,
        patch_size=None,
    )

    ccf_annotations = ants.image_read(str(ccf_annotations_path)).numpy()

    bboxes = []
    for slice_idx in tqdm(range(len(slice_dataset))):
        input_slice, output_points, dataset_idx, slice_idx, _, _, _ = slice_dataset[slice_idx]

        points_index_space = convert_from_ants_space(template_parameters=ls_template_parameters, physical_pts=output_points.reshape((-1, 3)))
        input_slice_ccf_annotations = get_ccf_annotations(ccf_annotations, points_index_space).reshape(
            output_points.shape[:-1]
        )

        # Create binary mask of non-zero regions
        binary_mask = (input_slice_ccf_annotations != 0).astype(np.uint8)

        # Find bounding box
        coords = cv2.findNonZero(binary_mask)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Add padding
            height, width = input_slice_ccf_annotations.shape

            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(width - x_padded, w + 2 * padding)
            h_padded = min(height - y_padded, h + 2 * padding)

            bbox = {
                'y': y_padded,
                'x': x_padded,
                'width': w_padded,
                'height': h_padded
            }
        else:
            bbox = None

        bboxes.append(bbox)

    with open(out_dir / 'tissue_bboxes.json', 'w') as f:
        f.write(json.dumps(bboxes, indent=2))




if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
