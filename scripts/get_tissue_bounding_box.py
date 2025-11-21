"""
This script gets a bounding box around tissue in each slice of a volume using the ccf annotations
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from functools import partial

import ants
import click
import cv2
import numpy as np
import pandas as pd
import tensorstore
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from loguru import logger
from tqdm import tqdm

from deep_ccf_registration.utils.tensorstore_utils import create_kvstore
from deep_ccf_registration.utils.transforms import transform_points_to_template_ants_space, \
    apply_transforms_to_points
from deep_ccf_registration.utils.utils import get_ccf_annotations

from deep_ccf_registration.datasets.slice_dataset import _create_coordinate_dataframe
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)


def compute_bbox(ccf_annotations_slice, padding):
    """Compute bounding box for a single slice of CCF annotations."""
    # Create binary mask of non-zero regions
    binary_mask = (ccf_annotations_slice != 0).astype(np.uint8)

    # Find bounding box
    coords = cv2.findNonZero(binary_mask)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # Add padding
        height, width = ccf_annotations_slice.shape

        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(width - x_padded, w + 2 * padding)
        h_padded = min(height - y_padded, h + 2 * padding)

        bbox = {
            'y': int(y_padded),
            'x': int(x_padded),
            'width': int(w_padded),
            'height': int(h_padded)
        }
    else:
        bbox = None

    return bbox


def process_batch(
    batch_info,
    experiment_meta,
    ls_template_parameters,
    ccf_annotations,
    slice_axis,
    height,
    width,
    padding
):
    """
    Worker function to process a batch of slices.

    Args:
        batch_info: Tuple of (batch_start, batch_end)
        experiment_meta: SubjectMetadata instance
        ls_template_parameters: Template parameters
        ccf_annotations: CCF annotations array
        slice_axis: Slice axis info
        height: Slice height
        width: Slice width
        padding: Padding amount for bounding box

    Returns:
        List of tuples: [(slice_idx, bbox), ...]
    """
    try:
        batch_start, batch_end = batch_info
        batch_slice_indices = range(batch_start, batch_end)

        # Open tensorstore objects in worker process
        volume = tensorstore.open(
            spec={
                'driver': 'auto',
                'kvstore': create_kvstore(
                    path=str(experiment_meta.stitched_volume_path) + '/3',
                    aws_credentials_method="anonymous"
                )
            },
            read=True
        ).result()

        warp = tensorstore.open(
            spec={
                'driver': 'zarr3',
                'kvstore': create_kvstore(
                    path=str(experiment_meta.ls_to_template_inverse_warp_path),
                    aws_credentials_method='ecs'
                )
            },
            read=True
        ).result()

        # Create point grids for all slices in this batch
        point_grids = []
        for slice_idx in batch_slice_indices:
            point_grid = _create_coordinate_dataframe(
                patch_height=height,
                patch_width=width,
                start_x=0,
                start_y=0,
                fixed_index_value=slice_idx,
                axes=experiment_meta.axes,
                slice_axis=slice_axis
            )
            point_grids.append(point_grid)

        # Concatenate all point grids for batch
        batch_point_grids = pd.concat(point_grids, ignore_index=True)

        # Transform all points in batch at once
        points = transform_points_to_template_ants_space(
            points=batch_point_grids,
            input_volume_shape=volume.shape[2:],
            acquisition_axes=experiment_meta.axes,
            ls_template_info=ls_template_parameters,
            registration_downsample=experiment_meta.registration_downsample
        )

        ls_template_points = apply_transforms_to_points(
            points=points,
            template_parameters=ls_template_parameters,
            affine_path=experiment_meta.ls_to_template_affine_matrix_path,
            warp=warp,
            crop_warp_to_bounding_box=True
        )

        # Split transformed points back per slice and compute bboxes
        batch_results = []
        points_per_slice = height * width

        for i, slice_idx in enumerate(batch_slice_indices):
            # Extract points for this slice
            start_idx = i * points_per_slice
            end_idx = start_idx + points_per_slice
            slice_points = ls_template_points[start_idx:end_idx]

            # Reshape to image dimensions
            output_points = slice_points.reshape((height, width, 3))

            # Convert to index space and get CCF annotations
            points_index_space = convert_from_ants_space(
                template_parameters=ls_template_parameters,
                physical_pts=output_points.reshape((-1, 3))
            )
            input_slice_ccf_annotations = get_ccf_annotations(
                ccf_annotations,
                points_index_space
            ).reshape(output_points.shape[:-1])

            # Compute bbox for this slice
            bbox = compute_bbox(input_slice_ccf_annotations, padding)
            batch_results.append((slice_idx, bbox))

        return batch_results

    except Exception as e:
        logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
        # Return None bboxes for failed batch
        return [(idx, None) for idx in range(batch_start, batch_end)]


@click.command()
@click.option('--subject-id', required=True)
@click.option('--dataset-meta-path', type=click.Path(path_type=Path),
              default='/data/smartspim_dataset/subject_metadata.json')
@click.option('--light-sheet-template-path', type=click.Path(path_type=Path),
              default='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz')
@click.option("--out-dir", type=click.Path(path_type=Path, dir_okay=True, writable=True),
              default='/results')
@click.option('--ccf-annotations-path', type=click.Path(path_type=Path, exists=True),
              default='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/ccf_annotation_to_template_moved_25.nii.gz')
@click.option('--padding', type=int, default=30, help='Padding around bounding box')
@click.option('--batch-size', type=int, default=10, help='Number of slices per batch')
@click.option('--num-workers', type=int, default=4,
              help='Number of worker processes (default: CPU count)')
def main(
    dataset_meta_path: Path,
    subject_id: str,
    light_sheet_template_path: Path,
    out_dir: Path,
    ccf_annotations_path: Path,
    padding: int,
    batch_size: int,
    num_workers: int
):
    # Setup output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(dataset_meta_path) as f:
        dataset_meta = json.load(f)
    dataset_meta = [SubjectMetadata.model_validate(x) for x in dataset_meta]
    subject_meta = [x for x in dataset_meta if x.subject_id == subject_id]

    if not subject_meta:
        logger.error(f"No metadata found for subject {subject_id}")
        return

    experiment_meta = subject_meta[0]

    # Load template
    ls_template = ants.image_read(str(light_sheet_template_path))
    ls_template_parameters = AntsImageParameters.from_ants_image(ls_template)

    # Open volume to get dimensions and slice count
    volume = tensorstore.open(
        spec={
            'driver': 'auto',
            'kvstore': create_kvstore(
                path=str(experiment_meta.stitched_volume_path) + '/3',
                aws_credentials_method="anonymous"
            )
        },
        read=True
    ).result()

    # Load CCF annotations
    ccf_annotations = ants.image_read(str(ccf_annotations_path)).numpy()

    # Get slice info
    slice_axis = experiment_meta.get_slice_axis(orientation=SliceOrientation.SAGITTAL)
    num_slices = volume.shape[slice_axis.dimension + 2]

    # Get slice dimensions (assuming all slices have same dimensions)
    volume_slice = [0, 0, slice(None), slice(None), slice(None)]
    volume_slice[slice_axis.dimension + 2] = 0
    height, width = volume[tuple(volume_slice)].shape

    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Create batch ranges
    batch_ranges = []
    for batch_start in range(0, num_slices, batch_size):
        batch_end = min(batch_start + batch_size, num_slices)
        batch_ranges.append((batch_start, batch_end))

    logger.info(f"Processing {num_slices} slices in {len(batch_ranges)} batches "
                f"(batch_size={batch_size}) with {num_workers} workers")

    # Create partial function with fixed arguments
    worker_fn = partial(
        process_batch,
        experiment_meta=experiment_meta,
        ls_template_parameters=ls_template_parameters,
        ccf_annotations=ccf_annotations,
        slice_axis=slice_axis,
        height=height,
        width=width,
        padding=padding
    )

    # Process batches in parallel
    with mp.Pool(processes=num_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(worker_fn, batch_ranges),
            total=len(batch_ranges),
            desc="Processing batches"
        ))

    # Flatten results and sort by slice index
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)

    all_results.sort(key=lambda x: x[0])
    all_bboxes = [bbox for _, bbox in all_results]

    # Save results
    output_path = out_dir / f'{subject_id}_tissue_bboxes.json'
    with open(output_path, 'w') as f:
        json.dump(all_bboxes, f, indent=2)

    logger.info(f"Saved {len(all_bboxes)} bounding boxes to {output_path}")
    logger.info(f"Slices with tissue: {sum(b is not None for b in all_bboxes)}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()