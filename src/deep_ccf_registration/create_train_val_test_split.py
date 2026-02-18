import json
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from deep_ccf_registration.datasets.aquisition_meta import AcquisitionDirection
from deep_ccf_registration.metadata import SubjectMetadata, SliceOrientation, TissueBoundingBoxes


def split_train_val_test(
    subject_metadata: list[SubjectMetadata],
    train_split: float,
    seed: int = 1234
) -> tuple[list[SubjectMetadata], list[SubjectMetadata], list[SubjectMetadata]]:
    """
    Parameters
    ----------
    subject_metadata: List of all subject metadata
    train_split: Fraction of data for training
    seed: Random seed for splitting

    Return
    --------
    Tuple of (train_metadata, val_metadata, test_metadata)

    Note: test_split = 1 - train_split - val_split
    """
    val_test_split = 1 - train_split
    val_split, test_split = val_test_split / 2, val_test_split / 2

    # Use random split
    logger.info(
        f"Using random train/val/test split: {train_split:.1%} train, "
        f"{val_split:.1%} val, {test_split:.1%} test"
    )

    shuffled_metadata = subject_metadata.copy()
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(len(subject_metadata))
    rng.shuffle(indices)
    shuffled_metadata = [shuffled_metadata[i] for i in indices]

    # Split
    n_train = int(len(shuffled_metadata) * train_split)
    n_val = int(len(shuffled_metadata) * val_split)

    train_subjects = shuffled_metadata[:n_train]
    val_subjects = shuffled_metadata[n_train:n_train + n_val]
    test_subjects = shuffled_metadata[n_train + n_val:]

    # Validation
    if len(train_subjects) == 0:
        raise ValueError("Training set is empty!")
    if len(val_subjects) == 0:
        raise ValueError("Validation set is empty!")
    if len(test_subjects) == 0:
        raise ValueError("Test set is empty!")

    return train_subjects, val_subjects, test_subjects

def sample_slices(
    subjects: list[SubjectMetadata],
    tissue_bounding_boxes_path: Path,
    orientations: list[SliceOrientation],
    seed: int = 1234,
    sample_fraction: float = 0.25
):
    samples = []
    rng = np.random.default_rng(seed=seed)
    for subject in tqdm(subjects):
        for orientation in orientations:
            subject_bboxes = pd.read_parquet(
                tissue_bounding_boxes_path / f'subject_id={subject.subject_id}')
            valid_indices = [x for x in subject_bboxes['index'].astype(int).tolist() if x is not None]
            sampled_indices = rng.choice(valid_indices, size=int(len(valid_indices) * sample_fraction))
            subject_samples = np.array(list(zip(
                [subject.subject_id]*len(sampled_indices),
                sampled_indices.tolist(),
                [orientation.value]*len(sampled_indices)
            )))
            samples.append(subject_samples)
    samples = np.concat(samples)
    return samples

@click.command()
@click.option(
    "--subject-metadata-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to subject metadata",
)
@click.option(
    "--tissue-bboxes-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to tissue bboxes",
)
@click.option(
    "--out-dir",
    type=click.Path(exists=True, path_type=Path, writable=True),
    required=True,
    help="Path to write output",
)
@click.option(
    "--train-frac",
    type=float,
    default=0.9,
)
@click.option(
    "--seed",
    type=int,
    default=1234,
)
@click.option(
    "--subject-sample-fraction",
    type=float,
    default=0.25,
    help='Fraction of slices to sample from each subject'
)
@click.option(
    "--orientations",
    type=str,
    default='sagittal',
    help='Comma separated list of orientations to sample'
)
def main(
        subject_metadata_path: Path,
        tissue_bboxes_path: Path,
        train_frac: float,
        seed: int,
        orientations: str,
        out_dir: Path,
        subject_sample_fraction: float
):
    with open(subject_metadata_path) as f:
        subject_metadata = json.load(f)
    orientations = [SliceOrientation(x) for x in orientations.split(',')]

    subject_metadata = [SubjectMetadata(**x) for x in subject_metadata]
    train_subjects, val_subjects, test_subjects = split_train_val_test(
        subject_metadata=subject_metadata,
        train_split=train_frac,
        seed=seed
    )
    logger.info('getting train samples')
    train_samples = sample_slices(
        subjects=train_subjects,
        tissue_bounding_boxes_path=tissue_bboxes_path,
        seed=seed,
        orientations=orientations,
        sample_fraction=subject_sample_fraction,
    )
    logger.info('getting val samples')
    val_samples = sample_slices(
        subjects=val_subjects,
        tissue_bounding_boxes_path=tissue_bboxes_path,
        seed=seed,
        orientations=orientations,
        sample_fraction=subject_sample_fraction,
    )
    logger.info('getting test samples')
    test_samples = sample_slices(
        subjects=test_subjects,
        tissue_bounding_boxes_path=tissue_bboxes_path,
        seed=seed,
        orientations=orientations,
        sample_fraction=subject_sample_fraction,
    )

    logger.info('writing train')
    np.save(out_dir / 'train.npy', train_samples)

    logger.info('writing val')
    np.save(out_dir / 'val.npy', val_samples)

    logger.info('writing test')
    np.save(out_dir / 'test.npy', test_samples)

if __name__ == '__main__':
    main()
