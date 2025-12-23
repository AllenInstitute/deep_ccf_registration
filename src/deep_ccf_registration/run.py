import multiprocessing
import os
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from typing import Optional

import click
import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from monai.networks.nets import UNet

from torch.utils.data import DataLoader
from contextlib import nullcontext
from loguru import logger
import json
import ants
import random

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.slice_dataset_cache import (
    PatchSample,
    SliceDatasetCache,
    ShardedMultiDatasetCache,
    ShuffledBatchIterator,
    collate_patch_samples, )
from deep_ccf_registration.datasets.transforms import build_transform, TemplatePointsNormalization, TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, TissueBoundingBoxes
from deep_ccf_registration.train import train


def _identity_collate(batch):
    """Pass through batch as-is (list of PatchSample)."""
    return batch


def _clone_patch_sample(sample: PatchSample) -> PatchSample:
    """Create a detached copy of a PatchSample for safe reuse."""
    return PatchSample(
        slice_idx=sample.slice_idx,
        start_y=sample.start_y,
        start_x=sample.start_x,
        data=np.array(sample.data, copy=True),
        template_points=np.array(sample.template_points, copy=True),
        dataset_idx=sample.dataset_idx,
        worker_id=sample.worker_id,
        orientation=getattr(sample, "orientation", ""),
        subject_id=getattr(sample, "subject_id", ""),
    )


class RepeatSinglePatchIterator:
    """Yield batches made from a single cached patch (debug helper)."""

    def __init__(self, base_iterator, batch_size: int):
        self._base_iterator = base_iterator
        self._batch_size = batch_size

    def __iter__(self):
        sample: Optional[PatchSample] = None

        for batch in self._base_iterator:
            if not batch:
                continue
            sample = _clone_patch_sample(batch[0])
            break

        if sample is None:
            return

        while True:
            yield [_clone_patch_sample(sample=sample) for _ in range(self._batch_size)]


class CollatedBatchIterator:
    """Wrap an iterator of PatchSample batches and emit collated tensors."""

    def __init__(self, base_iterator, patch_size: int):
        self._base_iterator = base_iterator
        self._patch_size = patch_size

    def __iter__(self):
        for batch in self._base_iterator:
            if not batch:
                continue
            yield collate_patch_samples(batch, patch_size=self._patch_size)


def create_dataloader(
    metadata: list[SubjectMetadata],
    tissue_bboxes: TissueBoundingBoxes,
    config: TrainConfig,
    batch_size: int,
    num_workers: int,
    ls_template_parameters: TemplateParameters,
    template_points_normalizer: Optional[TemplatePointsNormalization],
    buffer_batches: int = 8,
):
    """
    Create a dataloader using SliceDatasetCache with worker sharding and batch shuffling.

    Returns a ShuffledBatchIterator that yields collated batch dicts.
    """
    datasets = []
    for meta in metadata:
        transform = build_transform(
            config=config,
        )
        datasets.append(
            SliceDatasetCache(
                dataset_meta=meta,
                tissue_bboxes=tissue_bboxes.bounding_boxes[meta.subject_id],
                sample_fraction=0.1,
                orientation=config.orientation,
                tensorstore_aws_credentials_method=config.tensorstore_aws_credentials_method,
                registration_downsample_factor=config.registration_downsample_factor,
                patch_size=config.patch_size[0],
                max_chunks_per_dataset=1,
                transform=transform,
                template_points_normalizer=template_points_normalizer,
                template_parameters=TemplateParameters(
                    origin=ls_template_parameters.origin,
                    scale=ls_template_parameters.scale,
                    direction=ls_template_parameters.direction,
                    shape=ls_template_parameters.shape
                )
            )
        )

    sharded_dataset = ShardedMultiDatasetCache(datasets=datasets)

    dataloader = DataLoader(
        dataset=sharded_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        pin_memory=False,
    )

    iterator = ShuffledBatchIterator(
        dataloader=dataloader,
        batch_size=batch_size,
        buffer_batches=buffer_batches,
    )

    if config.debug:
        logger.warning("Debug mode: repeating a single cached patch for all batches")
        iterator = RepeatSinglePatchIterator(iterator, batch_size)

    return CollatedBatchIterator(iterator, patch_size=config.patch_size[0])

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)


def _get_git_commit_from_package(package_name="deep-ccf-registration"):
    """Extract git commit from installed package"""
    dist = distribution(package_name)

    direct_url_text = dist.read_text("direct_url.json")

    direct_url = json.loads(direct_url_text)
    commit = direct_url["vcs_info"]["commit_id"]
    url = direct_url["url"]
    return commit, url

@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration JSON file",
)
def main(config_path: Path):
    """Train a model to predict points in light sheet template space given a light sheet image."""

    with open(config_path) as f:
        config = json.load(f)
    config = TrainConfig.model_validate(config)

    if config.log_file:
        logger.add(sink=config.logging.log_file, rotation="500 MB", level="INFO")

    logger.info("=" * 60)
    logger.info("Starting training run")
    logger.info("=" * 60)

    torch.manual_seed(seed=config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=config.seed)
    logger.info(f"Random seed set to: {config.seed}")

    if config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = config.device
    logger.info(f"Using device: {config.device}")

    # Setup mixed precision
    if config.mixed_precision and device == "cuda":
        autocast_context = torch.cuda.amp.autocast()
        logger.info("Mixed precision training enabled")
    else:
        autocast_context = nullcontext()
        if config.mixed_precision and device != "cuda":
            logger.warning("Mixed precision only supported on CUDA, disabling")

    # Load light sheet template
    logger.info(f"Loading light sheet template from: {config.ls_template_path}")
    ls_template = ants.image_read(filename=str(config.ls_template_path))
    ls_template_ants_parameters = AntsImageParameters.from_ants_image(image=ls_template)
    ls_template_parameters = TemplateParameters(
        origin=ls_template_ants_parameters.origin,
        scale=ls_template_ants_parameters.scale,
        direction=ls_template_ants_parameters.direction,
        shape=ls_template.shape,
    )
    del ls_template

    # Load dataset metadata
    logger.info(f"Loading dataset metadata from: {config.dataset_meta_path}")
    with open(file=config.dataset_meta_path, mode='r') as f:
        subject_metadata_dicts = json.load(fp=f)
    subject_metadata = [SubjectMetadata.model_validate(x) for x in subject_metadata_dicts]
    logger.info(f"Total subjects loaded: {len(subject_metadata)}")

    # Split into train/val
    train_metadata, val_metadata, test_metadata = split_train_val_test(
        subject_metadata=subject_metadata,
        train_split=config.train_val_split,
        val_split=(1 - config.train_val_split) / 2,
    )

    logger.info(f"Train subjects: {len(train_metadata)}")
    logger.info(f"Val subjects: {len(val_metadata)}")
    logger.info(f"Test subjects: {len(test_metadata)}")

    logger.info('loading ccf annotations volume')
    ccf_annotations = ants.image_read(str(config.ccf_annotations_path)).numpy()

    # write ccf_annotations to memmap. this avoids RAM overhead of multiple workers
    # spawning with a copy of this data
    ccf_annotations_path = Path(tempfile.mktemp(suffix='.npy'))
    np.save(ccf_annotations_path, ccf_annotations)
    del ccf_annotations
    ccf_annotations = np.load(ccf_annotations_path, mmap_mode='r')

    with open(config.tissue_bounding_boxes_path) as f:
        tissue_bboxes = json.load(f)
    tissue_bboxes = TissueBoundingBoxes(bounding_boxes=tissue_bboxes)

    # Create template points normalizer
    template_points_normalizer = None
    if config.normalize_template_points:
        template_points_normalizer = TemplatePointsNormalization(
            origin=ls_template_parameters.origin,
            scale=ls_template_parameters.scale,
            direction=ls_template_parameters.direction,
            shape=ls_template_parameters.shape
        )
        logger.info(f"Template points normalizer created with extent: {template_points_normalizer._physical_extent}")

    logger.info("Creating train dataloader")
    train_dataloader = create_dataloader(
        metadata=train_metadata,
        tissue_bboxes=tissue_bboxes,
        config=config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        buffer_batches=8,
        template_points_normalizer=template_points_normalizer,
        ls_template_parameters=ls_template_parameters
    )

    logger.info("Creating validation dataloader")
    val_dataloader = create_dataloader(
        metadata=val_metadata,
        tissue_bboxes=tissue_bboxes,
        config=config,
        batch_size=config.batch_size,
        num_workers=min(2, config.num_workers),
        buffer_batches=4,
        template_points_normalizer=template_points_normalizer,
        ls_template_parameters=ls_template_parameters
    )

    logger.info(f"Train subjects: {len(train_metadata)}")
    logger.info(f"Val subjects: {len(val_metadata)}")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=config.model.unet_channels,
        strides=config.model.unet_stride,
    )
    logger.info(model)

    if config.load_checkpoint:
        logger.info(f"Loading checkpoint from: {config.load_checkpoint}")
        checkpoint = torch.load(f=config.load_checkpoint, map_location=device)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    else:
        checkpoint = None

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    opt = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.load_checkpoint and 'optimizer_state_dict' in checkpoint:
        opt.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state from checkpoint")

    logger.info(config)

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    mlflow.set_experiment(config.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    # disable seeding so mlflow run name can be unique
    state = random.getstate()
    random.seed()

    mlflow_run = mlflow.start_run() if config.use_mlflow else nullcontext()
    with mlflow_run:
        mlflow.log_params(params=config.model_dump())
        try:
            commit, repo_url = _get_git_commit_from_package()
            mlflow.set_tag("mlflow.source.git.commit", commit)
            mlflow.set_tag("mlflow.source.git.repoURL", repo_url)
        except:
            logger.warning('Could not parse git commit')

        # Restore original seeded state
        random.setstate(state)

        best_val_loss = train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=opt,
            max_iters=config.max_iters,
            model_weights_out_dir=config.model_weights_out_dir,
            learning_rate=config.learning_rate,
            decay_learning_rate=config.decay_learning_rate,
            eval_interval=config.eval_interval,
            patience=config.patience,
            min_delta=config.min_delta,
            autocast_context=autocast_context,
            device=device,
            eval_iters=config.eval_iters,
            is_debug=config.debug,
            template_points_normalizer=template_points_normalizer,
            ls_template_parameters=ls_template_parameters,
            ccf_annotations=ccf_annotations,
            val_viz_samples=config.val_viz_samples,
            exclude_background_pixels=config.exclude_background_pixels,
        )

    logger.info("=" * 60)
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    logger.info("=" * 60)

    os.remove(ccf_annotations_path)


def split_train_val_test(
        subject_metadata: list[SubjectMetadata],
        train_split: float,
        val_split: float,
) -> tuple[list[SubjectMetadata], list[SubjectMetadata], list[SubjectMetadata]]:
    """
    Parameters
    ----------
    subject_metadata: List of all subject metadata
    train_split: Fraction of data for training
    val_split: Fraction of data for validation
    seed: Random seed for splitting

    Return
    --------
    Tuple of (train_metadata, val_metadata, test_metadata)

    Note: test_split = 1 - train_split - val_split
    """
    # Validate splits
    test_split = 1 - train_split - val_split
    if test_split < 0:
        raise ValueError(f"train_split ({train_split}) + val_split ({val_split}) must be <= 1")

    # Use random split
    logger.info(
        f"Using random train/val/test split: {train_split:.1%} train, "
        f"{val_split:.1%} val, {test_split:.1%} test"
    )

    shuffled_metadata = subject_metadata.copy()
    random.shuffle(shuffled_metadata)

    # Split
    n_train = int(len(shuffled_metadata) * train_split)
    n_val = int(len(shuffled_metadata) * val_split)

    train_metadata = shuffled_metadata[:n_train]
    val_metadata = shuffled_metadata[n_train:n_train + n_val]
    test_metadata = shuffled_metadata[n_train + n_val:]

    # Validation
    if len(train_metadata) == 0:
        raise ValueError("Training set is empty!")
    if len(val_metadata) == 0:
        raise ValueError("Validation set is empty!")
    if len(test_metadata) == 0:
        raise ValueError("Test set is empty!")

    return train_metadata, val_metadata, test_metadata

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)   # tensorstore complains "fork" not allowed
    main()
