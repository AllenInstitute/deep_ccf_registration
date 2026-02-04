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
import torch.distributed as dist
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters

from torch.utils.data import DataLoader
from contextlib import nullcontext
from loguru import logger
import json
import ants
import random

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.collation import collate_patch_samples
from deep_ccf_registration.datasets.iterable_slice_dataset import (
    IterableSubjectSliceDataset,
)
from deep_ccf_registration.datasets.subject_slice_sampler import SubjectSliceSampler
from deep_ccf_registration.datasets.transforms import build_transform
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, TissueBoundingBoxes
from deep_ccf_registration.models import UNetWithRegressionHeads
from deep_ccf_registration.train import train

def create_dataloader(
    metadata: list[SubjectMetadata],
    tissue_bboxes: TissueBoundingBoxes,
    config: TrainConfig,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    ls_template_parameters: TemplateParameters,
    ccf_annotations: np.ndarray,
    include_tissue_mask: bool = False,
):
    """
    Create a dataloader using IterableSubjectSliceDataset with SubjectSliceSampler.

    Returns an iterator that yields collated batch dicts.
    """
    template_parameters = TemplateParameters(
        origin=ls_template_parameters.origin,
        scale=ls_template_parameters.scale,
        direction=ls_template_parameters.direction,
        shape=ls_template_parameters.shape,
        orientation=ls_template_parameters.orientation,
    )

    transform = build_transform(
        config=config,
        template_parameters=template_parameters,
        square_symmetry=is_train and config.apply_square_symmetry_transform,
        resample_to_fixed_resolution=config.resample_to_fixed_resolution is not None,
        rotate_slices=config.rotate_slices and is_train,
        normalize_template_points=config.normalize_template_points,
        longest_max_size=config.longest_max_size is not None,
    )

    sampler = SubjectSliceSampler(
        subjects=metadata,
        orientations=[config.orientation] if config.orientation is not None else [],
        slice_fraction=config.epoch_subject_slice_fraction,
        shuffle_subjects=is_train,
        shuffle_slices_within_subject=is_train,
        seed=config.seed,
        tissue_bboxes=tissue_bboxes,
        is_debug=config.debug,
        debug_start_y=config.debug_start_y,
        debug_start_x=config.debug_start_x,
        debug_slice_idx=config.debug_slice_idx,
    )

    dataset = IterableSubjectSliceDataset(
        slice_generator=sampler,
        template_parameters=template_parameters,
        tensorstore_aws_credentials_method=config.tensorstore_aws_credentials_method,
        is_train=is_train,
        tissue_bboxes=tissue_bboxes,
        crop_size=config.patch_size,
        registration_downsample_factor=config.registration_downsample_factor,
        transform=transform,
        include_tissue_mask=include_tissue_mask,
        ccf_annotations=ccf_annotations,
        scratch_path=config.tmp_path,
        rotate_slices=config.rotate_slices,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # using 0 workers (main process) for eval,
        # to keep mem usage lower
        num_workers=num_workers if is_train else 0,
        collate_fn=collate_patch_samples,
        pin_memory=device == 'cuda',
        persistent_workers=is_train and num_workers > 0
    )

    return dataloader

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)


def is_main_process() -> bool:
    """Check if this is the main process (rank 0) for logging/mlflow."""
    return int(os.environ.get('RANK', 0)) == 0


def get_world_size() -> int:
    """Get the total number of DDP processes."""
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get('LOCAL_RANK', 0))


def setup_ddp() -> tuple[str, int]:
    """
    Initialize DDP if running in distributed mode.

    Returns:
        tuple: (device string, world_size)
    """
    world_size = get_world_size()
    local_rank = get_local_rank()

    if world_size > 1:
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        if is_main_process():
            logger.info(f"DDP initialized: world_size={world_size}, local_rank={local_rank}")
    else:
        device = None  # Will be set by config

    return device, world_size


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

    # Setup DDP if running in distributed mode
    ddp_device, world_size = setup_ddp()

    if ddp_device is not None:
        # DDP mode - use the assigned device
        device = ddp_device
    elif config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = config.device

    if is_main_process():
        logger.info(f"Using device: {device}")
        if world_size > 1:
            logger.info(f"DDP training with {world_size} GPUs")

    # Setup mixed precision
    if config.mixed_precision and device == "cuda":
        autocast_context = torch.cuda.amp.autocast()
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled with GradScaler")
    else:
        autocast_context = nullcontext()
        scaler = None
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
        orientation=ls_template_ants_parameters.orientation,
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

    if config.debug:
        logger.warning("Debug mode: using training metadata for validation to ensure shared subjects")
        val_metadata = train_metadata

    logger.info(f"Train subjects: {len(train_metadata)}")
    logger.info(f"Val subjects: {len(val_metadata)}")
    logger.info(f"Test subjects: {len(test_metadata)}")

    logger.info('loading ccf annotations volume')
    ccf_annotations = ants.image_read(str(config.ccf_annotations_path)).numpy()

    # write ccf_annotations to memmap. this avoids RAM overhead of multiple workers
    # spawning with a copy of this data
    ccf_annotations_path = Path(tempfile.mktemp(suffix='.npy', dir=config.tmp_path))
    np.save(ccf_annotations_path, ccf_annotations)
    del ccf_annotations
    ccf_annotations = np.load(ccf_annotations_path, mmap_mode='r')

    with open(config.tissue_bounding_boxes_path) as f:
        tissue_bboxes = json.load(f)
    tissue_bboxes = TissueBoundingBoxes(bounding_boxes=tissue_bboxes)


    train_dataloader = create_dataloader(
        metadata=train_metadata,
        tissue_bboxes=tissue_bboxes,
        config=config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        ls_template_parameters=ls_template_parameters,
        is_train=True,
        ccf_annotations=ccf_annotations,
        include_tissue_mask=config.predict_tissue_mask,
        device=device,
    )
    val_dataloader = create_dataloader(
        metadata=val_metadata,
        tissue_bboxes=tissue_bboxes,
        config=config,
        batch_size=config.batch_size,
        num_workers=min(2, config.num_workers),
        ls_template_parameters=ls_template_parameters,
        is_train=False,
        ccf_annotations=ccf_annotations,
        include_tissue_mask=config.predict_tissue_mask,
        device=device,
    )

    logger.info(f"Train subjects: {len(train_metadata)}")
    logger.info(f"Val subjects: {len(val_metadata)}")

    model = UNetWithRegressionHeads(
        spatial_dims=2,
        in_channels=1,
        channels=config.model.unet_channels,
        strides=config.model.unet_stride,
        out_coords=3,
        include_tissue_mask=config.predict_tissue_mask,
        use_positional_encoding=config.use_positional_encoding,
        feature_channels=config.model.feature_channels,
        input_dims=(config.pad_dim, config.pad_dim),
        pos_encoding_channels=config.model.pos_encoding_channels,
        positional_embedding_type=config.model.positional_embedding_type,
        positional_embedding_placement=config.model.positional_embedding_placement,
        coord_head_channels=config.model.coord_head_channels,
    )

    if is_main_process():
        logger.info(model)

    if config.load_checkpoint:
        if is_main_process():
            logger.info(f"Loading checkpoint from: {config.load_checkpoint}")
        checkpoint = torch.load(f=config.load_checkpoint, map_location=device)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        # Extract resume state from checkpoint
        start_step = checkpoint.get('global_step', 0)
        start_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_patience_counter = checkpoint.get('patience_counter', 0)
        scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
        if is_main_process():
            logger.info(f"Resuming from step {start_step}, best_val_loss={start_best_val_loss:.6f}")
    else:
        checkpoint = None
        start_step = 0
        start_best_val_loss = float('inf')
        start_patience_counter = 0
        scheduler_state_dict = None

    # Move model to device before wrapping with DDP
    model.to(device)

    # Wrap model with DDP if in distributed mode
    if world_size > 1:
        local_rank = get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        if is_main_process():
            logger.info("Model wrapped with DistributedDataParallel")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
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
        if is_main_process():
            logger.info("Loaded optimizer state from checkpoint")

    if is_main_process():
        logger.info(config)

    # mlflow setup - only on main process to avoid conflicts
    if is_main_process():
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        mlflow.set_experiment(config.mlflow_experiment_name)
        mlflow.enable_system_metrics_logging()

    # disable seeding so mlflow run name can be unique
    state = random.getstate()
    random.seed()

    # Only main process creates mlflow run
    if is_main_process() and config.use_mlflow:
        if config.resume_mlflow_run_id:
            mlflow_run = mlflow.start_run(run_id=config.resume_mlflow_run_id)
            logger.info(f"Resuming MLflow run: {config.resume_mlflow_run_id}")
        else:
            mlflow_run = mlflow.start_run()
            logger.info(f"Started new MLflow run: {mlflow_run.info.run_id}")
    else:
        mlflow_run = nullcontext()

    with mlflow_run:
        if is_main_process() and not config.resume_mlflow_run_id:
            # Only log params on new runs (not resume)
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
            eval_interval=config.eval_interval,
            patience=config.patience,
            min_delta=config.min_delta,
            autocast_context=autocast_context,
            scaler=scaler,
            device=device,
            eval_iters=config.eval_iters,
            is_debug=config.debug,
            ls_template_parameters=ls_template_parameters,
            ccf_annotations=ccf_annotations,
            val_viz_samples=config.val_viz_samples,
            exclude_background_pixels=config.exclude_background_pixels,
            lr_scheduler=config.lr_scheduler,
            normalize_target_points=config.normalize_template_points,
            predict_tissue_mask=config.predict_tissue_mask,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            grad_clip_max_norm=config.grad_clip_max_norm,
            warmup_steps=config.warmup_steps,
            # Resume state from checkpoint
            start_step=start_step,
            start_best_val_loss=start_best_val_loss,
            start_patience_counter=start_patience_counter,
            scheduler_state_dict=scheduler_state_dict,
        )

    if is_main_process():
        logger.info("=" * 60)
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        logger.info("=" * 60)

    os.remove(ccf_annotations_path)

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()


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
