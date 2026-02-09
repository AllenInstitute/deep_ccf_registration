import multiprocessing
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import distribution
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd
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
from deep_ccf_registration.datasets.slice_dataset import SubjectSliceDataset
from deep_ccf_registration.datasets.transforms import build_transform
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, TissueBoundingBoxes, RotationAngles, \
    SubjectRotationAngle
from deep_ccf_registration.models import UNetWithRegressionHeads
from deep_ccf_registration.train import train
from deep_ccf_registration.utils.logging_utils import ProgressLogger

_SENTINEL = object()


def create_dataloader(
    metadata: list[SubjectMetadata],
    tissue_bboxes: TissueBoundingBoxes,
    rotation_angles: RotationAngles,
    config: TrainConfig,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    ls_template_parameters: TemplateParameters,
    ccf_annotations: np.ndarray,
    include_tissue_mask: bool = False,
    local_cache_dir: Path | None = _SENTINEL,
):
    """
    Create a dataloader using SubjectSliceDataset (Map-style).

    Returns a DataLoader that yields collated batch dicts.
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
        square_symmetry=is_train and config.data_augmentation.apply_square_symmetry_transform,
        resample_to_fixed_resolution=config.resample_to_fixed_resolution is not None,
        rotate_slices=config.data_augmentation.rotate_slices and is_train,
        normalize_template_points=config.normalize_template_points,
        apply_grid_distortion=config.data_augmentation.apply_grid_distortion and is_train,
        rotation_angles=rotation_angles,
    )

    dataset = SubjectSliceDataset(
        subjects=metadata,
        template_parameters=template_parameters,
        is_train=is_train,
        tissue_bboxes=tissue_bboxes,
        rotation_angles=rotation_angles,
        orientations=[config.orientation] if config.orientation is not None else [],
        crop_size=config.patch_size,
        transform=transform,
        include_tissue_mask=include_tissue_mask,
        ccf_annotations=ccf_annotations,
        cache_dir=config.cache_dir,
        local_cache_dir=config.local_cache_dir if local_cache_dir is _SENTINEL else local_cache_dir,
        rotate_slices=config.data_augmentation.rotate_slices and is_train,
        is_debug=config.debug,
        debug_slice_idx=config.debug_slice_idx,
        subject_slice_fraction=config.subject_slice_fraction,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
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


def _stage_single_file(src_path: Path, local_path: Path) -> str:
    """Copy a single file from remote to local storage atomically.

    Returns the filename for logging purposes.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(local_path.parent), suffix=".npy.tmp"
    )
    try:
        os.close(tmp_fd)
        shutil.copy2(str(src_path), tmp_path)
        Path(tmp_path).rename(local_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return src_path.name


def _prestage_data(
    subject_ids: set[str],
    cache_dir: Path,
    local_cache_dir: Path,
    max_workers: int = 32,
) -> None:
    """Copy volume and warp files from remote storage to local disk.

    Runs once before training starts so that DataLoader workers can
    mmap from fast local storage instead of slow remote filesystems.
    Files that already exist locally are skipped.

    Uses a thread pool to maximise EFS throughput (which scales with
    concurrent connections).
    """
    subdirs = ["volumes", "warps"]
    for subdir in subdirs:
        (local_cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Collect files that need staging
    to_stage: list[tuple[Path, Path]] = []

    for subject_id in subject_ids:
        for src_path, local_path in [
            (
                cache_dir / "volumes" / f"{subject_id}.npy",
                local_cache_dir / "volumes" / f"{subject_id}.npy",
            ),
            (
                cache_dir / "warps" / f"{subject_id}_warp.npy",
                local_cache_dir / "warps" / f"{subject_id}_warp.npy",
            ),
        ]:
            if not local_path.exists() and src_path.exists():
                to_stage.append((src_path, local_path))

    if not to_stage:
        logger.info("All subject data already staged locally")
        return

    total = len(to_stage)
    logger.info(
        f"Staging {total} files from {cache_dir} -> {local_cache_dir} "
        f"(max_workers={max_workers})"
    )
    progress_logger = ProgressLogger(
        desc=f'copying data to {local_cache_dir}',
        log_every=1,
        total=total,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_stage_single_file, src, dst): (src, dst)
            for src, dst in to_stage
        }
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                name = future.result()
                completed += 1
                logger.info(f"  [{completed}/{total}] {name}")
            except Exception:
                logger.exception(f"Failed to stage {src} -> {dst}")
                raise
            progress_logger.log_progress()

    logger.info("Data staging complete")


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

    with open(config.rotation_angles_path) as f:
        rotation_angles = pd.read_csv(f).set_index('subject_id')
    rotation_angles = RotationAngles(
        rotation_angles={x.Index: SubjectRotationAngle(AP_rot=x.AP_rotation, ML_rot=x.ML_rotation, SI_rot=x.SI_rotation) for x in rotation_angles.itertuples(index=True)},
        SI_range=(
            rotation_angles['SI_rotation'].mean() - rotation_angles['SI_rotation'].std()*2,
            rotation_angles['SI_rotation'].mean() + rotation_angles['SI_rotation'].std()*2
        ),
        ML_range=(
            rotation_angles['ML_rotation'].mean() - rotation_angles['ML_rotation'].std()*2,
            rotation_angles['ML_rotation'].mean() + rotation_angles['ML_rotation'].std()*2
        ),
        AP_range=(
            rotation_angles['AP_rotation'].mean() - rotation_angles['AP_rotation'].std()*2,
            rotation_angles['AP_rotation'].mean() + rotation_angles['AP_rotation'].std()*2
        ),
    )

    # Determine whether to use subject cycling or stage everything upfront
    use_subject_cycling = (
        config.subject_cache_size is not None
        and config.local_cache_dir is not None
        and config.subject_cache_size < len(train_metadata)
    )

    if not use_subject_cycling:
        # Original behavior: stage all subjects, create dataloaders once
        if config.local_cache_dir is not None:
            config.local_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local data cache: {config.local_cache_dir}")
            all_subjects = set(s.subject_id for s in train_metadata + val_metadata)
            _prestage_data(
                subject_ids=all_subjects,
                cache_dir=config.cache_dir,
                local_cache_dir=config.local_cache_dir,
                max_workers=config.prestage_workers,
            )

        train_dataloader = create_dataloader(
            metadata=train_metadata,
            tissue_bboxes=tissue_bboxes,
            rotation_angles=rotation_angles,
            config=config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            ls_template_parameters=ls_template_parameters,
            is_train=True,
            ccf_annotations=ccf_annotations,
            include_tissue_mask=config.predict_tissue_mask,
            device=device,
        )
        # Preload all subjects into RAM once — zero disk I/O during training
        train_dataloader.dataset.preload_subjects()

    # Val dataloader: created once. In cycling mode, mmap from EFS directly
    # (no local cache) since val uses 0 workers and runs infrequently.
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
        rotation_angles=rotation_angles,
        local_cache_dir=None if use_subject_cycling else _SENTINEL,
    )

    logger.info(f"Train subjects: {len(train_metadata)}")
    logger.info(f"Val subjects: {len(val_metadata)}")

    model = UNetWithRegressionHeads(
        in_channels=1,
        out_coords=3,
        include_tissue_mask=config.predict_tissue_mask,
        use_positional_encoding=config.use_positional_encoding,
        feature_channels=config.model.feature_channels,
        input_dims=config.patch_size,
        pos_encoding_channels=config.model.pos_encoding_channels,
        positional_embedding_type=config.model.positional_embedding_type,
        positional_embedding_placement=config.model.positional_embedding_placement,
        coord_head_channels=config.model.coord_head_channels,
        encoder_name=config.model.encoder_name,
        encoder_weights=config.model.encoder_weights,
        encoder_depth=config.model.encoder_depth,
        decoder_channels=config.model.decoder_channels,
        decoder_use_norm=config.model.decoder_use_norm,
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

    # Common kwargs for train() calls
    train_kwargs = dict(
        val_dataloader=val_dataloader,
        model=model,
        optimizer=opt,
        val_dataset=val_dataloader.dataset,
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
    )

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

        if use_subject_cycling:
            config.local_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Subject cycling: {config.subject_cache_size} subjects per group, "
                        f"local cache: {config.local_cache_dir}")

            all_train = list(train_metadata)
            n = config.subject_cache_size
            current_step = start_step
            current_best_val_loss = start_best_val_loss
            current_patience = start_patience_counter
            current_sched_state = scheduler_state_dict
            done = False
            overall_progress = ProgressLogger(
                desc='Training', total=config.max_iters, log_every=20,
            )

            while current_step < config.max_iters and not done:
                random.shuffle(all_train)
                groups = [all_train[i:i + n] for i in range(0, len(all_train), n)]

                for group_idx, group in enumerate(groups):
                    group_ids = {s.subject_id for s in group}
                    logger.info(f"Subject group {group_idx + 1}/{len(groups)}: "
                                f"{len(group)} subjects, step {current_step}")

                    _prestage_data(
                        subject_ids=group_ids,
                        cache_dir=config.cache_dir,
                        local_cache_dir=config.local_cache_dir,
                        max_workers=config.prestage_workers,
                    )

                    train_dl = create_dataloader(
                        metadata=group,
                        tissue_bboxes=tissue_bboxes,
                        rotation_angles=rotation_angles,
                        config=config,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        ls_template_parameters=ls_template_parameters,
                        is_train=True,
                        ccf_annotations=ccf_annotations,
                        include_tissue_mask=config.predict_tissue_mask,
                        device=device,
                        local_cache_dir=config.local_cache_dir,
                    )
                    # Preload group's subjects into RAM — zero disk I/O during training
                    train_dl.dataset.preload_subjects()

                    # One pass through this group's data
                    group_max_iters = min(
                        current_step + len(train_dl),
                        config.max_iters,
                    )

                    result = train(
                        train_dataloader=train_dl,
                        max_iters=group_max_iters,
                        train_dataset=train_dl.dataset,
                        start_step=current_step,
                        start_best_val_loss=current_best_val_loss,
                        start_patience_counter=current_patience,
                        scheduler_state_dict=current_sched_state,
                        progress_logger=overall_progress,
                        **train_kwargs,
                    )

                    current_step = result["global_step"]
                    current_best_val_loss = result["best_val_loss"]
                    current_patience = result["patience_counter"]
                    current_sched_state = result["scheduler_state_dict"]

                    # Clean up dataloader workers before next group
                    del train_dl

                    if current_step >= config.max_iters:
                        done = True
                        break
                    if current_patience >= config.patience:
                        done = True
                        break

            best_val_loss = current_best_val_loss

        else:
            # Original path: single train() call with all subjects
            result = train(
                train_dataloader=train_dataloader,
                max_iters=config.max_iters,
                train_dataset=train_dataloader.dataset,
                start_step=start_step,
                start_best_val_loss=start_best_val_loss,
                start_patience_counter=start_patience_counter,
                scheduler_state_dict=scheduler_state_dict,
                **train_kwargs,
            )
            best_val_loss = result["best_val_loss"]

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
