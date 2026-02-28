import datetime
import multiprocessing
import os
import sys
import traceback
from importlib.metadata import distribution
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from torch.distributed.elastic.multiprocessing.errors import record

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

def _record(main_func):
    return record(main_func)

def create_dataloader(
    subjects: list[SubjectMetadata],
    samples: np.ndarray,
    tissue_bboxes_path: Path,
    rotation_angles: RotationAngles,
    config: TrainConfig,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    ls_template_parameters: TemplateParameters,
    ccf_annotations_path: str,
    include_tissue_mask: bool = False,
    world_size: int = 1,
):
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
        subjects=subjects,
        samples=samples,
        template_parameters=template_parameters,
        is_train=is_train,
        tissue_bboxes_path=tissue_bboxes_path,
        rotation_angles=rotation_angles,
        orientations=[config.orientation] if config.orientation is not None else [],
        crop_size=config.patch_size,
        transform=transform,
        include_tissue_mask=include_tissue_mask,
        ccf_annotations_path=ccf_annotations_path,
        rotate_slices=config.data_augmentation.rotate_slices and is_train,
        is_debug=config.debug,
        debug_slice_idx=config.debug_slice_idx,
        aws_credentials_method=config.tensorstore_aws_credentials_method,
    )

    num_workers = num_workers if is_train else 0

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_patch_samples,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return dataloader, sampler

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
        # Initialize the process group with extended timeout for long validation runs
        timeout = datetime.timedelta(minutes=60)  # Increase from default 30 min to 60 min
        dist.init_process_group(backend='nccl', timeout=timeout, device_id=torch.device(f'cuda:{local_rank}'))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
        if is_main_process():
            logger.info(f"DDP initialized: world_size={world_size}, local_rank={local_rank}, timeout={timeout}")
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
    try:
        _main(config_path)
    except Exception:
        rank = int(os.environ.get('RANK', 0))
        logger.error(f"Rank {rank} failed with exception:\n{traceback.format_exc()}")
        raise


def _main(config_path: Path):
    with open(config_path) as f:
        config = json.load(f)
    config = TrainConfig.model_validate(config)

    if config.log_file:
        logger.add(sink=config.logging.log_file, rotation="500 MB", level="INFO")

    logger.info("=" * 60)
    logger.info("Starting training run")
    logger.info("=" * 60)

    # Setup DDP if running in distributed mode
    ddp_device, world_size = setup_ddp()

    # Offset random seeds by rank so each process has different randomness
    rank = int(os.environ.get('RANK', 0))
    torch.manual_seed(seed=config.seed + rank)
    np.random.seed(config.seed)
    random.seed(config.seed)

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
    if config.mixed_precision:
        autocast_context = torch.cuda.amp.autocast()
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled with GradScaler")
    else:
        autocast_context = nullcontext()
        scaler = None

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

    logger.info(f"Loading dataset metadata from: {config.dataset_meta_path}")
    with open(file=config.dataset_meta_path, mode='r') as f:
        subject_metadata_dicts = json.load(fp=f)
    subject_metadata = [SubjectMetadata.model_validate(x) for x in subject_metadata_dicts]
    logger.info(f"Total subjects loaded: {len(subject_metadata)}")

    train_samples = np.load(config.train_samples_path, mmap_mode='r')
    val_samples = np.load(config.val_samples_path, mmap_mode='r')

    if config.debug:
        logger.warning("Debug mode: using training metadata for validation to ensure shared subjects")
        val_samples = train_samples

    logger.info(f"Train subjects: {len(np.unique(train_samples[:, 0]))}")
    logger.info(f"Val subjects: {len(np.unique(val_samples[:, 0]))}")

    # Write ccf_annotations to memmap. This avoids RAM overhead of multiple workers
    # spawning with a copy of this data.
    ccf_annotations_path = config.tmp_path / 'ccf_annotations.npy'
    if is_main_process():
        logger.info('loading ccf annotations volume')
        ccf_annotations = ants.image_read(str(config.ccf_annotations_path)).numpy()
        np.save(ccf_annotations_path, ccf_annotations)
        del ccf_annotations
        logger.info('ccf annotations saved to memmap')

    # Wait for rank 0 to finish writing before any rank reads
    if dist.is_initialized():
        dist.barrier(device_ids=[get_local_rank()])

    ccf_annotations_memmap_path = str(ccf_annotations_path)
    ccf_annotations = np.load(ccf_annotations_path, mmap_mode='r')

    rotation_angles = pd.read_csv(config.rotation_angles_path).set_index('subject_id')
    rotation_angles = RotationAngles(
        rotation_angles={x.Index: SubjectRotationAngle(AP_rot=x.AP_rotation, ML_rot=x.ML_rotation, SI_rot=x.SI_rotation) for x in rotation_angles.itertuples(index=True)},
        SI_range=(
            rotation_angles['SI_rotation'].min(),
            rotation_angles['SI_rotation'].max()
        ),
        ML_range=(
            rotation_angles['ML_rotation'].min(),
            rotation_angles['ML_rotation'].max()
        ),
        AP_range=(
            rotation_angles['AP_rotation'].min(),
            rotation_angles['AP_rotation'].max()
        ),
    )

    train_dataloader, train_sampler = create_dataloader(
        subjects=subject_metadata,
        samples=train_samples,
        tissue_bboxes_path=config.tissue_bounding_boxes_path,
        rotation_angles=rotation_angles,
        config=config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        ls_template_parameters=ls_template_parameters,
        is_train=True,
        ccf_annotations_path=ccf_annotations_memmap_path,
        include_tissue_mask=config.predict_tissue_mask,
        device=device,
        world_size=world_size,
    )
    val_dataloader, _ = create_dataloader(
        subjects=subject_metadata,
        samples=val_samples,
        tissue_bboxes_path=config.tissue_bounding_boxes_path,
        config=config,
        batch_size=config.batch_size,
        num_workers=min(2, config.num_workers),
        ls_template_parameters=ls_template_parameters,
        is_train=False,
        ccf_annotations_path=ccf_annotations_memmap_path,
        include_tissue_mask=config.predict_tissue_mask,
        device=device,
        rotation_angles=rotation_angles,
        world_size=world_size,
    )

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
        start_epoch = checkpoint.get('epoch', 0)
        start_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
        dwa_state_dict = checkpoint.get('dwa_state_dict', None)
        if is_main_process():
            logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={start_best_val_loss:.6f}")
    else:
        checkpoint = None
        start_epoch = 0
        start_best_val_loss = float('inf')
        scheduler_state_dict = None
        dwa_state_dict = None

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

    if is_main_process():
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        mlflow.set_experiment(config.mlflow_experiment_name)
        mlflow.enable_system_metrics_logging()

    # disable seeding so mlflow run name can be unique
    state = random.getstate()
    random.seed()

    if is_main_process() and config.use_mlflow:
        mlflow_run = mlflow.start_run()
        logger.info(f"Started new MLflow run: {mlflow_run.info.run_id}")
    else:
        mlflow_run = nullcontext()

    with mlflow_run:
        if is_main_process():
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
            num_epochs=config.num_epochs,
            train_sampler=train_sampler,
            model_weights_out_dir=config.model_weights_out_dir,
            learning_rate=config.learning_rate,
            autocast_context=autocast_context,
            scaler=scaler,
            device=device,
            eval_iters=config.eval_iters,
            is_debug=config.debug,
            ls_template_parameters=ls_template_parameters,
            ccf_annotations=ccf_annotations,
            val_viz_samples=config.val_viz_samples,
            lr_scheduler_type=config.lr_scheduler,
            cosine_warm_restarts_T_0=config.cosine_warm_restarts_T_0,
            normalize_target_points=config.normalize_template_points,
            predict_tissue_mask=config.predict_tissue_mask,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            grad_clip_max_norm=config.grad_clip_max_norm,
            warmup_steps=config.warmup_steps,
            log_interval=config.log_interval,
            start_epoch=start_epoch,
            start_best_val_loss=start_best_val_loss,
            scheduler_state_dict=scheduler_state_dict,
            terminology_path=config.terminology_path,
            checkpoint_interval=config.checkpoint_interval,
            multi_task_loss_init_weights=config.multi_task_loss_init_weights,
            dwa_state_dict=dwa_state_dict,
            eval_interval=config.eval_interval,
        )

    if is_main_process():
        logger.info("=" * 60)
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        logger.info("=" * 60)

    if is_main_process():
        os.remove(ccf_annotations_path)

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)   # tensorstore complains "fork" not allowed
    main = _record(main)
    main()
