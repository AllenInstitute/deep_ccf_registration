import multiprocessing
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

import albumentations
import click
import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters

from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext
from loguru import logger
import json
import ants
import random

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.inference import RegionAcronymCCFIdsMap
from deep_ccf_registration.datasets.slice_dataset import (
    SliceDataset,
    TrainMode, TissueBoundingBoxes,
)
from deep_ccf_registration.metadata import SubjectMetadata
from deep_ccf_registration.train import train
from deep_ccf_registration.models import UNetWithRegressionHeads

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
    ls_template_parameters = AntsImageParameters.from_ants_image(image=ls_template)

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

    with open(config.tissue_bounding_boxes_path) as f:
        tissue_bboxes = json.load(f)
    tissue_bboxes = TissueBoundingBoxes(bounding_boxes=tissue_bboxes)

    train_dataset = SliceDataset(
        dataset_meta=train_metadata,
        ls_template_parameters=ls_template_parameters,
        orientation=config.orientation,
        registration_downsample_factor=config.registration_downsample_factor,
        tensorstore_aws_credentials_method=config.tensorstore_aws_credentials_method,
        crop_warp_to_bounding_box=config.crop_warp_to_bounding_box,
        patch_size=config.patch_size,
        mode=TrainMode.TEST if config.debug else TrainMode.TRAIN,   # deterministic if debug
        normalize_orientation_map=config.normalize_orientation_map,
        limit_sagittal_slices_to_hemisphere=config.limit_sagittal_slices_to_hemisphere,
        input_image_transforms=[
            albumentations.PadIfNeeded(min_height=config.patch_size[0], min_width=config.patch_size[1]),
            albumentations.ToTensorV2()
        ],
        mask_transforms=[
            albumentations.PadIfNeeded(min_height=config.patch_size[0], min_width=config.patch_size[1]),
            albumentations.ToTensorV2()
        ],
        output_points_transforms=[
            albumentations.PadIfNeeded(min_height=config.patch_size[0], min_width=config.patch_size[1]),
            albumentations.ToTensorV2()
        ],
        ccf_annotations=ccf_annotations,
        return_tissue_mask=config.exclude_background_pixels,
        tissue_bboxes=tissue_bboxes
    )

    val_dataset = SliceDataset(
        dataset_meta=val_metadata,
        ls_template_parameters=ls_template_parameters,
        orientation=config.orientation,
        registration_downsample_factor=config.registration_downsample_factor,
        tensorstore_aws_credentials_method=config.tensorstore_aws_credentials_method,
        crop_warp_to_bounding_box=config.crop_warp_to_bounding_box,
        patch_size=config.patch_size,
        mode=TrainMode.TEST,
        normalize_orientation_map=config.normalize_orientation_map,
        limit_sagittal_slices_to_hemisphere=config.limit_sagittal_slices_to_hemisphere,
        input_image_transforms=[
            albumentations.PadIfNeeded(min_height=config.patch_size[0], min_width=config.patch_size[1]),
            albumentations.ToTensorV2()
        ],
        mask_transforms=[
            albumentations.PadIfNeeded(min_height=config.patch_size[0], min_width=config.patch_size[1]),
            albumentations.ToTensorV2()
        ],
        output_points_transforms=[
            albumentations.PadIfNeeded(min_height=config.patch_size[0], min_width=config.patch_size[1]),
            albumentations.ToTensorV2()
        ],
        ccf_annotations=ccf_annotations,
        return_tissue_mask=config.exclude_background_pixels,
        tissue_bboxes=tissue_bboxes
    )

    if config.debug:
        train_dataset = Subset(train_dataset, indices=[1000])
        val_dataset = Subset(val_dataset, indices=[1000])

    train_eval_subset = Subset(train_dataset, indices=random.sample(range(len(train_dataset)), k=int(len(train_dataset) * config.train_eval_frac)))
    val_eval_subset = Subset(val_dataset, indices=random.sample(range(len(val_dataset)), k=int(len(val_dataset) * config.val_eval_frac)))

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )

    train_eval_dataloader = DataLoader(
        dataset=train_eval_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_eval_dataloader = DataLoader(
        dataset=val_eval_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Num train slices: {len(train_dataset)}")
    logger.info(f"Num val slices: {len(val_dataset)}")

    logger.info(f"Num train eval slices: {len(train_eval_subset)}")
    logger.info(f"Num val eval slices: {len(val_eval_subset)}")

    model = UNetWithRegressionHeads(
        spatial_dims=2,
        in_channels=1,
        feature_channels=config.model.unet_feature_channels,
        dropout=config.model.unet_dropout,
        channels=config.model.unet_channels,
        strides=config.model.unet_stride,
        out_coords=3,
        include_tissue_mask=config.exclude_background_pixels,
        head_size=config.model.unet_head_size,
        use_positional_encoding=config.model.unet_use_positional_encoding,
        pos_encoding_channels=config.model.unet_pos_encoding_channels,
        image_height=config.patch_size[0],
        image_width=config.patch_size[1],
    )

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

    with open(config.region_ccf_ids_map_path) as f:
        region_ccf_ids_map = json.load(f)
    region_ccf_ids_map = RegionAcronymCCFIdsMap.model_validate(region_ccf_ids_map)

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    mlflow.set_experiment(config.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()
    mlflow.pytorch.autolog()

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

        best_val_rmse = train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_eval_dataloader=train_eval_dataloader,
            val_eval_dataloader=val_eval_dataloader,
            model=model,
            optimizer=opt,
            n_epochs=config.n_epochs,
            model_weights_out_dir=config.model_weights_out_dir,
            learning_rate=config.learning_rate,
            decay_learning_rate=config.decay_learning_rate,
            warmup_iters=config.warmup_iters,
            eval_interval=config.eval_interval,
            patience=config.patience,
            min_delta=config.min_delta,
            autocast_context=autocast_context,
            device=device,
            ccf_annotations=ccf_annotations,
            ls_template=ls_template,
            ls_template_parameters=ls_template_parameters,
            region_ccf_ids_map=region_ccf_ids_map,
            exclude_background_pixels=config.exclude_background_pixels,
        )

    logger.info("=" * 60)
    logger.info(f"Training completed! Best validation RMSE: {best_val_rmse:.6f}")
    logger.info("=" * 60)


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
    multiprocessing.set_start_method('spawn', force=True)
    main()