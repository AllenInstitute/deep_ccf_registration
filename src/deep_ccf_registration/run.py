import click
import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
from loguru import logger
import json
from typing import Optional
import ants
import random

from deep_ccf_registration.models.unet import UNet
from train import train
from deep_ccf_registration.datasets.slice_dataset import (
    SliceDataset,
    SubjectMetadata,
    SliceOrientation,
    TrainMode,
    AcquisitionDirection,
)


@click.command()
# Data arguments
@click.option(
    "--dataset-meta-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to dataset metadata JSON file (list of SubjectMetadata)",
)
@click.option(
    "--train-val-split",
    type=float,
    default=0.8,
    help="Fraction of data to use for training (e.g., 0.8 means 80% train, 20% val)",
)
@click.option(
    "--ls-template-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to smartSPIM light sheet template image",
)
@click.option(
    "--orientation",
    type=click.Choice(["AXIAL", "SAGITTAL", "CORONAL"]),
    default=None,
    help="Slice orientation to load",
)
@click.option(
    "--registration-downsample-factor",
    type=int,
    default=3,
    help="Downsample factor used during registration",
)
@click.option(
    "--tensorstore-aws-credentials-method",
    type=str,
    default="default",
    help="Credentials lookup method for tensorstore",
)
@click.option(
    "--crop-warp-to-bounding-box/--no-crop-warp-to-bounding-box",
    default=True,
    help="Whether to load a cropped region of warp (faster) rather than full warp",
)
@click.option(
    "--patch-size",
    type=str,
    default="256,256",
    help="Patch size as comma-separated values (e.g., '256,256'). Use 'None' to disable patching.",
)
@click.option(
    "--normalize-orientation-map-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file with orientation normalization mapping",
)
@click.option(
    "--limit-sagittal-slices-to-hemisphere/--no-limit-sagittal-slices-to-hemisphere",
    default=False,
    help="Limit sampling to LEFT hemisphere for sagittal slices",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for training and validation",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of data loading workers",
)
# Model arguments
@click.option(
    "--load-checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Path to checkpoint to resume training from",
)
@click.option(
    "--unet-init-features",
    type=int,
    default=64,
    help="Controls the width of the network. The number of feature channels are doubled 4 times from this base.",
)
# Training arguments
@click.option(
    "--n-epochs",
    type=int,
    default=100,
    help="Number of training epochs",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Initial learning rate",
)
@click.option(
    "--optimizer",
    type=click.Choice(["adam", "adamw", "sgd", "rmsprop"]),
    default="adam",
    help="Optimizer type",
)
@click.option(
    "--weight-decay",
    type=float,
    default=0.0,
    help="Weight decay (L2 regularization)",
)
@click.option(
    "--decay-learning-rate/--no-decay-learning-rate",
    default=True,
    help="Enable/disable learning rate decay",
)
@click.option(
    "--warmup-iters",
    type=int,
    default=1000,
    help="Number of warmup iterations for learning rate",
)
# Evaluation arguments
@click.option(
    "--loss-eval-interval",
    type=int,
    default=500,
    help="Evaluate loss every N iterations",
)
@click.option(
    "--eval-iters",
    type=int,
    default=100,
    help="Number of iterations to average for evaluation",
)
# Early stopping arguments
@click.option(
    "--patience",
    type=int,
    default=10,
    help="Early stopping patience (number of evaluations without improvement)",
)
@click.option(
    "--min-delta",
    type=float,
    default=1e-4,
    help="Minimum change in validation loss to be considered improvement",
)
# Output arguments
@click.option(
    "--model-weights-out-dir",
    type=click.Path(),
    default="./checkpoints",
    help="Directory to save model checkpoints",
)
# Device arguments
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device to train on",
)
@click.option(
    "--mixed-precision/--no-mixed-precision",
    default=False,
    help="Enable/disable mixed precision training",
)
# Logging arguments
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Path to log file (if not specified, logs only to console)",
)
@click.option(
    "--seed",
    type=int,
    default=1234,
    help="Random seed for reproducibility",
)
def main(
        dataset_meta_path,
        train_val_split,
        ls_template_path,
        orientation,
        registration_downsample_factor,
        tensorstore_aws_credentials_method,
        crop_warp_to_bounding_box,
        patch_size,
        normalize_orientation_map_path,
        limit_sagittal_slices_to_hemisphere,
        batch_size,
        num_workers,
        unet_init_features,
        load_checkpoint,
        n_epochs,
        learning_rate,
        optimizer,
        weight_decay,
        decay_learning_rate,
        warmup_iters,
        loss_eval_interval,
        eval_iters,
        patience,
        min_delta,
        model_weights_out_dir,
        device,
        mixed_precision,
        log_file,
        seed,
):
    """Train a model with MSE loss and early stopping using SliceDataset."""

    # Setup logging
    if log_file:
        logger.add(sink=log_file, rotation="500 MB", level="INFO")

    logger.info("=" * 60)
    logger.info("Starting training run")
    logger.info("=" * 60)

    # Set random seed
    torch.manual_seed(seed=seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
    logger.info(f"Random seed set to: {seed}")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Setup mixed precision
    if mixed_precision and device == "cuda":
        autocast_context = torch.cuda.amp.autocast()
        logger.info("Mixed precision training enabled")
    else:
        autocast_context = nullcontext()
        if mixed_precision and device != "cuda":
            logger.warning("Mixed precision only supported on CUDA, disabling")

    # Parse patch_size
    if patch_size.lower() == "none":
        patch_size_tuple = None
    else:
        try:
            patch_size_tuple = tuple(map(int, patch_size.split(',')))
            if len(patch_size_tuple) != 2:
                raise ValueError("Patch size must have exactly 2 dimensions")
        except Exception as e:
            raise ValueError(
                f"Invalid patch_size format: {patch_size}. Use format '256,256' or 'None'") from e

    # Parse orientation
    orientation_enum = SliceOrientation[orientation] if orientation else None

    # Load normalize_orientation_map if provided
    normalize_orientation_map = None
    if normalize_orientation_map_path:
        logger.info(
            f"Loading orientation normalization map from: {normalize_orientation_map_path}")
        with open(file=normalize_orientation_map_path, mode='r') as f:
            normalize_orientation_map_dict = json.load(fp=f)
            # Convert string keys to enum
            normalize_orientation_map = {
                SliceOrientation[k]: [AcquisitionDirection[d] for d in v]
                for k, v in normalize_orientation_map_dict.items()
            }

    # Load light sheet template
    logger.info(f"Loading light sheet template from: {ls_template_path}")
    ls_template = ants.image_read(filename=ls_template_path)

    # Load dataset metadata
    logger.info(f"Loading dataset metadata from: {dataset_meta_path}")
    with open(file=dataset_meta_path, mode='r') as f:
        subject_metadata_dicts = json.load(fp=f)
    subject_metadata = [SubjectMetadata.model_validate(x) for x in subject_metadata_dicts]
    logger.info(f"Total subjects loaded: {len(subject_metadata)}")

    # Split into train/val
    train_metadata, val_metadata = split_train_val(
        subject_metadata=subject_metadata,
        train_val_split=train_val_split,
        seed=seed,
    )

    logger.info(f"Train subjects: {len(train_metadata)}")
    logger.info(f"Val subjects: {len(val_metadata)}")

    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = SliceDataset(
        dataset_meta=train_metadata,
        ls_template=ls_template,
        orientation=orientation_enum,
        registration_downsample_factor=registration_downsample_factor,
        tensorstore_aws_credentials_method=tensorstore_aws_credentials_method,
        crop_warp_to_bounding_box=crop_warp_to_bounding_box,
        patch_size=patch_size_tuple,
        mode=TrainMode.TRAIN,
        normalize_orientation_map=normalize_orientation_map,
        limit_sagittal_slices_to_hemisphere=limit_sagittal_slices_to_hemisphere,
    )

    logger.info("Creating validation dataset...")
    val_dataset = SliceDataset(
        dataset_meta=val_metadata,
        ls_template=ls_template,
        orientation=orientation_enum,
        registration_downsample_factor=registration_downsample_factor,
        tensorstore_aws_credentials_method=tensorstore_aws_credentials_method,
        crop_warp_to_bounding_box=crop_warp_to_bounding_box,
        patch_size=patch_size_tuple,
        mode=TrainMode.TEST,
        normalize_orientation_map=normalize_orientation_map,
        limit_sagittal_slices_to_hemisphere=limit_sagittal_slices_to_hemisphere,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create or load model
    model = UNet(init_features=unet_init_features, in_channels=1, out_channels=3)

    if load_checkpoint:
        logger.info(f"Loading checkpoint from: {load_checkpoint}")
        checkpoint = torch.load(f=load_checkpoint, map_location=device)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    logger.info(f"Creating optimizer: {optimizer}")
    opt = torch.optim.AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if load_checkpoint and 'optimizer_state_dict' in checkpoint:
        opt.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state from checkpoint")

    # Log training configuration
    logger.info("\nDataset Configuration:")
    logger.info(f"  Orientation: {orientation}")
    logger.info(f"  Registration downsample: {registration_downsample_factor}")
    logger.info(f"  Patch size: {patch_size_tuple}")
    logger.info(f"  Crop warp to bbox: {crop_warp_to_bounding_box}")
    logger.info(f"  Limit sagittal to hemisphere: {limit_sagittal_slices_to_hemisphere}")

    logger.info("\nTraining Configuration:")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  LR decay: {decay_learning_rate}")
    logger.info(f"  Warmup iters: {warmup_iters}")
    logger.info(f"  Loss eval interval: {loss_eval_interval}")
    logger.info(f"  Eval iters: {eval_iters}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Min delta: {min_delta}")
    logger.info(f"  Output directory: {model_weights_out_dir}")
    logger.info("")

    # Train model
    best_val_loss = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=opt,
        n_epochs=n_epochs,
        model_weights_out_dir=model_weights_out_dir,
        learning_rate=learning_rate,
        decay_learning_rate=decay_learning_rate,
        warmup_iters=warmup_iters,
        loss_eval_interval=loss_eval_interval,
        eval_iters=eval_iters,
        patience=patience,
        min_delta=min_delta,
        autocast_context=autocast_context,
        device=device,
    )

    logger.info("=" * 60)
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    logger.info("=" * 60)


def split_train_val(
        subject_metadata: list[SubjectMetadata],
        train_val_split: float,
        seed: int,
) -> tuple[list[SubjectMetadata], list[SubjectMetadata]]:
    """
    Split subject metadata into train and validation sets.

    Args:
        subject_metadata: List of all subject metadata
        train_val_split: Fraction of data for training (ignored if subject ID files provided)
        seed: Random seed for splitting

    Returns:
        Tuple of (train_metadata, val_metadata)
    """
    # Use random split
    logger.info(
        f"Using random train/val split: {train_val_split:.1%} train, {1 - train_val_split:.1%} val")

    # Shuffle with seed
    random.seed(seed)
    shuffled_metadata = subject_metadata.copy()
    random.shuffle(shuffled_metadata)

    # Split
    n_train = int(len(shuffled_metadata) * train_val_split)
    train_metadata = shuffled_metadata[:n_train]
    val_metadata = shuffled_metadata[n_train:]

    if len(train_metadata) == 0:
        raise ValueError("Training set is empty!")
    if len(val_metadata) == 0:
        raise ValueError("Validation set is empty!")

    return train_metadata, val_metadata


if __name__ == "__main__":
    main()