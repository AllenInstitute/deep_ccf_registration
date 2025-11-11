from pathlib import Path
from typing import ContextManager
from contextlib import nullcontext
import os
import math

import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from segmentation_models_pytorch import Unet
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from loguru import logger

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.inference import evaluate, RegionAcronymCCFIdsMap
from deep_ccf_registration.losses.mse import HemisphereAgnosticMSE
from deep_ccf_registration.metadata import SliceOrientation


# https://github.com/karpathy/nanoGPT/blob/master/train.py
def _get_lr(
        iteration: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr
):
    """
    Calculate learning rate with linear warmup and cosine decay.

    Parameters
    ----------
    iteration: Current training iteration
    warmup_iters: Number of warmup iterations
    learning_rate: Maximum learning rate after warmup
    lr_decay_iters: Total iterations for learning rate decay
    min_lr: Minimum learning rate after decay

    Returns
    -------
    Learning rate for current iteration
    """
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train(
        train_dataloader,
        val_dataloader,
        model: Unet,
        optimizer,
        n_epochs: int,
        model_weights_out_dir: Path,
        ccf_annotations: np.ndarray,
        ls_template_to_ccf_affine_path: Path,
        ls_template_to_ccf_inverse_warp: np.ndarray,
        ls_template: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        ccf_template_parameters: AntsImageParameters,
        region_ccf_ids_map: RegionAcronymCCFIdsMap,
        learning_rate: float = 0.001,
        decay_learning_rate: bool = True,
        warmup_iters: int = 1000,
        loss_eval_interval: int = 500,
        patience: int = 10,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train slice registration model

    Parameters
    ----------
    train_dataloader: DataLoader for training data
    val_dataloader: DataLoader for validation data
    model: Neural network model to train
    optimizer: Optimizer for training
    n_epochs: Number of epochs to train
    model_weights_out_dir: Directory to save model checkpoints
    learning_rate: Initial learning rate
    decay_learning_rate: Whether to decay learning rate during training
    warmup_iters: Number of warmup iterations for learning rate
    loss_eval_interval: Evaluate loss every N iterations
    patience: Number of evaluations without improvement before stopping
    min_delta: Minimum change in validation loss to be considered improvement
    autocast_context: Context manager for mixed precision training
    device: Device to train on
    ccf_annotations: 25 micron resolution CCF annotation volume
    ls_template: light sheet template volume
    ls_template_to_ccf_affine_path: path to ls template to ccf affine
    ls_template_to_ccf_inverse_warp: ls template to ccf inverse warp
    ls_template_parameters: ls template AntsImageParameters
    ccf_template_parameters: ccf template AntsImageParameters
    region_ccf_ids_map: `RegionAcronymCCFIdsMap`

    Returns
    -------
    Best validation loss achieved during training
    """
    train_dataset: SliceDataset = train_dataloader.dataset
    if isinstance(train_dataset, Subset):
        train_dataset = train_dataset.dataset

    os.makedirs(model_weights_out_dir, exist_ok=True)

    criterion = HemisphereAgnosticMSE(
        ml_dim_size=ls_template.shape[0] * ls_template_parameters.scale[0]
    )
    best_val_rmse = float("inf")
    patience_counter = 0
    global_step = 0
    lr_decay_iters = len(train_dataloader) * n_epochs
    min_lr = learning_rate / 10 # should be ~= learning_rate/10 per Chinchilla

    model.to(device)

    logger.info(f"Starting training for {n_epochs} epochs")
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Device: {device}")

    for epoch in range(1, n_epochs + 1):
        # Set epoch for distributed training
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch=epoch)

        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            if train_dataset.patch_size is not None:
                input_images, target_ls_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, raw_shapes = batch
            else:
                input_images, target_ls_template_points, dataset_indices, slice_indices, orientations, input_image_transforms, raw_shapes = batch
            input_images, target_ls_template_points = input_images.to(device), target_ls_template_points.to(device)

            # Learning rate decay
            if decay_learning_rate:
                lr = _get_lr(
                    iteration=global_step,
                    warmup_iters=warmup_iters,
                    learning_rate=learning_rate,
                    lr_decay_iters=lr_decay_iters,
                    min_lr=min_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Forward pass
            optimizer.zero_grad()
            with autocast_context:
                pred_ls_template_points = model(input_images)
                loss = criterion(
                    pred_template_points=pred_ls_template_points,
                    true_template_points=target_ls_template_points,
                    orientations=[SliceOrientation(x) for x in orientations]
                )

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            # Periodic evaluation
            if global_step % loss_eval_interval == 0:
                train_rmse, train_major_region_dice, train_small_region_dice = evaluate(
                    val_loader=train_dataloader,
                    model=model,
                    ccf_annotations=ccf_annotations,
                    ls_template_to_ccf_affine_path=ls_template_to_ccf_affine_path,
                    ls_template_to_ccf_inverse_warp=ls_template_to_ccf_inverse_warp,
                    ls_template=ls_template,
                    ls_template_parameters=ls_template_parameters,
                    ccf_template_parameters=ccf_template_parameters,
                    region_ccf_ids_map=region_ccf_ids_map,
                    device=device
                )
                val_rmse, val_major_region_dice, val_small_region_dice = evaluate(
                    val_loader=val_dataloader,
                    model=model,
                    ccf_annotations=ccf_annotations,
                    ls_template_to_ccf_affine_path=ls_template_to_ccf_affine_path,
                    ls_template_to_ccf_inverse_warp=ls_template_to_ccf_inverse_warp,
                    ls_template=ls_template,
                    ls_template_parameters=ls_template_parameters,
                    ccf_template_parameters=ccf_template_parameters,
                    region_ccf_ids_map=region_ccf_ids_map,
                    device=device
                )

                current_lr = optimizer.param_groups[0]['lr']
                train_major_dice_avg = sum(train_major_region_dice.values()) / len(train_major_region_dice)
                train_small_dice_avg = sum(train_small_region_dice.values()) / len(
                    train_small_region_dice)

                val_major_dice_avg = sum(val_major_region_dice.values()) / len(
                    val_major_region_dice)
                val_small_dice_avg = sum(val_small_region_dice.values()) / len(
                    val_small_region_dice)

                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f} | "
                    f"Train major dice: {train_major_dice_avg:.6f} | Val major dice: {val_major_dice_avg:.6f} | "
                    f"Train small dice: {train_small_dice_avg} | Val major dice: {val_small_dice_avg:.6f} | "
                    f"LR: {current_lr:.6e}"
                )

                # Check for improvement
                if val_rmse < best_val_rmse - min_delta:
                    best_val_rmse = val_rmse
                    patience_counter = 0

                    # Save best model
                    checkpoint_path = Path(model_weights_out_dir) / "model.pt"
                    torch.save(
                        obj={
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_rmse': val_rmse,
                        },
                        f=checkpoint_path,
                    )
                    logger.info(f"New best model saved! Val RMSE: {val_rmse:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered after {global_step} steps")
                    logger.info(f"Best validation RMSE: {best_val_rmse:.6f}")
                    return best_val_rmse

                model.train()

        # End of epoch summary
        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{n_epochs} completed | Avg Train Loss: {avg_train_loss:.6f}")
        logger.info(f"{'=' * 60}\n")

        # Save epoch checkpoint
        checkpoint_path = Path(model_weights_out_dir) / f"epoch_{epoch}.pt"
        torch.save(
            obj={
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            },
            f=checkpoint_path,
        )

    logger.info(f"\nTraining completed! Best validation loss: {best_val_rmse:.6f}")
    return best_val_rmse
