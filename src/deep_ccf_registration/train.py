import time
from datetime import timedelta
from pathlib import Path
from typing import ContextManager, Any
from contextlib import nullcontext
import os
import math

import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from monai.networks.nets import UNet
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
import torch.nn.functional as F

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.inference import evaluate, RegionAcronymCCFIdsMap
from deep_ccf_registration.losses.coord_loss import HemisphereAgnosticCoordLoss
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

def _mask_pad_pixels(tissue_loss_per_pixel: torch.Tensor, input_image_transforms: dict[str, Any]) -> torch.Tensor:
    """
    Ignore pad pixels for loss

    :param tissue_loss_per_pixel: B, H, W tensor for BCE loss per pixel
    :param input_image_transforms: transforms applied to input image
    :return: tissue loss, ignoring pad pixels
    """
    batch_size, height, width = tissue_loss_per_pixel.shape
    valid_mask = torch.ones((batch_size, height, width), dtype=torch.bool, device='cpu')

    for i in range(batch_size):
        pad_top = input_image_transforms['pad_top'][i].item()
        pad_bottom = input_image_transforms['pad_bottom'][i].item()
        pad_left = input_image_transforms['pad_left'][i].item()
        pad_right = input_image_transforms['pad_right'][i].item()

        # Mark padded regions as False
        if pad_top > 0:
            valid_mask[i, :pad_top, :] = False
        if pad_bottom > 0:
            valid_mask[i, -pad_bottom:, :] = False
        if pad_left > 0:
            valid_mask[i, :, :pad_left] = False
        if pad_right > 0:
            valid_mask[i, :, -pad_right:] = False

    tissue_loss = tissue_loss_per_pixel[valid_mask].mean()
    return tissue_loss

def train(
        train_dataloader,
        val_dataloader,
        train_eval_dataloader,
        val_eval_dataloader,
        model: UNet,
        optimizer,
        n_epochs: int,
        model_weights_out_dir: Path,
        ccf_annotations: np.ndarray,
        ls_template: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        region_ccf_ids_map: RegionAcronymCCFIdsMap,
        learning_rate: float = 0.001,
        decay_learning_rate: bool = True,
        warmup_iters: int = 1000,
        eval_interval: int = 500,
        patience: int = 10,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        exclude_background_pixels: bool = True,
        n_eval_visualize: int = 10,
        log_interval: int = 10
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
    eval_interval: Evaluate model every N iterations
    patience: Number of evaluations without improvement before stopping
    min_delta: Minimum change in validation loss to be considered improvement
    autocast_context: Context manager for mixed precision training
    device: Device to train on
    ccf_annotations: 25 micron resolution CCF annotation volume
    ls_template: light sheet template volume
    ls_template_parameters: ls template AntsImageParameters
    region_ccf_ids_map: `RegionAcronymCCFIdsMap`
    exclude_background_pixels: whether to use a tissue mask to exclude background pixels in loss/evaluation.
        Otherwise, just excludes pad pixels
    n_eval_visualize: how many samples to visualize during evaluation
    log_interval: how often to log

    Returns
    -------
    Best validation loss achieved during training
    """
    train_dataset: SliceDataset = train_dataloader.dataset
    if isinstance(train_dataset, Subset):
        train_dataset = train_dataset.dataset

    os.makedirs(model_weights_out_dir, exist_ok=True)

    coord_loss = HemisphereAgnosticCoordLoss(
        ml_dim_size=ls_template.shape[0],
        template_parameters=ls_template_parameters
    )
    best_val_coord_loss = float("inf")
    patience_counter = 0
    global_step = 0
    lr_decay_iters = len(train_dataloader) * n_epochs
    min_lr = learning_rate / 10 # should be ~= learning_rate/10 per Chinchilla

    train_viz_indices = list(range(min(n_eval_visualize, len(train_eval_dataloader.dataset))))
    val_viz_indices = list(range(min(n_eval_visualize, len(val_eval_dataloader.dataset))))

    logger.info(f"Fixed train visualization indices: {train_viz_indices}")
    logger.info(f"Fixed val visualization indices: {val_viz_indices}")

    model.to(device)

    logger.info(f"Starting training for {n_epochs} epochs")
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Device: {device}")

    iteration_times = []
    training_start_time = time.time()
    total_iterations = len(train_dataloader) * n_epochs

    for epoch in range(1, n_epochs + 1):
        # Set epoch for distributed training
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch=epoch)

        model.train()
        train_losses = []
        train_coord_losses = []
        train_mask_losses = []

        iter_start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):

            if train_dataset.patch_size is not None:
                input_images, target_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, tissue_masks = batch
            else:
                input_images, target_template_points, dataset_indices, slice_indices, orientations, input_image_transforms, tissue_masks = batch
            input_images, target_template_points, tissue_masks = input_images.to(device), target_template_points.to(device), tissue_masks.to(device)

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
                model_out = model(input_images)
                coordinate_loss = coord_loss(
                    pred_template_points=model_out[:, :-1] if exclude_background_pixels else model_out,
                    true_template_points=target_template_points,
                    tissue_masks=tissue_masks,
                    orientations=[SliceOrientation(x) for x in orientations]
                )

                if exclude_background_pixels:
                    tissue_loss_per_pixel = F.binary_cross_entropy_with_logits(
                        model_out[:, -1].cpu().float(), # moving to cpu since bug with BCE and mps locally
                        tissue_masks.cpu().float(),  # moving to cpu since bug with BCE and mps locally
                        reduction='none'
                    )
                    tissue_loss = _mask_pad_pixels(
                        tissue_loss_per_pixel=tissue_loss_per_pixel,
                        input_image_transforms=input_image_transforms
                    )
                    loss = coordinate_loss + tissue_loss
                else:
                    loss = coordinate_loss

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_coord_losses.append(coordinate_loss.item())
            if exclude_background_pixels:
                train_mask_losses.append(tissue_loss.item())

            global_step += 1

            mlflow.log_metrics({
                "train/coord_loss": coordinate_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            }, step=global_step)

            iter_time = time.time() - iter_start_time
            iteration_times.append(iter_time)

            if global_step % log_interval == 0:
                avg_iter_time = sum(iteration_times[-100:]) / min(len(iteration_times),
                                                                  100)  # Last 100 iters
                remaining_iters = total_iterations - global_step
                eta_seconds = remaining_iters * avg_iter_time
                eta_str = str(timedelta(seconds=int(eta_seconds)))

                elapsed_time = time.time() - training_start_time
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))

                logger.info(
                    f"Step {global_step}/{total_iterations} | "
                    f"Iter time: {iter_time:.3f}s | Avg: {avg_iter_time:.3f}s | "
                    f"Loss: {loss.item():.6f} | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                )

            # Periodic evaluation
            if global_step % eval_interval == 0:
                train_rmse, train_major_region_dice, train_small_region_dice, train_tissue_mask_dice = evaluate(
                    val_loader=train_eval_dataloader,
                    model=model,
                    ccf_annotations=ccf_annotations,
                    ls_template=ls_template,
                    ls_template_parameters=ls_template_parameters,
                    region_ccf_ids_map=region_ccf_ids_map,
                    device=device,
                    iteration=global_step,
                    exclude_background_pixels=exclude_background_pixels,
                    is_train=True,
                    viz_slice_indices=train_viz_indices
                )
                val_rmse, val_major_region_dice, val_small_region_dice, val_tissue_mask_dice = evaluate(
                    val_loader=val_eval_dataloader,
                    model=model,
                    ccf_annotations=ccf_annotations,
                    ls_template=ls_template,
                    ls_template_parameters=ls_template_parameters,
                    region_ccf_ids_map=region_ccf_ids_map,
                    device=device,
                    iteration=global_step,
                    exclude_background_pixels=exclude_background_pixels,
                    is_train=False,
                    viz_slice_indices=val_viz_indices
                )

                current_lr = optimizer.param_groups[0]['lr']
                train_major_dice_avg = sum(train_major_region_dice.values()) / len(train_major_region_dice) if len(train_major_region_dice) > 0 else np.nan
                train_small_dice_avg = sum(train_small_region_dice.values()) / len(
                    train_small_region_dice) if len(train_small_region_dice) > 0 else np.nan

                val_major_dice_avg = sum(val_major_region_dice.values()) / len(
                    val_major_region_dice) if len(val_major_region_dice) > 0 else np.nan
                val_small_dice_avg = sum(val_small_region_dice.values()) / len(
                    val_small_region_dice) if len(val_small_region_dice) > 0 else np.nan

                mlflow.log_metrics(metrics={
                    "eval/train_rmse": train_rmse,
                    "eval/val_rmse": val_rmse,
                    "eval/train_major_dice": train_major_dice_avg,
                    "eval/val_major_dice": val_major_dice_avg,
                    "eval/train_small_dice": train_small_dice_avg,
                    "eval/val_small_dice": val_small_dice_avg,
                },
                    step=global_step
                )

                if exclude_background_pixels:
                    mask_log = f"Train mask dice: {train_tissue_mask_dice} | "
                    f"Val mask dice: {val_tissue_mask_dice} | "
                else:
                    mask_log = ""
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Train RMSE: {train_rmse:.6f} microns | Val RMSE: {val_rmse:.6f} microns | "
                    f"Train major dice: {train_major_dice_avg:.6f} | Val major dice: {val_major_dice_avg:.6f} | "
                    f"Train small dice: {train_small_dice_avg} | Val small dice: {val_small_dice_avg:.6f} | "
                    f"{mask_log} | "
                    f"LR: {current_lr:.6e}"
                )

                # Check for improvement
                if val_rmse < best_val_coord_loss - min_delta:
                    best_val_coord_loss = val_rmse
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

                    mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
                    mlflow.log_metric("best_val_rmse", best_val_coord_loss, step=global_step)

                    logger.info(f"New best model saved! Val RMSE: {val_rmse:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                    torch.save(
                        obj={
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_rmse': val_rmse,
                        },
                        f=Path(model_weights_out_dir) / f"{global_step}.pt",
                    )

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered after {global_step} steps")
                    logger.info(f"Best validation MAE: {best_val_coord_loss:.6f}")
                    mlflow.log_metric("final_best_val_rmse", best_val_coord_loss)

                    return best_val_coord_loss

                model.train()

                iter_start_time = time.time()

        # End of epoch summary
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_coord_loss = sum(train_coord_losses) / len(train_coord_losses)
        if exclude_background_pixels:
            avg_mask_loss = sum(train_mask_losses) / len(train_mask_losses)

        mlflow.log_metrics(metrics={
                "epoch/train_coord_loss": avg_coord_loss,
            }, step=global_step)

        logger.info(f"\n{'=' * 60}")
        mask_loss_log = f"| Avg mask loss {avg_mask_loss:.6f}" if exclude_background_pixels else ""
        logger.info(f"Epoch {epoch}/{n_epochs} completed | Avg Train Loss: {avg_train_loss:.6f} | Avg coord loss {avg_coord_loss:.6f} {mask_loss_log}")
        logger.info(f"{'=' * 60}\n")


    logger.info(f"\nTraining completed! Best validation loss: {best_val_coord_loss:.6f}")

    mlflow.log_metric("final_best_val_coord_loss", best_val_coord_loss)

    return best_val_coord_loss
