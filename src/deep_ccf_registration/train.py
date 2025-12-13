from pathlib import Path
from typing import ContextManager, Any, Optional
from contextlib import nullcontext
import os

import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from monai.networks.nets import UNet
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from loguru import logger
import torch.nn.functional as F
from tqdm import tqdm

from deep_ccf_registration.inference import evaluate_batch
from deep_ccf_registration.utils.logging_utils import timed

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

def _evaluate_loss(
        model_out: torch.Tensor,
        exclude_background_pixels: bool,
        coord_loss: MSELoss,
        target_template_points: torch.Tensor,
        tissue_masks: torch.Tensor,
        pad_masks: torch.Tensor,
        input_image_transforms: dict[str, Any],
        predict_tissue_mask: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if predict_tissue_mask:
        pred = model_out[:, :-1]
    else:
        pred = model_out
    coordinate_loss = coord_loss(
        pred[torch.stack([tissue_masks] * 3, dim=1)] if exclude_background_pixels else pred[torch.stack([pad_masks] * 3, dim=1)],
        target_template_points[torch.stack([tissue_masks] * 3, dim=1)] if exclude_background_pixels else target_template_points[torch.stack([pad_masks] * 3, dim=1)],
    )

    loss = coordinate_loss

    if predict_tissue_mask:
        tissue_loss_per_pixel = F.binary_cross_entropy_with_logits(
            model_out[:, -1].cpu().float(),  # moving to cpu since bug with BCE and mps locally
            tissue_masks.cpu().float(),  # moving to cpu since bug with BCE and mps locally
            reduction='none'
        )
        tissue_loss = _mask_pad_pixels(
            tissue_loss_per_pixel=tissue_loss_per_pixel,
            input_image_transforms=input_image_transforms
        )
        loss += tissue_loss
    else:
        tissue_loss = None

    return coordinate_loss, tissue_loss, loss

def train(
        train_dataloader: DataLoader,
        train_eval_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: UNet,
        optimizer,
        max_iters: int,
        model_weights_out_dir: Path,
        ccf_annotations: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        learning_rate: float = 0.001,
        eval_iters: int = 200,
        decay_learning_rate: bool = True,
        eval_interval: int = 500,
        patience: int = 10,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        exclude_background_pixels: bool = True,
        predict_tissue_mask: bool = True,
        is_debug: bool = False,
):
    """
    Train slice registration model

    Parameters
    ----------
    model: Neural network model to train
    optimizer: Optimizer for training
    model_weights_out_dir: Directory to save model checkpoints
    learning_rate: Initial learning rate
    decay_learning_rate: Whether to decay learning rate during training
    eval_interval: Evaluate model every N iterations
    patience: Number of evaluations without improvement before stopping
    min_delta: Minimum change in validation loss to be considered improvement
    autocast_context: Context manager for mixed precision training
    device: Device to train on
    ccf_annotations: 25 micron resolution CCF annotation volume
    ls_template_parameters: ls template AntsImageParameters
    exclude_background_pixels: whether to use a tissue mask to exclude background pixels in loss/evaluation.
        Otherwise, just excludes pad pixels

    Returns
    -------
    Best validation loss achieved during training
    """
    os.makedirs(model_weights_out_dir, exist_ok=True)

    coord_loss = MSELoss()
    best_val_coord_loss = float("inf")
    patience_counter = 0
    global_step = 0
    min_lr = learning_rate / 10 # should be ~= learning_rate/10 per Chinchilla

    model.to(device)

    logger.info(f"Starting training for {max_iters} iters")
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Device: {device}")

    pbar = None
    pbar_postfix_entries: dict[str, Any] = {}

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_iters, eta_min=min_lr)

    while True:
        model.train()
        train_losses = []
        train_coord_losses = []
        train_mask_losses = []

        for batch in train_dataloader:
            if pbar is None:
                # start timing once first batch has been loaded
                pbar = tqdm(total=max_iters, desc="Training", smoothing=0)

            sampler = train_dataloader.sampler
            pbar_postfix_entries['subject_group'] = sampler.current_subject_batch_idx
            pbar.set_postfix(pbar_postfix_entries, refresh=False)
            input_images, target_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, tissue_masks, pad_masks, subject_ids = batch
            input_images, target_template_points, tissue_masks, pad_masks = input_images.to(device), target_template_points.to(device), tissue_masks.to(device), pad_masks.to(device)

            optimizer.zero_grad()
            with autocast_context:
                with timed():
                    model_out = model(input_images)
                coordinate_loss, tissue_loss, loss = _evaluate_loss(
                    model_out=model_out,
                    exclude_background_pixels=exclude_background_pixels,
                    coord_loss=coord_loss,
                    target_template_points=target_template_points,
                    tissue_masks=tissue_masks.bool(),
                    pad_masks=pad_masks.bool(),
                    input_image_transforms=input_image_transforms,
                    predict_tissue_mask=predict_tissue_mask,
                )

            loss.backward()
            optimizer.step()

            if decay_learning_rate:
                scheduler.step()

            train_losses.append(loss.item())
            train_coord_losses.append(coordinate_loss.item())
            if tissue_loss is not None:
                train_mask_losses.append(tissue_loss.item())

            global_step += 1

            mlflow.log_metrics({
                "train/coord_loss": coordinate_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            }, step=global_step)

            pbar_postfix_entries['loss'] = f"{loss.item():.6f}"
            pbar_postfix_entries['coord_loss'] = f"{coordinate_loss.item():.6f}"
            pbar.set_postfix(pbar_postfix_entries, refresh=False)
            pbar.update(1)


            # Periodic evaluation
            if global_step % eval_interval == 0:
                with torch.no_grad():
                    train_rmse, train_rmse_tissue_only, train_tissue_mask_dice = evaluate_batch(
                        dataloader=train_eval_dataloader,
                        model=model,
                        ccf_annotations=ccf_annotations,
                        ls_template_parameters=ls_template_parameters,
                        device=device,
                        iteration=global_step,
                        is_train=True,
                        autocast_context=autocast_context,
                        exclude_background_pixels=exclude_background_pixels,
                        predict_tissue_mask=predict_tissue_mask,
                        max_iters=1 if is_debug else eval_iters,
                    )
                    val_rmse, val_rmse_tissue_only, val_tissue_mask_dice = evaluate_batch(
                        dataloader=val_dataloader,
                        model=model,
                        ccf_annotations=ccf_annotations,
                        ls_template_parameters=ls_template_parameters,
                        device=device,
                        iteration=global_step,
                        is_train=False,
                        autocast_context=autocast_context,
                        exclude_background_pixels=exclude_background_pixels,
                        predict_tissue_mask=predict_tissue_mask,
                        max_iters=1 if is_debug else eval_iters,
                    )

                    current_lr = optimizer.param_groups[0]['lr']

                    mlflow.log_metrics(
                        metrics={
                            "eval/train_rmse": train_rmse,
                            "eval/val_rmse": val_rmse,
                    },
                        step=global_step
                    )

                    if predict_tissue_mask:
                        assert train_rmse_tissue_only is not None
                        assert val_rmse_tissue_only is not None
                        assert train_tissue_mask_dice is not None
                        assert val_tissue_mask_dice is not None
                        mlflow.log_metrics(metrics={
                            "eval/val_rmse_tissue_only": val_rmse_tissue_only,
                            "eval/train_tissue_mask_dice": train_tissue_mask_dice,
                            "eval/val_tissue_mask_dice": val_tissue_mask_dice,
                            "eval/train_rmse_tissue_only": train_rmse_tissue_only,
                        }, step=global_step)

                    if predict_tissue_mask:
                        mask_log = f"Train mask dice: {train_tissue_mask_dice} | "
                        f"Val mask dice: {val_tissue_mask_dice} | "
                        f"Train RMSE tissue only: {train_rmse_tissue_only:.6f} microns | Val RMSE tissue only: {val_rmse_tissue_only:.6f} microns | "
                    else:
                        mask_log = ""
                    logger.info(
                        f"Step {global_step} | "
                        f"Train RMSE: {train_rmse:.6f} microns | Val RMSE: {val_rmse:.6f} microns | "
                        f"{mask_log} | "
                        f"LR: {current_lr:.6e}"
                    )

                    checkpoint_path = Path(model_weights_out_dir) / f"{global_step}.pt"
                    torch.save(
                        obj={
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_rmse': val_rmse,
                            'lr': scheduler.get_lr()
                        },
                        f=checkpoint_path,
                    )

                    # Check for improvement
                    if val_rmse < best_val_coord_loss - min_delta:
                        best_val_coord_loss = val_rmse
                        patience_counter = 0

                        mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
                        mlflow.log_metric("best_val_rmse", best_val_coord_loss, step=global_step)

                        logger.info(f"New best model saved! Val RMSE: {val_rmse:.6f}")
                    else:
                        patience_counter += 1
                        logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                    # Early stopping
                    if patience_counter >= patience:
                        logger.info(f"\nEarly stopping triggered after {global_step} steps")
                        logger.info(f"Best validation MAE: {best_val_coord_loss:.6f}")
                        mlflow.log_metric("final_best_val_rmse", best_val_coord_loss)

                        return best_val_coord_loss

                    model.train()

            if global_step == max_iters:
                logger.info(
                    f"\nTraining completed! Best validation loss: {best_val_coord_loss:.6f}")

                mlflow.log_metric("final_best_val_coord_loss", best_val_coord_loss)
                return best_val_coord_loss
