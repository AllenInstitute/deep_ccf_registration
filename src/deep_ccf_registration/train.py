import os
from pathlib import Path
from typing import ContextManager, Iterator, Optional
from contextlib import nullcontext

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import UNet
from deep_ccf_registration.losses import PointMSELoss
from loguru import logger

from deep_ccf_registration.datasets.slice_dataset_cache import ShuffledBatchIterator
from deep_ccf_registration.datasets.transforms import TemplatePointsNormalization, \
    TemplateParameters
from deep_ccf_registration.utils.logging_utils import timed, ProgressLogger
from deep_ccf_registration.inference import viz_sample


def _evaluate(
    dataloader: Iterator,
    model: torch.nn.Module,
    coord_loss: PointMSELoss,
    device: str,
    autocast_context: ContextManager,
    max_iters: int,
    template_points_normalizer: Optional[TemplatePointsNormalization] = None,
    viz_sample_count: int = 10,
    ls_template_parameters: Optional[TemplateParameters] = None,
    ccf_annotations: Optional[np.ndarray] = None,
    global_step: Optional[int] = None,
    exclude_background_pixels: bool = False,
) -> tuple[float, float]:
    """Evaluate model"""
    model.eval()
    losses = []
    losses_denorm = []
    collected_samples = []
    enable_viz = (
        viz_sample_count > 0
        and ls_template_parameters is not None
        and ccf_annotations is not None
    )

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_iters:
                break

            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)
            pad_masks = batch["pad_masks"].to(device)

            with autocast_context:
                model_out = model(input_images)
                loss = coord_loss(model_out, target_template_points, pad_masks)
                losses.append(loss.item())

                # denormalize if normalizer is provided
                if template_points_normalizer is not None:
                    model_out_denorm = template_points_normalizer.inverse(model_out)
                    target_template_points_denorm = template_points_normalizer.inverse(target_template_points)
                    loss_denorm = coord_loss(model_out_denorm, target_template_points_denorm, pad_masks)
                    losses_denorm.append(loss_denorm.item())

            if enable_viz and len(collected_samples) < viz_sample_count:
                errors = (model_out - target_template_points) ** 2
                slice_indices = batch["slice_indices"]
                patch_ys = batch["patch_ys"]
                patch_xs = batch["patch_xs"]
                subject_ids = batch["subject_ids"]

                remaining = viz_sample_count - len(collected_samples)
                take = min(remaining, input_images.shape[0])
                for sample_idx in range(take):
                    collected_samples.append({
                        "input_image": input_images[sample_idx].detach().cpu().squeeze(0),
                        "pred_coords": model_out[sample_idx].detach().cpu(),
                        "gt_coords": target_template_points[sample_idx].detach().cpu(),
                        "pad_mask": pad_masks[sample_idx].detach().cpu(),
                        "errors": errors[sample_idx].detach().cpu().numpy(),
                        "slice_idx": int(slice_indices[sample_idx].item()),
                        "patch_y": int(patch_ys[sample_idx].item()),
                        "patch_x": int(patch_xs[sample_idx].item()),
                        "subject_id": subject_ids[sample_idx],
                    })
                    if len(collected_samples) >= viz_sample_count:
                        break


        if losses_denorm:
            val_loss = np.mean(losses)
            val_rmse = np.sqrt(np.mean(losses_denorm))
        else:
            val_loss = np.mean(losses)
            val_rmse = np.sqrt(np.mean(losses))

        if enable_viz and collected_samples:
            iteration = global_step or 0
            for idx, sample in enumerate(collected_samples):
                fig = viz_sample(
                    ls_template_parameters=ls_template_parameters,
                    pred_coords=sample["pred_coords"],
                    gt_coords=sample["gt_coords"],
                    ccf_annotations=ccf_annotations,
                    iteration=iteration,
                    pad_mask=sample["pad_mask"],
                    input_image=sample["input_image"],
                    slice_idx=sample["slice_idx"],
                    errors=sample["errors"],
                    pred_tissue_mask=None,
                    tissue_mask=sample["pad_mask"],
                    exclude_background=exclude_background_pixels,
                )
                fig_filename = (
                    f"subject_{sample['subject_id']}_slice_{sample['slice_idx']}"
                    f"_y_{sample['patch_y']}_x_{sample['patch_x']}_step_{iteration}_viz_{idx}.png"
                )
                mlflow.log_figure(
                    fig,
                    f"validation_samples/step_{iteration}/{fig_filename}"
                )
                plt.close(fig)

        return val_loss, val_rmse

def train(
        train_dataloader: ShuffledBatchIterator,
        val_dataloader: ShuffledBatchIterator,
        model: UNet,
        optimizer,
        max_iters: int,
        model_weights_out_dir: Path,
        learning_rate: float = 0.001,
        eval_iters: int = 200,
        decay_learning_rate: bool = True,
        eval_interval: int = 500,
        patience: int = 10,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_debug: bool = False,
        log_interval: int = 20,
        template_points_normalizer: Optional[TemplatePointsNormalization] = None,
        ls_template_parameters: Optional[TemplateParameters] = None,
        ccf_annotations: Optional[np.ndarray] = None,
        val_viz_samples: int = 0,
        exclude_background_pixels: bool = False,
):
    """
    Train slice registration model

    Parameters
    ----------
    train_dataloader: Iterator yielding batch dicts with keys:
        input_images, target_template_points, dataset_indices, slice_indices,
        patch_ys, patch_xs, orientations, subject_ids
    train_eval_dataloader: Iterator for train evaluation
    val_dataloader: Iterator for validation
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
    val_viz_samples: number of validation samples to visualize each evaluation
    exclude_background_pixels: whether to zero-out background pixels in visualizations

    Returns
    -------
    Best validation loss achieved during training
    """
    os.makedirs(model_weights_out_dir, exist_ok=True)

    coord_loss = PointMSELoss()
    best_val_rmse = float("inf")
    patience_counter = 0
    global_step = 0
    min_lr = learning_rate / 10 # should be ~= learning_rate/10 per Chinchilla

    model.to(device)

    logger.info(f"Starting training for {max_iters} iters")
    logger.info(f"Device: {device}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_iters, eta_min=min_lr)

    progress_logger = None

    while True:
        model.train()
        train_losses = []

        for batch in train_dataloader:
            if progress_logger is None:
                progress_logger = ProgressLogger(desc='Training', total=max_iters, log_every=log_interval)

            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)
            pad_masks = batch["pad_masks"].to(device)

            optimizer.zero_grad()
            with autocast_context:
                with timed():
                    model_out = model(input_images)
                loss = coord_loss(model_out, target_template_points, pad_masks)

            loss.backward()
            optimizer.step()

            if decay_learning_rate:
                scheduler.step()

            train_losses.append(loss.item())

            global_step += 1

            mlflow.log_metrics({
                "train/loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            }, step=global_step)

            # Periodic evaluation
            if global_step % eval_interval == 0:
                logger.info(f"Evaluating at step {global_step}")
                val_loss, val_rmse = _evaluate(
                    dataloader=val_dataloader,
                    model=model,
                    coord_loss=coord_loss,
                    device=device,
                    autocast_context=autocast_context,
                    max_iters=1 if is_debug else eval_iters,
                    template_points_normalizer=template_points_normalizer,
                    viz_sample_count=val_viz_samples,
                    ls_template_parameters=ls_template_parameters,
                    ccf_annotations=ccf_annotations,
                    global_step=global_step,
                    exclude_background_pixels=exclude_background_pixels,
                )

                current_lr = optimizer.param_groups[0]['lr']

                mlflow.log_metrics(
                    metrics={
                        "eval/loss": val_loss,
                        "eval/val_rmse": val_rmse
                    },
                    step=global_step
                )

                logger.info(
                    f"Step {global_step} | "
                    f"Train loss: {loss.item():.6f} | Val loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} |"
                    f"LR: {current_lr:.6e}"
                )

                checkpoint_path = Path(model_weights_out_dir) / f"{global_step}.pt"
                torch.save(
                    obj={
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rmse': val_rmse,
                        'lr': scheduler.get_last_lr()
                    },
                    f=checkpoint_path,
                )

                # Check for improvement
                if val_rmse < best_val_rmse - min_delta:
                    best_val_rmse = val_rmse
                    patience_counter = 0

                    mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
                    mlflow.log_metric("best_val_rmse", best_val_rmse, step=global_step)

                    logger.info(f"New best model saved! Val RMSE: {val_rmse:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered after {global_step} steps")
                    logger.info(f"Best validation RMSE: {best_val_rmse:.6f}")
                    mlflow.log_metric("final_best_val_rmse", best_val_rmse)

                    return best_val_rmse

                # Reset train losses for next eval period
                train_losses = []
                model.train()

            if global_step == max_iters:
                logger.info(
                    f"\nTraining completed! Best validation RMSE: {best_val_rmse:.6f}")

                mlflow.log_metric("final_best_val_rmse", best_val_rmse)
                return best_val_rmse

            progress_logger.log_progress(other=f'loss={loss.item():.3f}')