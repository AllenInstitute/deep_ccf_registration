import os
from pathlib import Path
from typing import ContextManager, Iterator
from contextlib import nullcontext

import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from monai.networks.nets import UNet
from torch.nn import MSELoss
from loguru import logger
from torch.utils.data import DataLoader

from deep_ccf_registration.datasets.slice_dataset_cache import ShuffledBatchIterator
from deep_ccf_registration.utils.logging_utils import timed, ProgressLogger


def _evaluate_loss(
        model_out: torch.Tensor,
        coord_loss: MSELoss,
        target_template_points: torch.Tensor,
) -> torch.Tensor:
    """Compute coordinate regression loss."""
    coordinate_loss = coord_loss(model_out, target_template_points)
    return coordinate_loss


def _evaluate_on_dataloader(
        dataloader: Iterator,
        model: torch.nn.Module,
        coord_loss: MSELoss,
        device: str,
        autocast_context: ContextManager,
        max_iters: int,
) -> float:
    """Evaluate model on a dataloader, returning average loss."""
    model.eval()
    losses = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_iters:
                break

            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)

            with autocast_context:
                model_out = model(input_images)
                loss = _evaluate_loss(
                    model_out=model_out,
                    coord_loss=coord_loss,
                    target_template_points=target_template_points,
                )
            losses.append(loss.item())

    return np.mean(losses) if losses else 0.0

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
        log_interval: int = 20
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

            optimizer.zero_grad()
            with autocast_context:
                with timed():
                    model_out = model(input_images)
                loss = _evaluate_loss(
                    model_out=model_out,
                    coord_loss=coord_loss,
                    target_template_points=target_template_points,
                )

            loss.backward()
            optimizer.step()

            if decay_learning_rate:
                scheduler.step()

            train_losses.append(loss.item())

            global_step += 1

            mlflow.log_metrics({
                "train/coord_loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            }, step=global_step)

            # Periodic evaluation
            if global_step % eval_interval == 0:
                val_loss = _evaluate_on_dataloader(
                    dataloader=val_dataloader,
                    model=model,
                    coord_loss=coord_loss,
                    device=device,
                    autocast_context=autocast_context,
                    max_iters=1 if is_debug else eval_iters,
                )

                current_lr = optimizer.param_groups[0]['lr']
                avg_train_loss = np.mean(train_losses) if train_losses else 0.0

                mlflow.log_metrics(
                    metrics={
                        "eval/train_loss": avg_train_loss,
                        "eval/val_loss": val_loss,
                    },
                    step=global_step
                )

                logger.info(
                    f"Step {global_step} | "
                    f"Train loss: {avg_train_loss:.6f} | Val loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.6e}"
                )

                checkpoint_path = Path(model_weights_out_dir) / f"{global_step}.pt"
                torch.save(
                    obj={
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'lr': scheduler.get_last_lr()
                    },
                    f=checkpoint_path,
                )

                # Check for improvement
                if val_loss < best_val_coord_loss - min_delta:
                    best_val_coord_loss = val_loss
                    patience_counter = 0

                    mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
                    mlflow.log_metric("best_val_loss", best_val_coord_loss, step=global_step)

                    logger.info(f"New best model saved! Val loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered after {global_step} steps")
                    logger.info(f"Best validation loss: {best_val_coord_loss:.6f}")
                    mlflow.log_metric("final_best_val_loss", best_val_coord_loss)

                    return best_val_coord_loss

                # Reset train losses for next eval period
                train_losses = []
                model.train()

            if global_step == max_iters:
                logger.info(
                    f"\nTraining completed! Best validation loss: {best_val_coord_loss:.6f}")

                mlflow.log_metric("final_best_val_loss", best_val_coord_loss)
                return best_val_coord_loss

            progress_logger.log_progress(other=f'loss={loss.item():.3f}')