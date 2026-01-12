import os
from pathlib import Path
from typing import ContextManager, Iterator, Optional
from contextlib import nullcontext

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import UNet
from torch import nn
import torch.nn.functional as F

from loguru import logger

from deep_ccf_registration.configs.train_config import LRScheduler
from deep_ccf_registration.datasets.slice_dataset_cache import ShuffledBatchIterator
from deep_ccf_registration.datasets.transforms import TemplateParameters, get_template_point_normalization_inverse
from deep_ccf_registration.utils.logging_utils import timed, ProgressLogger
from deep_ccf_registration.utils.visualization import viz_sample


class MaskedCoordLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # Handle mask shape: if pred is (B, C, H, W) and mask is (B, H, W), 
        # unsqueeze to (B, 1, H, W) before expanding to (B, C, H, W)
        if mask.dim() == pred.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(pred).bool()
        return F.smooth_l1_loss(pred[mask], target[mask])

class RMSE(nn.Module):
    """
    Computes root mean squared Euclidean distance between predicted and target points.
    """

    def __init__(self, coordinate_dim: int = 1):
        """
        Parameters
        ----------
        coordinate_dim : int
            The dimension containing coordinates (default=1 for shape B, C, H, W)
        """
        super().__init__()
        self.coordinate_dim = coordinate_dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute mean squared point distance loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted coordinates, shape (B, C, H, W)
        target : torch.Tensor
            Target coordinates, shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Scalar mean squared Euclidean distance
        """
        squared_errors = (pred - target) ** 2
        per_point_squared_distance = squared_errors.sum(dim=self.coordinate_dim)

        if mask is not None:
            if mask.dim() == per_point_squared_distance.dim() + 1:
                mask = mask.sum(dim=self.coordinate_dim)
            if mask.dim() != per_point_squared_distance.dim():
                raise ValueError("Mask must have same spatial dimensions as coordinates")

            mask = mask.to(per_point_squared_distance.device, per_point_squared_distance.dtype)
            squared_errors = per_point_squared_distance * mask
            valid_points = mask.sum().clamp(min=1.0)
            mse = squared_errors.sum() / valid_points
        else:
            mse = per_point_squared_distance.mean()

        return mse.sqrt()


def _evaluate(
    dataloader: Iterator,
    model: torch.nn.Module,
    coord_loss: MaskedCoordLoss,
    device: str,
    autocast_context: ContextManager,
    max_iters: int,
    denormalize_pred_template_points: bool,
    viz_sample_count: int = 10,
    ls_template_parameters: Optional[TemplateParameters] = None,
    ccf_annotations: Optional[np.ndarray] = None,
    global_step: Optional[int] = None,
    exclude_background_pixels: bool = False,
    is_debug: bool = False
) -> tuple[float, float, float]:
    """
    Evaluate model and report losses/RMSEs at multiple resolutions.

    Returns
    -------
    tuple[float, float, float]
        Validation loss, RMSE at the prediction resolution, and RMSE after
        upsampling predictions back to the raw template resolution.
    """
    model.eval()
    losses = []
    rmses_downsampled = []
    rmses_raw = []
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
            pad_mask_heights = batch["pad_mask_heights"].to(device)
            pad_mask_widths = batch["pad_mask_widths"].to(device)

            with autocast_context:
                pred = model(input_images)
                if pred.shape != target_template_points.shape:
                    # downsample the target to match the prediction resolution
                    target_template_points_downsample = F.interpolate(
                        target_template_points,
                        size=pred.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    pad_masks_downsample = _resize_pad_masks_from_sizes(
                        valid_heights=pad_mask_heights,
                        valid_widths=pad_mask_widths,
                        original_size=pad_masks.shape[-2:],
                        target_size=pred.shape[-2:],
                    )
                else:
                    target_template_points_downsample = target_template_points
                    pad_masks_downsample = pad_masks
                loss = coord_loss(pred=pred, target=target_template_points_downsample, mask=pad_masks_downsample)
                losses.append(loss.item())

            if denormalize_pred_template_points:
                pred = get_template_point_normalization_inverse(
                    x=pred,
                    template_parameters=ls_template_parameters,
                )
                target_template_points = get_template_point_normalization_inverse(
                    x=target_template_points,
                    template_parameters=ls_template_parameters,
                )
                target_template_points_downsample = get_template_point_normalization_inverse(
                    x=target_template_points_downsample,
                    template_parameters=ls_template_parameters,
                )

            pred_upsample = _resize_to_target(pred=pred, target_template_points=target_template_points)
            rmse_downsample = RMSE()(pred=pred, target=target_template_points_downsample, mask=pad_masks_downsample)
            rmse_raw = RMSE()(pred=pred_upsample, target=target_template_points, mask=pad_masks)

            rmses_downsampled.append(rmse_downsample.item())
            rmses_raw.append(rmse_raw.item())

            if enable_viz and len(collected_samples) < viz_sample_count:
                errors = (pred - target_template_points_downsample) ** 2
                slice_indices = batch["slice_indices"]
                patch_ys = batch["patch_ys"]
                patch_xs = batch["patch_xs"]
                subject_ids = batch["subject_ids"]

                remaining = viz_sample_count - len(collected_samples)
                take = min(remaining, input_images.shape[0])
                for sample_idx in range(take):
                    collected_samples.append({
                        "input_image": input_images[sample_idx].detach().cpu().squeeze(0),
                        "pred_coords": pred[sample_idx].detach().cpu(),
                        "gt_coords": target_template_points_downsample[sample_idx].detach().cpu(),
                        "pad_mask": pad_masks_downsample[sample_idx].detach().cpu(),
                        "errors": errors[sample_idx].detach().cpu().numpy(),
                        "slice_idx": int(slice_indices[sample_idx].item()),
                        "patch_y": int(patch_ys[sample_idx].item()),
                        "patch_x": int(patch_xs[sample_idx].item()),
                        "subject_id": subject_ids[sample_idx],
                    })
                    if len(collected_samples) >= viz_sample_count:
                        break

    val_loss = np.mean(losses)
    val_rmse_downsampled = np.mean(rmses_downsampled)
    val_rmse_raw = np.mean(rmses_raw)

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
                tissue_mask=None, # TODO
                exclude_background=exclude_background_pixels,
            )
            fig_filename = (
                f"subject_{sample['subject_id']}_slice_{sample['slice_idx']}"
                f"_y_{sample['patch_y']}_x_{sample['patch_x']}_step_{iteration}_viz_{idx}.png"
            )
            if is_debug:
                plt.show()
            mlflow.log_figure(
                fig,
                f"validation_samples/step_{iteration}/{fig_filename}"
            )
            plt.close(fig)

    model.train()
    return val_loss, val_rmse_downsampled, val_rmse_raw


def _resize_to_target(
    pred: torch.Tensor,
    target_template_points: torch.Tensor,
) -> torch.Tensor:
    """Ensure predictions share spatial size with targets."""
    target_size = target_template_points.shape[-2:]
    if pred.shape[-2:] == target_size:
        return pred

    pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
    return pred


def _resize_pad_masks_from_sizes(
    valid_heights: torch.Tensor,
    valid_widths: torch.Tensor,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Construct rectangular pad masks scaled to a new spatial resolution."""
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    if (orig_h, orig_w) == (target_h, target_w):
        raise ValueError("Pad mask resizing requested without a size change")

    device = valid_heights.device

    scaled_heights = torch.clamp(
        (valid_heights.float() * target_h / orig_h).ceil().long(),
        min=0,
        max=target_h,
    )
    scaled_widths = torch.clamp(
        (valid_widths.float() * target_w / orig_w).ceil().long(),
        min=0,
        max=target_w,
    )

    row_indices = torch.arange(target_h, device=device).unsqueeze(0)
    col_indices = torch.arange(target_w, device=device).unsqueeze(0)

    row_mask = row_indices < scaled_heights.unsqueeze(1)
    col_mask = col_indices < scaled_widths.unsqueeze(1)

    return row_mask.unsqueeze(2) & col_mask.unsqueeze(1)


def train(
        train_dataloader: ShuffledBatchIterator,
        val_dataloader: ShuffledBatchIterator,
        model: UNet,
        optimizer,
        max_iters: int,
        model_weights_out_dir: Path,
        ls_template_parameters: TemplateParameters,
        normalize_target_points: bool = True,
        learning_rate: float = 0.0001,
        eval_iters: int = 200,
        eval_interval: int = 500,
        patience: int = 10,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_debug: bool = False,
        log_interval: int = 20,
        ccf_annotations: Optional[np.ndarray] = None,
        val_viz_samples: int = 0,
        exclude_background_pixels: bool = False,
        lr_scheduler: Optional[LRScheduler] = None
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

    coord_loss = MaskedCoordLoss()
    best_val_rmse = float("inf")
    patience_counter = 0
    global_step = 0

    model.to(device)

    logger.info(f"Starting training for {max_iters} iters")
    logger.info(f"Device: {device}")

    if lr_scheduler == LRScheduler.ReduceLROnPlateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=100)
    else:
        scheduler = None

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
                loss = coord_loss(pred=model_out, target=target_template_points, mask=pad_masks)

            train_losses.append(loss.item())

            global_step += 1

            # Periodic evaluation
            if global_step % eval_interval == 0:
                logger.info(f"Evaluating at step {global_step}")
                val_loss, val_rmse_downsampled, val_rmse = _evaluate(
                    dataloader=val_dataloader,
                    model=model,
                    device=device,
                    autocast_context=autocast_context,
                    max_iters=1 if is_debug else eval_iters,
                    denormalize_pred_template_points=normalize_target_points,
                    viz_sample_count=val_viz_samples,
                    ls_template_parameters=ls_template_parameters,
                    ccf_annotations=ccf_annotations,
                    global_step=global_step,
                    exclude_background_pixels=exclude_background_pixels,
                    coord_loss=coord_loss,
                    is_debug=is_debug
                )

                current_lr = optimizer.param_groups[0]['lr']

                mlflow.log_metrics(
                    metrics={
                        "eval/loss": val_loss,
                        "eval/val_rmse": val_rmse,
                        "eval/val_rmse_downsampled": val_rmse_downsampled,
                    },
                    step=global_step
                )

                logger.info(
                    f"Step {global_step} | "
                    f"Train loss: {loss.item():.6f} | Val loss: {val_loss:.6f} | "
                    f"Val RMSE (downsampled): {val_rmse_downsampled:.6f} | Val RMSE (raw): {val_rmse:.6f} |"
                    f"LR: {current_lr:.6e}"
                )

                checkpoint_path = Path(model_weights_out_dir) / f"{global_step}.pt"
                torch.save(
                    obj={
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rmse': val_rmse,
                        'val_rmse_downsampled': val_rmse_downsampled,
                        'lr': scheduler.get_last_lr() if scheduler is not None else learning_rate
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

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                if lr_scheduler == LRScheduler.ReduceLROnPlateau:
                    scheduler.step(metrics=loss.item())
                else:
                    raise ValueError(f'{lr_scheduler} not supported')

            mlflow.log_metrics({
                "train/loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            }, step=global_step)

            if global_step == max_iters:
                logger.info(
                    f"\nTraining completed! Best validation RMSE: {best_val_rmse:.6f}")

                mlflow.log_metric("final_best_val_rmse", best_val_rmse)
                return best_val_rmse

            progress_logger.log_progress(other=f'loss={loss.item():.3f}')