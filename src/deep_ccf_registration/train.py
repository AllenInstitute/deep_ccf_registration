import os
import tempfile
from pathlib import Path
from typing import ContextManager, Iterator, Optional, Any
from contextlib import nullcontext

import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch import nn
import torch.nn.functional as F

from loguru import logger


def is_main_process() -> bool:
    """Check if this is the main process (rank 0) for logging/mlflow."""
    return int(os.environ.get('RANK', 0)) == 0

from deep_ccf_registration.configs.train_config import LRScheduler, TrainConfig
from deep_ccf_registration.datasets.transforms import get_template_point_normalization_inverse
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.utils.logging_utils import timed, ProgressLogger
from deep_ccf_registration.utils.visualization import viz_sample


class MSE(nn.Module):
    """
    Computes root mean squared Euclidean distance between predicted and target points.
    """

    def __init__(self, coordinate_dim: int = 1, reduction: Optional[str] = None):
        """
        Parameters
        ----------
        coordinate_dim : int
            The dimension containing coordinates (default=1 for shape B, C, H, W)
        """
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self._reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted coordinates, shape (B, C, H, W)
        target : torch.Tensor
            Target coordinates, shape (B, C, H, W)
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
            valid_points = mask.sum(dim=(1, 2)).clamp(min=1.0)
            mse = squared_errors.sum(dim=(1, 2)) / valid_points
        else:
            mse = per_point_squared_distance.mean(dim=(1, 2))

        if self._reduction == 'mean':
            mse = mse.mean()
        return mse


def _evaluate(
    dataloader: Iterator,
    model: torch.nn.Module,
    coord_loss: MSE,
    device: str,
    autocast_context: ContextManager,
    max_iters: int,
    denormalize_pred_template_points: bool,
    global_step: int,
    viz_sample_count: int = 10,
    ls_template_parameters: Optional[TemplateParameters] = None,
    ccf_annotations: Optional[np.ndarray] = None,
    exclude_background_pixels: bool = False,
    predict_tissue_mask: bool = False,
    is_debug: bool = False,
    tissue_mask_weight: float = 0.1
) -> dict[str, Any]:
    """
    Evaluate model and report losses/RMSEs.
    """
    model.eval()
    losses = []
    point_losses = []
    tissue_mask_losses = []
    rmses = []
    registration_res_rmses = []
    collected_samples = []
    eval_records = []
    enable_viz = (
        viz_sample_count > 0
        and ls_template_parameters is not None
        and ccf_annotations is not None
    )
    tissue_mask_dice_metric = DiceMetric(num_classes=2, include_background=False)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_iters:
                break

            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)
            pad_masks = batch["pad_masks"].to(device)

            if predict_tissue_mask:
                tissue_masks = batch["tissue_masks"].to(device)
            else:
                tissue_masks = None

            with autocast_context:
                pred = model(input_images)
                if predict_tissue_mask:
                    pred_points, pred_tissue_mask_logits = pred[:, :-1], pred[:, -1]
                    pred_tissue_mask = F.sigmoid(pred_tissue_mask_logits) > 0.5
                else:
                    pred_points = pred
                    pred_tissue_mask_logits = None
                    pred_tissue_mask = None

                if predict_tissue_mask:
                    masks = tissue_masks
                else:
                    masks = pad_masks

                point_loss = coord_loss(pred=pred_points, target=target_template_points, mask=masks)

                if predict_tissue_mask:
                    tissue_mask_loss = F.binary_cross_entropy_with_logits(pred_tissue_mask_logits, tissue_masks)
                    loss = point_loss + tissue_mask_weight * tissue_mask_loss
                else:
                    loss = point_loss
                    tissue_mask_loss = None

                losses.append(loss.item())
                point_losses.append(point_loss.item())

                if predict_tissue_mask:
                    tissue_mask_losses.append(tissue_mask_loss.item())

            if denormalize_pred_template_points:
                pred_points = get_template_point_normalization_inverse(
                    x=pred_points,
                    template_parameters=ls_template_parameters,
                )
                target_template_points = get_template_point_normalization_inverse(
                    x=target_template_points,
                    template_parameters=ls_template_parameters,
                )

            rmse = MSE()(pred=pred_points, target=target_template_points, mask=masks).sqrt()

            if "eval_target_template_points" in batch:
                eval_target = batch["eval_target_template_points"].to(device)
                eval_pad_masks = batch["eval_pad_masks"].to(device)
                if predict_tissue_mask:
                    eval_tissue_masks = batch["eval_tissue_masks"].to(device)
                heights = batch["pad_mask_heights"]
                widths = batch["pad_mask_widths"]

                for si in range(pred_points.shape[0]):
                    # Crop pred to content region (remove pad_dim padding)
                    h, w = int(heights[si]), int(widths[si])
                    pred_crop = pred_points[si:si+1, :, :h, :w]

                    # Find eval content extent from eval_pad_masks
                    eval_mask_i = eval_pad_masks[si]
                    eval_h = int(eval_mask_i.any(dim=1).sum())
                    eval_w = int(eval_mask_i.any(dim=0).sum())
                    eval_target_crop = eval_target[si:si+1, :, :eval_h, :eval_w]

                    # Upsample cropped pred to eval content size
                    pred_up = F.interpolate(
                        pred_crop, size=(eval_h, eval_w),
                        mode="bilinear", align_corners=False
                    )

                    # Select mask
                    if predict_tissue_mask:
                        mask_crop = eval_tissue_masks[si:si+1, :eval_h, :eval_w]
                    else:
                        mask_crop = eval_pad_masks[si:si+1, :eval_h, :eval_w]

                    rmse_val = MSE()(pred=pred_up, target=eval_target_crop, mask=mask_crop).sqrt()
                    registration_res_rmses += rmse_val.cpu().tolist()

            if predict_tissue_mask:
                tissue_mask_dice_metric(y_pred=pred_tissue_mask.unsqueeze(1), y=tissue_masks.unsqueeze(1))

            rmses += rmse.cpu().tolist()

            # Collect per-sample metrics for CSV logging
            batch_size = input_images.shape[0]
            for sample_idx in range(batch_size):
                record_idx = len(eval_records)
                eval_records.append({
                    'subject_id': batch["subject_ids"][sample_idx],
                    'slice_idx': int(batch["slice_indices"][sample_idx].item()),
                    'patch_y': int(batch["patch_ys"][sample_idx].item()),
                    'patch_x': int(batch["patch_xs"][sample_idx].item()),
                    'rmse': rmses[record_idx],
                })

            if enable_viz and len(collected_samples) < viz_sample_count:
                errors = (pred_points - target_template_points) ** 2
                slice_indices = batch["slice_indices"]
                patch_ys = batch["patch_ys"]
                patch_xs = batch["patch_xs"]
                subject_ids = batch["subject_ids"]

                remaining = viz_sample_count - len(collected_samples)
                take = min(remaining, input_images.shape[0])
                for sample_idx in range(take):
                    collected_samples.append({
                        "input_image": input_images[sample_idx].squeeze().detach().cpu(),
                        "pred_coords": pred_points[sample_idx].detach().cpu(),
                        "gt_coords": target_template_points[sample_idx].detach().cpu(),
                        "pad_mask": pad_masks[sample_idx].detach().cpu(),
                        "tissue_mask": tissue_masks[sample_idx].detach().cpu() if predict_tissue_mask else None,
                        "pred_tissue_masks": F.sigmoid(pred_tissue_mask_logits[sample_idx]).detach().cpu() if predict_tissue_mask else None,
                        "errors": errors[sample_idx].detach().cpu().numpy(),
                        "slice_idx": int(slice_indices[sample_idx].item()),
                        "patch_y": int(patch_ys[sample_idx].item()),
                        "patch_x": int(patch_xs[sample_idx].item()),
                        "subject_id": subject_ids[sample_idx],
                    })
                    if len(collected_samples) >= viz_sample_count:
                        break

    val_loss = np.mean(losses)
    val_rmse = np.mean(rmses)
    val_rmse_registration_res = np.mean(registration_res_rmses) if registration_res_rmses else None

    if predict_tissue_mask:
        val_point_loss = np.mean(point_losses)
        val_tissue_mask_loss = np.mean(tissue_mask_losses)
        val_tissue_mask_dice = np.mean(tissue_mask_dice_metric.aggregate().item())
    else:
        val_point_loss = val_loss
        val_tissue_mask_dice = None
        val_tissue_mask_loss = None

    if enable_viz and collected_samples:
        iteration = global_step or 0
        for idx, sample in enumerate(collected_samples):
            fig = viz_sample(
                predicted_template_points=sample["pred_coords"].moveaxis(0, -1).view(-1, 3).cpu().numpy(),
                predicted_tissue_masks=sample['pred_tissue_masks'].cpu().numpy() if predict_tissue_mask else None,
                gt_template_points=sample["gt_coords"].moveaxis(0, -1).view(-1, 3).cpu().numpy(),
                ls_template_info=ls_template_parameters,
                input_image=sample["input_image"].cpu().numpy(),
                tissue_mask=sample['tissue_mask'].cpu().numpy() if predict_tissue_mask else None,
                template_parameters=ls_template_parameters,
                predict_tissue_mask=predict_tissue_mask,
                pad_mask=sample['pad_mask'].cpu().numpy(),
            )
            fig_filename = (
                f"subject_{sample['subject_id']}_slice_{sample['slice_idx']}"
                f"_y_{sample['patch_y']}_x_{sample['patch_x']}_step_{iteration}_viz_{idx}.png"
            )
            if is_debug:
                plt.show()
            if is_main_process():
                mlflow.log_figure(
                    fig,
                    f"validation_samples/step_{iteration}/{fig_filename}"
                )
            plt.close(fig)

    # Log per-sample metrics as CSV (only on main process)
    if is_main_process():
        metrics_df = pd.DataFrame(eval_records)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            metrics_df.to_csv(f.name, index=False)
            csv_path = f.name
            mlflow.log_artifact(csv_path, artifact_path=f"eval_metrics/step_{global_step}")

    model.train()

    return {
        "val_point_loss": val_point_loss,
        "val_tissue_mask_loss": val_tissue_mask_loss,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "val_rmse_registration_res": val_rmse_registration_res,
        "val_tissue_mask_dice": val_tissue_mask_dice,
    }


def _resize_to_target(
    x: torch.Tensor,
    target_template_points: torch.Tensor,
) -> torch.Tensor:
    """Ensure predictions share spatial size with targets."""
    target_size = target_template_points.shape[-2:]
    if x.shape[-2:] == target_size:
        return x

    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    return x


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
        train_dataloader: Iterator,
        val_dataloader: Iterator,
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
        predict_tissue_mask: bool = False,
        lr_scheduler: Optional[LRScheduler] = None,
        tissue_mask_loss_weight: float = 0.1,
        gradient_accumulation_steps: int = 1,
        # Resume parameters for checkpoint recovery
        start_step: int = 0,
        start_best_val_loss: float = float("inf"),
        start_patience_counter: int = 0,
        scheduler_state_dict: Optional[dict] = None,
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
    gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step

    Returns
    -------
    Best validation loss achieved during training
    """
    os.makedirs(model_weights_out_dir, exist_ok=True)

    calc_coord_loss = MSE(reduction='mean')
    best_val_loss = start_best_val_loss
    patience_counter = start_patience_counter
    global_step = start_step
    accumulation_step = 0

    model.to(device)

    if start_step > 0:
        logger.info(f"Resuming training from step {start_step}")
        logger.info(f"Best val loss so far: {start_best_val_loss:.6f}, patience: {start_patience_counter}")
    logger.info(f"Training for {max_iters} iters total")
    logger.info(f"Device: {device}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    if lr_scheduler == LRScheduler.ReduceLROnPlateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=100)
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
            logger.info("Restored scheduler state from checkpoint")
    else:
        scheduler = None

    progress_logger = None

    while True:
        model.train()
        losses = []
        point_losses = []
        tissue_mask_losses = []

        for batch in train_dataloader:
            if progress_logger is None and is_main_process():
                progress_logger = ProgressLogger(desc='Training', total=max_iters, log_every=log_interval)

            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)
            pad_masks = batch["pad_masks"].to(device)

            if predict_tissue_mask:
                tissue_masks = batch["tissue_masks"].to(device)
            else:
                tissue_masks = None

            with autocast_context:
                with timed():
                    model_out = model(input_images)
                    if predict_tissue_mask:
                        pred_template_points, pred_tissue_masks = model_out[:, :-1], model_out[:, -1]
                        mask = tissue_masks
                        tissue_mask_loss = F.binary_cross_entropy_with_logits(pred_tissue_masks,
                                                                              tissue_masks)

                    else:
                        pred_template_points = model_out
                        mask = pad_masks
                        tissue_mask_loss = None
                point_loss = calc_coord_loss(pred=pred_template_points, target=target_template_points, mask=mask)
                if predict_tissue_mask:
                    loss = point_loss + tissue_mask_loss_weight * tissue_mask_loss
                else:
                    loss = point_loss

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            loss.backward()
            accumulation_step += 1

            # Unscale for logging
            point_losses.append(point_loss.item())
            if predict_tissue_mask:
                tissue_mask_losses.append(tissue_mask_loss.item())
            losses.append(loss.item() * gradient_accumulation_steps)

            # Only step optimizer after accumulating enough gradients
            if accumulation_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if scheduler is not None:
                    if lr_scheduler == LRScheduler.ReduceLROnPlateau:
                        scheduler.step(metrics=point_loss.item())
                    else:
                        raise ValueError(f'{lr_scheduler} not supported')

                train_metrics = {
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/point_loss": point_loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                }
                if predict_tissue_mask:
                    train_metrics['train/tissue_mask_loss'] = tissue_mask_loss.item()

                if is_main_process():
                    mlflow.log_metrics(train_metrics, step=global_step)

                if is_main_process():
                    log_msg = f'loss={loss.item() * gradient_accumulation_steps:.3f}'
                    if predict_tissue_mask:
                        log_msg += f'; point_loss={point_loss.item():.3f}; tissue_mask_loss={tissue_mask_loss.item():.3f}'
                    progress_logger.log_progress(other=log_msg)

                # Periodic evaluation
                if global_step % eval_interval == 0:
                    logger.info(f"Evaluating at step {global_step}")
                    if is_debug:
                        # evaluate train too
                        _evaluate(
                            dataloader=train_dataloader,
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
                            coord_loss=calc_coord_loss,
                            is_debug=is_debug,
                            predict_tissue_mask=predict_tissue_mask,
                        )
                    val_metrics = _evaluate(
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
                        coord_loss=calc_coord_loss,
                        is_debug=is_debug,
                        predict_tissue_mask=predict_tissue_mask,
                    )

                    current_lr = optimizer.param_groups[0]['lr']

                    metrics = {
                        "eval/loss": val_metrics["val_loss"],
                        "eval/point_loss": val_metrics['val_point_loss'],
                        "eval/val_rmse": val_metrics["val_rmse"],
                        "eval/val_rmse_registration_res": val_metrics["val_rmse_registration_res"]
                    }
                    if predict_tissue_mask:
                        metrics = {
                            **metrics,
                            "eval/tissue_mask_loss": val_metrics['val_tissue_mask_loss'],
                            "eval/tissue_mask_dice": val_metrics['val_tissue_mask_dice'],
                        }
                    if is_main_process():
                        mlflow.log_metrics(
                            metrics=metrics,
                            step=global_step
                        )

                    log_msg = f"Step {global_step} | " \
                              f"Train loss: {loss.item() * gradient_accumulation_steps:.6f} | Val loss: {val_metrics['val_loss']:.6f} | " \
                              f"Val RMSE (downsampled): {val_metrics['val_rmse']:.6f} | Val RMSE (raw): {val_metrics['val_rmse_registration_res']:.6f}| " \
                              f"LR: {current_lr:.6e}"

                    if predict_tissue_mask:
                        log_msg += f' | Train point loss: {point_loss.item():.6f} | val point loss: {val_metrics["val_point_loss"]:.6f} | Val tissue mask dice: {val_metrics["val_tissue_mask_dice"]:.6f}'

                    logger.info(log_msg)

                    checkpoint_path = Path(model_weights_out_dir) / f"{global_step}.pt"
                    torch.save(
                        obj={
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'best_val_loss': best_val_loss,
                            'patience_counter': patience_counter,
                            'val_rmse': val_metrics['val_rmse'],
                            'lr': scheduler.get_last_lr() if scheduler is not None else learning_rate
                        },
                        f=checkpoint_path,
                    )

                    # Check for improvement
                    if val_metrics['val_loss'] < best_val_loss - min_delta:
                        best_val_loss = val_metrics['val_loss']
                        patience_counter = 0

                        if is_main_process():
                            mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
                            mlflow.log_metric("best_val_loss", best_val_loss, step=global_step)

                        logger.info(f"New best model saved! Val loss: {best_val_loss:.6f}")
                    else:
                        patience_counter += 1
                        logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                    # Early stopping
                    if patience_counter >= patience:
                        logger.info(f"\nEarly stopping triggered after {global_step} steps")
                        logger.info(f"Best validation RMSE: {best_val_loss:.6f}")
                        if is_main_process():
                            mlflow.log_metric("final_best_val_rmse", best_val_loss)

                        return best_val_loss

                    # Reset train losses for next eval period
                    point_losses = []
                    losses = []
                    tissue_mask_losses = []
                    model.train()

                if global_step == max_iters:
                    logger.info(
                        f"\nTraining completed! Best validation RMSE: {best_val_loss:.6f}")

                    if is_main_process():
                        mlflow.log_metric("final_best_val_rmse", best_val_loss)
                    return best_val_loss