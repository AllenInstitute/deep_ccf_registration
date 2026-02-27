import os
from pathlib import Path
from typing import ContextManager, Iterator, Optional
from contextlib import nullcontext

import mlflow
import numpy as np
import torch
import torch.distributed as dist
from segmentation_models_pytorch import Unet
import torch.nn.functional as F

from loguru import logger

from deep_ccf_registration.utils.ddp import is_main_process, get_local_rank, reduce_mean
from deep_ccf_registration.utils.evaluation import evaluate

from deep_ccf_registration.configs.train_config import LRScheduler
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.utils.logging_utils import timed, ProgressLogger
from deep_ccf_registration.utils.losses import calc_multi_task_loss, calc_spatial_gradient_loss, \
    DynamicWeightAverageScheduler, MSE


def train(
        train_dataloader,
        val_dataloader,
        model: Unet,
        optimizer,
        max_epochs: int,
        model_weights_out_dir: Path,
        ccf_annotations: np.ndarray,
        ls_template_parameters: TemplateParameters,
        terminology_path: Path,
        save_every: int = 3000,
        normalize_target_points: bool = True,
        learning_rate: float = 0.0001,
        eval_iters: int = 200,
        eval_interval: int = 500,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_debug: bool = False,
        log_interval: int = 20,
        val_viz_samples: int = 0,
        predict_tissue_mask: bool = False,
        lr_scheduler: Optional[LRScheduler] = None,
        cosine_warm_restarts_T_0: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        grad_clip_max_norm: Optional[float] = 1.0,
        warmup_steps: int = 0,
        start_epoch: int = 0,
        start_best_val_loss: float = float("inf"),
        scheduler_state_dict: Optional[dict] = None,
        train_sampler = None,
        multi_task_loss_init_weights: Optional[tuple[float]] = None,
        dwa_state_dict: Optional[dict] = None,
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
    min_delta: Minimum change in validation loss to be considered improvement
    autocast_context: Context manager for mixed precision training
    device: Device to train on
    ccf_annotations: 25 micron resolution CCF annotation volume
    ls_template_parameters: ls template AntsImageParameters
    val_viz_samples: number of validation samples to visualize each evaluation
    exclude_background_pixels: whether to zero-out background pixels in visualizations
    gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
    grad_clip_max_norm: Maximum gradient norm for clipping. Set to None to disable.
    warmup_steps: Number of steps to linearly warmup learning rate from 0 to learning_rate.
    save_every: Save checkpoint every n iterations

    Returns
    -------
    Best validation loss achieved during training
    """
    os.makedirs(model_weights_out_dir, exist_ok=True)

    calc_coord_loss = MSE(reduction='mean', template_parameters=ls_template_parameters)
    num_tasks = 2
    if predict_tissue_mask:
        num_tasks += 1

    if multi_task_loss_init_weights is not None:
        if len(multi_task_loss_init_weights) != num_tasks:
            raise ValueError(f'expected multi_task_loss_init_weights to have {num_tasks} items')

    dwa_scheduler = DynamicWeightAverageScheduler(
        num_tasks=num_tasks,
        w_init=multi_task_loss_init_weights,
    )
    if dwa_state_dict is not None:
        dwa_scheduler.load_state_dict(dwa_state_dict)
        logger.info(f"Restored DWA scheduler state: {dwa_state_dict}")
    best_val_loss = start_best_val_loss
    steps_per_epoch = len(train_dataloader)
    optimizer_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    global_step = start_epoch * optimizer_steps_per_epoch
    accumulation_step = start_epoch * steps_per_epoch
    total_optimizer_steps = max_epochs * optimizer_steps_per_epoch
    model.to(device)

    if start_epoch > 0:
        logger.info(f"Resuming training from epoch {start_epoch} (global_step={global_step})")
        logger.info(f"Best val loss so far: {start_best_val_loss:.6f}")
    logger.info(f"Training for {max_epochs} epochs ({total_optimizer_steps} optimizer steps)")
    logger.info(f"Device: {device}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Gradient clipping max norm: {grad_clip_max_norm}")
    logger.info(f"Warmup steps: {warmup_steps}")

    if lr_scheduler == LRScheduler.CosineAnnealingLR:
        # T_max is remaining steps after warmup
        t_max = max(1, total_optimizer_steps - warmup_steps)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=learning_rate/10,
        )
        if scheduler_state_dict is not None:
            main_scheduler.load_state_dict(scheduler_state_dict)
            logger.info("Restored scheduler state from checkpoint")
        logger.info(f"Using CosineAnnealingLR with T_max={t_max}")
    elif lr_scheduler == LRScheduler.CosineAnnealingWarmRestarts:
        if cosine_warm_restarts_T_0 is None:
            raise ValueError("cosine_warm_restarts_T_0 must be set when using CosineAnnealingWarmRestarts")
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cosine_warm_restarts_T_0,
            eta_min=learning_rate/10,
        )
        if scheduler_state_dict is not None:
            main_scheduler.load_state_dict(scheduler_state_dict)
            logger.info("Restored scheduler state from checkpoint")
        logger.info(
            f"Using CosineAnnealingWarmRestarts with T_0={cosine_warm_restarts_T_0}"
        )
    else:
        main_scheduler = None

    # Setup warmup scheduler
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=learning_rate/10,
            total_iters=warmup_steps,
        )
        logger.info(f"Using linear warmup for {warmup_steps} steps")
    else:
        warmup_scheduler = None

    scheduler = main_scheduler

    progress_logger = None

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        losses = []
        point_losses = []
        tissue_mask_losses = []
        grad_losses = []

        for batch in train_dataloader:
            if progress_logger is None and is_main_process():
                progress_logger = ProgressLogger(desc='Training', total=total_optimizer_steps, log_every=log_interval, start_iter=global_step)

            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)
            pad_masks = batch["pad_masks"].to(device)
            orientations = batch["orientations"]

            if predict_tissue_mask:
                tissue_masks = batch["tissue_masks"].to(device)
            else:
                tissue_masks = None

            with autocast_context:
                with timed():
                    model_out = model(input_images)
                    if predict_tissue_mask:
                        pred_template_points, pred_tissue_mask_logits = model_out[:, :-1], model_out[:, -1]
                        mask = tissue_masks
                        # Mask out padded pixels from tissue mask loss
                        bce_per_pixel = F.binary_cross_entropy_with_logits(
                            pred_tissue_mask_logits, tissue_masks, reduction='none'
                        )
                        masked_bce = bce_per_pixel * pad_masks
                        tissue_mask_loss = masked_bce.sum() / pad_masks.sum().clamp(min=1.0)

                    else:
                        pred_template_points = model_out
                        mask = pad_masks
                        tissue_mask_loss = None
                point_loss = calc_coord_loss(
                    pred=pred_template_points,
                    target=target_template_points,
                    mask=mask,
                    orientations=orientations
                )
                grad_loss = calc_spatial_gradient_loss(
                    pred=pred_template_points,
                    mask=mask,
                )
                loss = calc_multi_task_loss(
                    point_loss=point_loss,
                    grad_loss=grad_loss,
                    tissue_mask_loss=tissue_mask_loss,
                    dwa_scheduler=dwa_scheduler,
                )

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Backward pass with optional gradient scaling for mixed precision
            # Skip gradient sync on intermediate accumulation steps (DDP only)
            accumulation_step += 1
            sync_gradients = (accumulation_step % gradient_accumulation_steps == 0)
            no_sync = model.no_sync() if (hasattr(model, 'no_sync') and not sync_gradients) else nullcontext()
            with no_sync:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            point_losses.append(point_loss.item())
            grad_losses.append(grad_loss.item())
            if predict_tissue_mask:
                tissue_mask_losses.append(tissue_mask_loss.item())
            losses.append(loss.item() * gradient_accumulation_steps)

            # Only step optimizer after accumulating enough gradients
            grad_norm = None
            if accumulation_step % gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Unscale gradients before clipping (required for accurate clipping)
                    scaler.unscale_(optimizer)

                # Gradient clipping before optimizer step
                if grad_clip_max_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_max_norm
                    )
                else:
                    grad_norm = None

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # Step warmup scheduler during warmup phase
                if warmup_scheduler is not None and global_step <= warmup_steps:
                    warmup_scheduler.step()
                # Step main scheduler after warmup
                # (ReduceLROnPlateau is stepped at eval time with reduced val loss)
                elif main_scheduler is not None:
                    if lr_scheduler in (LRScheduler.CosineAnnealingWarmRestarts, LRScheduler.CosineAnnealingLR):
                        main_scheduler.step()

            train_metrics = {
                "train/loss": loss.item() * gradient_accumulation_steps,
                "train/point_loss": point_loss.item(),
                "train/grad_loss": grad_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/point_loss_weight": dwa_scheduler.get_weights()[0],
                "train/grad_loss_weight": dwa_scheduler.get_weights()[2],
            }
            if grad_norm is not None:
                train_metrics["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if predict_tissue_mask:
                train_metrics['train/tissue_mask_loss'] = tissue_mask_loss.item()
                train_metrics['train/tissue_mask_loss_weight'] = dwa_scheduler.get_weights()[1]

            if is_main_process():
                mlflow.log_metrics(train_metrics, step=global_step)

            if is_main_process():
                log_msg = f'loss={loss.item() * gradient_accumulation_steps:.3f}'
                if predict_tissue_mask:
                    log_msg += f'; point_loss={point_loss.item():.3f}; tissue_mask_loss={tissue_mask_loss.item():.3f}, loss_weights: {dwa_scheduler.get_weights()}; grad_loss: {grad_loss.item():.3f}'
                progress_logger.log_progress(other=log_msg)

            # Periodic evaluation
            if global_step % eval_interval == 0:
                # Synchronize all processes before evaluation
                if dist.is_initialized():
                    dist.barrier(device_ids=[get_local_rank()])

                if is_main_process():
                    logger.info(f"Evaluating at step {global_step}")
                if is_debug:
                    # evaluate train too
                    evaluate(
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
                        coord_loss=calc_coord_loss,
                        is_debug=is_debug,
                        predict_tissue_mask=predict_tissue_mask,
                        terminology_path=terminology_path,
                        is_train=True,
                        dwa_scheduler=dwa_scheduler,
                    )
                val_metrics = evaluate(
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
                    coord_loss=calc_coord_loss,
                    is_debug=is_debug,
                    predict_tissue_mask=predict_tissue_mask,
                    terminology_path=terminology_path,
                    is_train=False,
                    dwa_scheduler=dwa_scheduler,
                )

                current_lr = optimizer.param_groups[0]['lr']

                metrics = {
                    "eval/loss": val_metrics["val_loss"],
                    "eval/point_loss": val_metrics['val_point_loss'],
                    "eval/val_rmse": val_metrics["val_rmse"],
                    "eval/ccf_annotation_dice": val_metrics["val_ccf_annotation_dice"]
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

                log_msg = (f"Epoch {epoch+1} | Step {global_step} | "
                          f"Train loss: {loss.item() * gradient_accumulation_steps:.6f} | Val loss: {val_metrics['val_loss']:.6f} | "
                          f"Val ccf annotation dice: {val_metrics['val_ccf_annotation_dice']:.3f} | "
                          f"Val RMSE: {val_metrics['val_rmse']:.6f} | "
                          f"LR: {current_lr:.6e}")

                if predict_tissue_mask:
                    log_msg += f' | Train point loss: {point_loss.item():.6f} | val point loss: {val_metrics["val_point_loss"]:.6f} | Val tissue mask dice: {val_metrics["val_tissue_mask_dice"]:.6f}'

                if is_main_process():
                    logger.info(log_msg)

                checkpoint_path = Path(model_weights_out_dir) / f"best.pt"
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                if is_main_process():
                    torch.save(
                        obj={
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'dwa_state_dict': dwa_scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                            'val_rmse': val_metrics['val_rmse'],
                            'lr': scheduler.get_last_lr() if scheduler is not None else learning_rate
                        },
                        f=checkpoint_path,
                    )
                if dist.is_initialized():
                    dist.barrier(device_ids=[get_local_rank()])

                # Check for improvement using unweighted loss sum
                # since multi-task loss weighting could be artificially lower
                val_unweighted_loss = val_metrics['val_point_loss']
                if predict_tissue_mask:
                    val_unweighted_loss += val_metrics['val_tissue_mask_loss']
                if val_unweighted_loss < best_val_loss - min_delta:
                    best_val_loss = val_unweighted_loss

                    if is_main_process():
                        mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
                        mlflow.log_metric("best_val_loss", best_val_loss, step=global_step)

                    if is_main_process():
                        logger.info(f"New best model saved! Val loss: {best_val_loss:.6f}")

                model.train()

            if global_step % save_every == 0:
                checkpoint_path = Path(model_weights_out_dir) / f"{global_step}.pt"
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                if is_main_process():
                    logger.info(f'Saving checkpoint to {checkpoint_path}')
                    torch.save(
                        obj={
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'dwa_state_dict': dwa_scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                            'lr': scheduler.get_last_lr() if scheduler is not None else learning_rate
                        },
                        f=checkpoint_path,
                    )
                    mlflow.log_artifact(str(checkpoint_path), artifact_path="models")


        if is_main_process():
            logger.info(f'Epoch {epoch + 1}/{max_epochs} completed')
        dwa_scheduler.update(
            avg_point_loss=reduce_mean(sum(point_losses) / steps_per_epoch, device=device),
            avg_tissue_mask_loss=reduce_mean(sum(tissue_mask_losses) / steps_per_epoch, device=device) if predict_tissue_mask else None,
            avg_spatial_gradient_loss=reduce_mean(sum(grad_losses) / steps_per_epoch, device=device)
        )

    if is_main_process():
        logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
        mlflow.log_metric("final_best_val_loss", best_val_loss)
    return best_val_loss
