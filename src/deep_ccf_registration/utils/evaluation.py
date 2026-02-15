import tempfile
from pathlib import Path
from typing import Iterator, ContextManager, Optional, Any

import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from monai.metrics import DiceMetric
from torch.nn import functional as F

from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.datasets.transforms import get_template_point_normalization_inverse
from deep_ccf_registration.utils.metrics import MSE
from deep_ccf_registration.utils.ddp import is_main_process, reduce_mean
from deep_ccf_registration.utils.visualization import viz_sample


def evaluate(
    dataloader: Iterator,
    model: torch.nn.Module,
    coord_loss: MSE,
    device: str,
    autocast_context: ContextManager,
    max_iters: int,
    denormalize_pred_template_points: bool,
    global_step: int,
    ccf_annotations: np.ndarray,
    terminology_path: Path,
    viz_sample_count: int = 10,
    ls_template_parameters: Optional[TemplateParameters] = None,
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
    viz_samples = []
    total_viz_candidates = 0  # Track total samples seen for reservoir sampling
    eval_records = []
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
                    # Mask out padded pixels from tissue mask loss
                    bce_per_pixel = F.binary_cross_entropy_with_logits(
                        pred_tissue_mask_logits, tissue_masks, reduction='none'
                    )
                    masked_bce = bce_per_pixel * pad_masks
                    tissue_mask_loss = masked_bce.sum() / pad_masks.sum().clamp(min=1.0)
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

            # Compute RMSE against eval targets at original resolution
            # Targets are NOT interpolated, predictions ARE upsampled to match
            if "eval_template_points" in batch and "eval_shapes" in batch:
                eval_targets = batch["eval_template_points"].to(device)
                eval_pad_masks = batch["eval_pad_masks"].to(device)
                eval_shapes = batch["eval_shapes"]

                if predict_tissue_mask and "eval_tissue_masks" in batch:
                    eval_tissue_masks = batch["eval_tissue_masks"].to(device)

                for si in range(pred_points.shape[0]):
                    eval_shape = eval_shapes[si]
                    if eval_shape is None:
                        continue

                    eval_h, eval_w = eval_shape

                    # Get the valid (non-padded) region of predictions
                    pad_mask_i = pad_masks[si]
                    valid_h = int(pad_mask_i.any(dim=1).sum())
                    valid_w = int(pad_mask_i.any(dim=0).sum())

                    # Crop out padding from predictions before upsampling
                    pred_content = pred_points[si:si+1, :, :valid_h, :valid_w]

                    # Upsample unpadded predictions to eval resolution
                    pred_upsampled = F.interpolate(
                        pred_content,
                        size=(eval_h, eval_w),
                        mode="bilinear",
                        align_corners=False
                    )

                    # Get eval target and mask for this sample
                    eval_target_i = eval_targets[si:si+1, :, :eval_h, :eval_w]

                    if predict_tissue_mask:
                        eval_mask_i = eval_tissue_masks[si:si+1, :eval_h, :eval_w]
                    else:
                        eval_mask_i = eval_pad_masks[si:si+1, :eval_h, :eval_w]

                    rmse_full_res = MSE()(pred=pred_upsampled, target=eval_target_i, mask=eval_mask_i).sqrt()
                    registration_res_rmses += rmse_full_res.cpu().tolist()

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

            slice_indices = batch["slice_indices"]
            patch_ys = batch["patch_ys"]
            patch_xs = batch["patch_xs"]
            subject_ids = batch["subject_ids"]

            # Reservoir sampling to randomly select viz_sample_count samples
            for sample_idx in range(input_images.shape[0]):
                total_viz_candidates += 1
                sample_data = {
                    "input_image": input_images[sample_idx].squeeze().detach().cpu(),
                    "pred_coords": pred_points[sample_idx].detach().cpu(),
                    "gt_coords": target_template_points[sample_idx].detach().cpu(),
                    "pad_mask": pad_masks[sample_idx].detach().cpu(),
                    "tissue_mask": tissue_masks[sample_idx].detach().cpu() if predict_tissue_mask else None,
                    "pred_tissue_masks": F.sigmoid(pred_tissue_mask_logits[sample_idx]).detach().cpu() if predict_tissue_mask else None,
                    "slice_idx": int(slice_indices[sample_idx].item()),
                    "patch_y": int(patch_ys[sample_idx].item()),
                    "patch_x": int(patch_xs[sample_idx].item()),
                    "subject_id": subject_ids[sample_idx],
                }

                if len(viz_samples) < viz_sample_count:
                    viz_samples.append(sample_data)
                else:
                    # Replace with probability viz_sample_count / total_viz_candidates
                    j = np.random.randint(0, total_viz_candidates)
                    if j < viz_sample_count:
                        viz_samples[j] = sample_data

    val_loss = reduce_mean(np.mean(losses), device)
    val_rmse = reduce_mean(np.mean(rmses), device)
    val_rmse_registration_res = reduce_mean(np.mean(registration_res_rmses), device) if registration_res_rmses else None

    if predict_tissue_mask:
        val_point_loss = reduce_mean(np.mean(point_losses), device)
        val_tissue_mask_loss = reduce_mean(np.mean(tissue_mask_losses), device)
        val_tissue_mask_dice = reduce_mean(np.mean(tissue_mask_dice_metric.aggregate().item()), device)
    else:
        val_point_loss = val_loss
        val_tissue_mask_dice = None
        val_tissue_mask_loss = None

    iteration = global_step or 0
    for idx, sample in enumerate(viz_samples):
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
            ccf_annotations=ccf_annotations,
            terminology_path=terminology_path,
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
