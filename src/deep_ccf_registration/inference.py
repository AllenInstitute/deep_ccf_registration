import random
from typing import Optional

import albumentations
import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import  torch.nn.functional as F

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.losses.coord_loss import HemisphereAgnosticCoordLoss, mirror_points
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.utils.utils import get_ccf_annotations
from deep_ccf_registration.utils.visualization import create_diagnostic_image


class RegionAcronymCCFIdsMap(BaseModel):
    major_regions: dict[str, list[int]] = Field(
        description="Mapping from major region ccf acronym to list of descendent node IDs including itself")
    small_regions: dict[str, list[int]] = Field(
        description="Mapping from small region ccf acronym to list of descendent node IDs including itself")


def _resize_to_original(
        img: torch.Tensor,
        gt_shape: tuple[int, int],
        pre_pad_shape: Optional[tuple[int, int]] = None,
        pad_top: Optional[int] = None,
        pad_left: Optional[int] = None,
) -> torch.Tensor:
    """if the input images are resized, resize it back to the original dimensions.
    This handles the fact that the input image might be padded, and so padding is reversed and then
    image excluding padding resized.

    :param img: The predicted points
    :param gt_shape: The unmodified ground truth patch shape
    :param pad_transform: output from albumentations transform

    :return the resized prediction
    """
    img = np.permute_dims(img, (1, 2, 0))

    if pre_pad_shape is not None:
        # 1. Crop out padding
        H_scaled, W_scaled = pre_pad_shape
        pred_cropped = img[pad_top:pad_top + H_scaled, pad_left:pad_left + W_scaled]
    else:
        pred_cropped = img

    # 2. Resize back to original
    resize = albumentations.Resize(height=gt_shape[0], width=gt_shape[1])
    pred_original = resize(image=pred_cropped)['image']

    pred_original = torch.tensor(pred_original).permute((2, 0, 1))
    return pred_original

@torch.no_grad()
def evaluate(
        val_loader: DataLoader,
        model: UNet,
        ccf_annotations: np.ndarray,
        ls_template: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        region_ccf_ids_map: RegionAcronymCCFIdsMap,
        iteration: int,
        is_train: bool,
        device: str = "cuda",
        exclude_background_pixels: bool = True,
        viz_slice_indices: Optional[list[int]] = None,
) -> tuple[float, dict[str, float], dict[str, float], float]:
    """
    :param val_loader: validation DataLoader
    :param model: model
    :param ccf_annotations: ccf annotation volume
    :param ls_template: light sheet template volume
    :param ls_template_parameters: ls template AntsImageParameters
    :param region_ccf_ids_map: `RegionAcronymCCFIdsMap`
    :param device:
    :param iteration
    :param exclude_background_pixels: whether to use a tissue mask to exclude background pixels
        Otherwise, just excludes pad pixels
    :param is_train: Whether train or val
    :param viz_slice_indices: Fixed list of slice indices to visualize throughout training.
        If None, no visualizations are saved. If provided, only slices matching these indices are visualized.
    :return: tuple of: rmse in microns, mapping from major brain
        region to dice, mapping from small region to dice)
    """
    slice_dataset: SliceDataset = val_loader.dataset
    if isinstance(slice_dataset, Subset):
        if isinstance(slice_dataset.dataset, Subset):
            slice_dataset = slice_dataset.dataset.dataset
        else:
            slice_dataset = slice_dataset.dataset

    major_region_map = _build_class_mapping(region_ccf_ids_map.major_regions)
    small_region_map = _build_class_mapping(region_ccf_ids_map.small_regions)

    errors = 0.0
    errors_direct = 0.0
    errors_flipped = 0.0
    coord_error_denominator = 0
    tissue_mask_tp_sum = 0
    tissue_mask_fp_sum = 0
    tissue_mask_fn_sum = 0

    n_major_classes = len(region_ccf_ids_map.major_regions) + 1  # +1 for background
    n_small_classes = len(region_ccf_ids_map.small_regions) + 1  # +1 for background
    major_confusion_matrix = np.zeros((n_major_classes, n_major_classes), dtype=np.int64)
    small_confusion_matrix = np.zeros((n_small_classes, n_small_classes), dtype=np.int64)

    model.eval()
    sample_idx = 0
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluation")):
        if slice_dataset.patch_size is not None:
            input_images, gt_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, masks = batch
        else:
            input_images, gt_template_points, dataset_indices, slice_indices, orientations, input_image_transforms, masks = batch

        input_images = input_images.to(device)
        gt_template_points = gt_template_points.cpu()

        # Run inference
        model_out = model(input_images).cpu()
        if exclude_background_pixels:
            pred_ls_template_points = model_out[:, :-1]
            pred_tissue_logits = model_out[:, -1]
            masks = (F.sigmoid(pred_tissue_logits) > 0.5).to(torch.uint8)
        else:
            pred_ls_template_points = model_out

        for i in range(pred_ls_template_points.shape[0]):
            pred_patch = pred_ls_template_points[i]  # (3, H, W)
            gt_patch = gt_template_points[i]  # (3, H, W)
            mask = masks[i]

            if pred_patch.shape != gt_patch.shape:
                pred_patch = _resize_to_original(
                    img=pred_patch.cpu().numpy(),
                    pre_pad_shape=tuple([int(input_image_transforms['shape'][ii][i].item()) for ii in range(2)]) if input_image_transforms else None,
                    pad_top=input_image_transforms['pad_top'][i].item() if input_image_transforms else None,
                    pad_left=input_image_transforms['pad_left'][i].item() if input_image_transforms else None,
                    gt_shape=gt_patch.shape[1:]
                )
                mask = _resize_to_original(
                    img=mask.unsqueeze(0).cpu().numpy(),
                    pre_pad_shape=tuple(
                        [int(input_image_transforms['shape'][ii][i].item()) for ii in
                         range(2)]) if input_image_transforms else None,

                    pad_top=input_image_transforms['pad_top'][
                        i].item() if input_image_transforms else None,
                    pad_left=input_image_transforms['pad_left'][
                        i].item() if input_image_transforms else None,
                    gt_shape=gt_patch.shape[1:]
                ).squeeze(0)

            pred_index_space = convert_from_ants_space(template_parameters=ls_template_parameters,
                                                       physical_pts=pred_patch.permute((1, 2, 0)).reshape((-1, 3)).numpy())
            gt_index_space = convert_from_ants_space(template_parameters=ls_template_parameters,
                                                     physical_pts=gt_patch.permute((1, 2, 0)).reshape((-1, 3)).numpy())

            pred_ccf_annot = get_ccf_annotations(ccf_annotations, pred_index_space).reshape(
                pred_patch.shape[1:])
            pred_ccf_annot[(1 - mask).bool()] = 0
            gt_ccf_annot = get_ccf_annotations(ccf_annotations, gt_index_space).reshape(
                gt_patch.shape[1:])

            gt_tissue_mask = gt_ccf_annot != 0

            orientation = SliceOrientation(orientations[i])

            # Calculate direct error (pred vs gt)
            direct_error = (pred_patch.unsqueeze(0) - gt_patch.unsqueeze(0)) ** 2

            # Calculate flipped error (pred vs flipped gt) for sagittal slices
            if orientation == SliceOrientation.SAGITTAL:
                gt_flipped = mirror_points(gt_patch.unsqueeze(0), ml_dim_size=ls_template.shape[0],
                                           template_parameters=ls_template_parameters)
                flipped_error = (pred_patch.unsqueeze(0) - gt_flipped) ** 2
            else:
                flipped_error = direct_error  # For non-sagittal, flipped = direct

            # Hemisphere-agnostic error (minimum of the two)
            hemisphere_agnostic_error = torch.minimum(direct_error, flipped_error)

            if exclude_background_pixels:
                errors_direct += direct_error[0][:, gt_tissue_mask].sum(dim=0).sum()
                errors_flipped += flipped_error[0][:, gt_tissue_mask].sum(dim=0).sum()
                errors += hemisphere_agnostic_error[0][:, gt_tissue_mask].sum(dim=0).sum()
                coord_error_denominator += gt_tissue_mask.sum()
                tissue_mask_tp_sum += ((gt_tissue_mask == 1) & (mask.cpu().numpy() == 1)).sum()
                tissue_mask_fp_sum += ((gt_tissue_mask == 0) & (mask.cpu().numpy() == 1)).sum()
                tissue_mask_fn_sum += ((gt_tissue_mask == 1) & (mask.cpu().numpy() == 0)).sum()
            else:
                errors_direct += direct_error[0][:, mask.bool()].sum(dim=0).sum()
                errors_flipped += flipped_error[0][:, mask.bool()].sum(dim=0).sum()
                errors += hemisphere_agnostic_error[0][:, mask.bool()].sum(dim=0).sum()
                coord_error_denominator += mask.sum()

            # Use direct error for diagnostic visualization
            patch_errors = direct_error

            # Visualize if this sample index is in the fixed visualization indices
            if viz_slice_indices is not None and sample_idx in viz_slice_indices:
                fig = create_diagnostic_image(
                    input_image=input_images[i].cpu().squeeze(0),
                    slice_idx=slice_indices[i],
                    errors=patch_errors.cpu().numpy().squeeze(0),
                    pred_ccf_annotations=pred_ccf_annot,
                    gt_ccf_annotations=gt_ccf_annot,
                    iteration=iteration,
                    pred_template_points=pred_patch,
                    gt_template_points=gt_patch,
                    gt_mask=(gt_ccf_annot != 0) if exclude_background_pixels else mask.bool().numpy(),
                    pred_mask=mask.bool().numpy() if exclude_background_pixels else None,
                )
                mlflow.log_figure(fig, f"inference/{"train" if is_train else "val"}/slice_{slice_indices[i]}/y_{patch_ys[i]}_x_{patch_xs[i]}/step_{iteration}.png")
                plt.close(fig)

            sample_idx += 1

            _update_confusion_matrix(
                confusion_matrix=major_confusion_matrix,
                pred_annotations=pred_ccf_annot,
                true_annotations=gt_ccf_annot,
                class_mapping=major_region_map
            )

            _update_confusion_matrix(
                confusion_matrix=small_confusion_matrix,
                pred_annotations=pred_ccf_annot,
                true_annotations=gt_ccf_annot,
                class_mapping=small_region_map
            )

    rmse = np.sqrt(errors / coord_error_denominator) if coord_error_denominator > 0 else 0.0
    rmse_direct = np.sqrt(errors_direct / coord_error_denominator) if coord_error_denominator > 0 else 0.0
    rmse_flipped = np.sqrt(errors_flipped / coord_error_denominator) if coord_error_denominator > 0 else 0.0

    # convert to microns
    rmse *= 1000
    rmse_direct *= 1000
    rmse_flipped *= 1000

    # Log the three error metrics
    logger.info(f"RMSE (hemisphere-agnostic/min): {rmse:.2f} microns")
    logger.info(f"RMSE (direct to GT): {rmse_direct:.2f} microns")
    logger.info(f"RMSE (flipped GT): {rmse_flipped:.2f} microns")

    # Log to MLflow as well
    mlflow.log_metrics({
        f"eval/rmse_hemisphere_agnostic_{'train' if is_train else 'val'}": rmse,
        f"eval/rmse_direct_{'train' if is_train else 'val'}": rmse_direct,
        f"eval/rmse_flipped_{'train' if is_train else 'val'}": rmse_flipped,
    })

    if exclude_background_pixels:
        tissue_mask_dice = (2 * tissue_mask_tp_sum) / (tissue_mask_fp_sum + tissue_mask_fn_sum + 1e-8)
    else:
        tissue_mask_dice = None

    logger.info('Calculating dice scores from confusion matrices')
    major_region_dice = _calc_dice_from_confusion_matrix(
        confusion_matrix=major_confusion_matrix,
        region_map=region_ccf_ids_map.major_regions
    )

    small_region_dice = _calc_dice_from_confusion_matrix(
        confusion_matrix=small_confusion_matrix,
        region_map=region_ccf_ids_map.small_regions
    )

    return rmse, major_region_dice, small_region_dice, tissue_mask_dice


def _build_class_mapping(region_map: dict[str, list[int]]) -> dict[int, int]:
    """
    Build mapping from CCF structure ID to class index (0-indexed, 0 is background)

    :param region_map: Mapping from region acronym to list of CCF IDs
    :return: Mapping from CCF structure ID to class index
    """
    structure_to_class = {}

    for class_idx, (acronym, structure_ids) in enumerate(region_map.items(), start=1):
        for struct_id in structure_ids:
            structure_to_class[struct_id] = class_idx

    return structure_to_class


def _update_confusion_matrix(
        confusion_matrix: np.ndarray,
        pred_annotations: np.ndarray,
        true_annotations: np.ndarray,
        class_mapping: dict[int, int]
):
    """
    Update confusion matrix with predictions from a patch

    :param confusion_matrix: Confusion matrix to update (modified in-place)
    :param pred_annotations: Predicted CCF annotations
    :param true_annotations: Ground truth CCF annotations
    :param class_mapping: Mapping from CCF structure ID to class index
    """
    # Flatten annotations
    pred_flat = pred_annotations.flatten()
    true_flat = true_annotations.flatten()

    # Map CCF IDs to class indices (default to 0 for background/unmapped)
    pred_classes = np.array([class_mapping.get(int(pid), 0) for pid in pred_flat])
    true_classes = np.array([class_mapping.get(int(tid), 0) for tid in true_flat])

    # Update confusion matrix
    for pred_cls, true_cls in zip(pred_classes, true_classes):
        confusion_matrix[true_cls, pred_cls] += 1


def _calc_dice_from_confusion_matrix(
        confusion_matrix: np.ndarray,
        region_map: dict[str, list[int]]
) -> dict[str, float]:
    """
    Calculate Dice scores from confusion matrix

    :param confusion_matrix: Confusion matrix (n_classes x n_classes)
    :param region_map: Mapping from region acronym to list of CCF IDs
    :return: Dictionary mapping region acronym to Dice score
    """
    result = {}

    acronym_to_class = {
        acronym: class_idx
        for class_idx, acronym in enumerate(region_map.keys(), start=1)
    }

    for acronym, class_idx in acronym_to_class.items():
        if confusion_matrix[class_idx].sum() == 0:
            continue
        tp = confusion_matrix[class_idx, class_idx]
        fp = confusion_matrix[:, class_idx].sum() - tp
        fn = confusion_matrix[class_idx, :].sum() - tp

        denominator = 2 * tp + fp + fn
        dice = (2 * tp / denominator)

        result[acronym] = float(dice)

    return result