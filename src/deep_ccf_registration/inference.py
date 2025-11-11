from pathlib import Path
from typing import Any, Optional

import albumentations
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils import utils
from loguru import logger
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.models.unet import UNet
from deep_ccf_registration.utils.transforms import apply_transforms_to_points
from deep_ccf_registration.utils.interpolation import interpolate


class RegionAcronymCCFIdsMap(BaseModel):
    major_regions: dict[str, list[int]] = Field(
        description="Mapping from major region ccf acronym to list of descendent node IDs including itself")
    small_regions: dict[str, list[int]] = Field(
        description="Mapping from small region ccf acronym to list of descendent node IDs including itself")


def _resize_to_original(
        pred_patch: torch.Tensor,
        gt_shape: tuple[int, int],
        pre_pad_shape: Optional[tuple[int, int]] = None,
        pad_top: Optional[int] = None,
        pad_left: Optional[int] = None,
) -> torch.Tensor:
    """if the input images are resized, resize it back to the original dimensions.
    This handles the fact that the input image might be padded, and so padding is reversed and then
    image excluding padding resized.

    :param pred_patch: The predicted points
    :param gt_shape: The unmodified ground truth patch shape
    :param pad_transform: output from albumentations transform

    :return the resized prediction
    """
    if pre_pad_shape is not None:
        # 1. Crop out padding
        H_scaled, W_scaled = pre_pad_shape
        pred_cropped = pred_patch[pad_top:pad_top + H_scaled, pad_left:pad_left + W_scaled]
    else:
        pred_cropped = pred_patch

    # 2. Resize back to original
    resize = albumentations.Resize(height=gt_shape[0], width=gt_shape[1])
    pred_original = resize(image=pred_cropped)['image']

    return pred_original

@torch.no_grad()
def evaluate(
        val_loader: DataLoader,
        model: UNet,
        ccf_annotations: np.ndarray,
        ls_template_to_ccf_affine_path: Path,
        ls_template_to_ccf_inverse_warp: np.ndarray,
        ls_template: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        ccf_template_parameters: AntsImageParameters,
        region_ccf_ids_map: RegionAcronymCCFIdsMap,
        device: str = "cuda",
) -> tuple[float, dict[str, float], dict[str, float]]:
    """
    :param val_loader: validation DataLoader
    :param model: model
    :param ccf_annotations: ccf annotation volume
    :param ls_template: light sheet template volume
    :param ls_template_to_ccf_affine_path: path to ls template to ccf affine
    :param ls_template_to_ccf_inverse_warp: ls template to ccf inverse warp
    :param ls_template_parameters: ls template AntsImageParameters
    :param ccf_template_parameters: ccf template AntsImageParameters
    :param region_ccf_ids_map: `RegionAcronymCCFIdsMap`
    :param device:
    :return: tuple of: (rmse in microns ignoring background (just tissue), mapping from major brain
        region to dice, mapping from small region to dice)
    """
    slice_dataset: SliceDataset = val_loader.dataset
    if isinstance(slice_dataset, Subset):
        slice_dataset = slice_dataset.dataset

    # Build class mapping for dice calculation
    major_region_map = _build_class_mapping(region_ccf_ids_map.major_regions)
    small_region_map = _build_class_mapping(region_ccf_ids_map.small_regions)

    # Initialize accumulators for metrics
    # For RMSE
    sum_squared_errors = 0.0
    tissue_pixel_count = 0

    # For Dice - confusion matrices
    n_major_classes = len(region_ccf_ids_map.major_regions) + 1  # +1 for background
    n_small_classes = len(region_ccf_ids_map.small_regions) + 1  # +1 for background
    major_confusion_matrix = np.zeros((n_major_classes, n_major_classes), dtype=np.int64)
    small_confusion_matrix = np.zeros((n_small_classes, n_small_classes), dtype=np.int64)

    # For sagittal orientation RMSE
    ml_dim_size = ls_template.shape[0] * ls_template_parameters.scale[0] * 1000


    model.eval()
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing patches")):
        if slice_dataset.patch_size is not None:
            input_images, output_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, raw_shapes = batch
        else:
            input_images, output_points, dataset_indices, slice_indices, orientations, input_image_transforms, raw_shapes = batch

        input_images = input_images.to(device)

        # Run inference
        pred_ls_template_points = model(input_images).cpu().numpy()  # (B, 3, H, W)
        pred_ls_template_points = pred_ls_template_points.transpose(0, 2, 3, 1)  # (B, H, W, 3)
        gt_ls_template_points = output_points.numpy().transpose(0, 2, 3, 1)  # (B, H, W, 3)

        # Process each patch in the batch
        for i in range(pred_ls_template_points.shape[0]):
            pred_patch = pred_ls_template_points[i]  # (H, W, 3)
            gt_patch = gt_ls_template_points[i]  # (H, W, 3)

            if pred_patch.shape != gt_patch.shape:
                pred_patch = _resize_to_original(
                    pred_patch=pred_patch,
                    pre_pad_shape=tuple([int(input_image_transforms['shape'][ii][i].item()) for ii in range(2)]) if input_image_transforms else None,
                    pad_top=input_image_transforms['pad_top'][i].item() if input_image_transforms else None,
                    pad_left=input_image_transforms['pad_left'][i].item() if input_image_transforms else None,
                    gt_shape=gt_patch.shape
                )

            # Transform patch to CCF space
            pred_ccf_pts = transform_ls_space_to_ccf_space(
                points=pred_patch,
                ls_template_to_ccf_affine_path=ls_template_to_ccf_affine_path,
                ls_template_to_ccf_inverse_warp=ls_template_to_ccf_inverse_warp,
                ls_template_parameters=ls_template_parameters,
                ccf_template_parameters=ccf_template_parameters
            )

            gt_ccf_pts = transform_ls_space_to_ccf_space(
                points=gt_patch,
                ls_template_to_ccf_affine_path=ls_template_to_ccf_affine_path,
                ls_template_to_ccf_inverse_warp=ls_template_to_ccf_inverse_warp,
                ls_template_parameters=ls_template_parameters,
                ccf_template_parameters=ccf_template_parameters
            )

            # Get CCF annotations for patch
            pred_ccf_annot = get_ccf_annotations(ccf_annotations, pred_ccf_pts).reshape(
                pred_patch.shape[:-1])
            gt_ccf_annot = get_ccf_annotations(ccf_annotations, gt_ccf_pts).reshape(
                gt_patch.shape[:-1])

            # Accumulate RMSE in microns, ignoring background
            tissue_mask = gt_ccf_annot != 0

            if np.any(tissue_mask):
                # ANTs uses millimeters. convert to microns
                pred_tissue = pred_patch[tissue_mask] * 1000
                gt_tissue = gt_patch[tissue_mask] * 1000

                orientation = SliceOrientation(orientations[i])
                if orientation == SliceOrientation.SAGITTAL:
                    # Hemisphere-agnostic sum of squared errors
                    ml_error_direct = (pred_tissue[..., 0] - gt_tissue[..., 0]) ** 2
                    ml_error_flipped = ((ml_dim_size - pred_tissue[..., 0]) - gt_tissue[..., 0]) ** 2
                    ml_error = np.minimum(ml_error_direct, ml_error_flipped)

                    ap_error = (pred_tissue[..., 1] - gt_tissue[..., 1]) ** 2
                    dv_error = (pred_tissue[..., 2] - gt_tissue[..., 2]) ** 2

                    patch_squared_errors = ml_error + ap_error + dv_error
                else:
                    # sum of squared error
                    patch_squared_errors = np.sum((pred_tissue - gt_tissue) ** 2, axis=1)

                sum_squared_errors += np.sum(patch_squared_errors)
                tissue_pixel_count += np.sum(tissue_mask)

            # Accumulate confusion matrices for Dice
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

    # Calculate final metrics
    logger.info('Calculating RMSE from accumulated statistics')
    rmse = np.sqrt(sum_squared_errors / tissue_pixel_count) if tissue_pixel_count > 0 else 0.0

    logger.info('Calculating dice scores from confusion matrices')
    major_region_dice = _calc_dice_from_confusion_matrix(
        confusion_matrix=major_confusion_matrix,
        region_map=region_ccf_ids_map.major_regions
    )

    small_region_dice = _calc_dice_from_confusion_matrix(
        confusion_matrix=small_confusion_matrix,
        region_map=region_ccf_ids_map.small_regions
    )

    return rmse, major_region_dice, small_region_dice


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

    # Map acronyms to class indices (1-indexed, skipping background)
    acronym_to_class = {
        acronym: class_idx
        for class_idx, acronym in enumerate(region_map.keys(), start=1)
    }

    for acronym, class_idx in acronym_to_class.items():
        # Dice = 2 * TP / (2 * TP + FP + FN)
        tp = confusion_matrix[class_idx, class_idx]
        fp = confusion_matrix[:, class_idx].sum() - tp
        fn = confusion_matrix[class_idx, :].sum() - tp

        denominator = 2 * tp + fp + fn
        dice = (2 * tp / denominator) if denominator > 0 else 0.0

        result[acronym] = float(dice)

    return result

def transform_ls_space_to_ccf_space(
        points: np.ndarray,
        ls_template_to_ccf_affine_path: Path,
        ls_template_to_ccf_inverse_warp: np.ndarray,
        ls_template_parameters: AntsImageParameters,
        ccf_template_parameters: AntsImageParameters,
) -> np.ndarray:
    """
    Transform points to CCF index space

    :param points: MxNx3 predicted points in LS template space
    :return: ccf points in index space
    """
    points = points.reshape((-1, 3))

    ccf_pts = apply_transforms_to_points(
        points=points,
        affine_path=ls_template_to_ccf_affine_path,
        warp=ls_template_to_ccf_inverse_warp,
        template_parameters=ccf_template_parameters,
        crop_warp_to_bounding_box=False
    )

    ccf_pts = utils.convert_from_ants_space(
        ccf_template_parameters, ccf_pts
    )

    _, swapped, _ = utils.get_orientation_transform(
        ls_template_parameters.orientation,
        ccf_template_parameters.orientation,
    )

    ccf_pts = ccf_pts[:, swapped]

    return ccf_pts


def get_ccf_annotations(ccf_annotations: np.ndarray, pts: np.ndarray):
    labels = interpolate(
        array=ccf_annotations.astype(np.float32),
        grid=pts,
        mode='nearest'
    )

    labels = labels.squeeze().cpu().numpy().astype(int)  # (M*N,)

    return labels
