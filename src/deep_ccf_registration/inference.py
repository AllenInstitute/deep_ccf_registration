from contextlib import nullcontext
from typing import ContextManager, Optional

import albumentations
import mlflow
import numpy as np
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanSquaredError
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
import  torch.nn.functional as F

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.losses.coord_loss import mirror_points
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.utils.transforms import convert_from_ants_space_tensor
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

# TODO leaving this here, currently unused and replaced by evaluate_batch since this was too slow
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
        autocast_context: ContextManager = nullcontext(),
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

    ccf_id_to_index = _get_ccf_id_to_index(
        ccf_annotations=ccf_annotations
    )
    major_class_mapping = _build_class_mapping(region_ccf_ids_map.major_regions, id_to_index=ccf_id_to_index)
    small_class_mapping = _build_class_mapping(region_ccf_ids_map.small_regions, id_to_index=ccf_id_to_index)

    errors = 0.0
    errors_direct = 0.0
    errors_flipped = 0.0
    coord_error_denominator = 0
    tissue_mask_tp_sum = 0
    tissue_mask_fp_sum = 0
    tissue_mask_fn_sum = 0

    n_major_classes = len(region_ccf_ids_map.major_regions) + 1  # +1 for background
    n_small_classes = len(region_ccf_ids_map.small_regions) + 1  # +1 for background
    major_confusion_matrix = torch.zeros((n_major_classes, n_major_classes), dtype=torch.int64)
    small_confusion_matrix = torch.zeros((n_small_classes, n_small_classes), dtype=torch.int64)

    ccf_annotations = torch.from_numpy(ccf_annotations)

    model.eval()
    sample_idx = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluation")):
        if slice_dataset.patch_size is not None:
            input_images, gt_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, masks = batch
        else:
            input_images, gt_template_points, dataset_indices, slice_indices, orientations, input_image_transforms, masks = batch

        input_images = input_images.to(device)

        # Run inference
        with autocast_context:
            model_out = model(input_images)

        model_out = model_out.cpu()

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

            pred_index_space = convert_from_ants_space_tensor(template_parameters=ls_template_parameters,
                                                       physical_pts=pred_patch.permute((1, 2, 0)).reshape((-1, 3)))
            gt_index_space = convert_from_ants_space_tensor(template_parameters=ls_template_parameters,
                                                     physical_pts=gt_patch.permute((1, 2, 0)).reshape((-1, 3)))

            pred_ccf_annot = get_ccf_annotations(ccf_annotations, pred_index_space, return_np=False).reshape(
                pred_patch.shape[1:])
            pred_ccf_annot[(1 - mask).bool()] = 0
            gt_ccf_annot = get_ccf_annotations(ccf_annotations, gt_index_space, return_np=False).reshape(
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
                errors_direct += direct_error[0][:, gt_tissue_mask].sum().item()
                errors_flipped += flipped_error[0][:, gt_tissue_mask].sum().item()
                errors += hemisphere_agnostic_error[0][:, gt_tissue_mask].sum().item()
                coord_error_denominator += gt_tissue_mask.sum().item()
                tissue_mask_tp_sum += ((gt_tissue_mask == 1) & (mask == 1)).sum().item()
                tissue_mask_fp_sum += ((gt_tissue_mask == 0) & (mask == 1)).sum().item()
                tissue_mask_fn_sum += ((gt_tissue_mask == 1) & (mask == 0)).sum().item()
            else:
                errors_direct += direct_error[0][:, mask.bool()].sum().item()
                errors_flipped += flipped_error[0][:, mask.bool()].sum().item()
                errors += hemisphere_agnostic_error[0][:, mask.bool()].sum().item()
                coord_error_denominator += mask.sum().item()

            # Use direct error for diagnostic visualization
            patch_errors = direct_error

            # Visualize if this sample index is in the fixed visualization indices
            if viz_slice_indices is not None and sample_idx in viz_slice_indices:
                fig = create_diagnostic_image(
                    input_image=input_images[i].cpu().squeeze(0),
                    slice_idx=slice_indices[i],
                    errors=patch_errors.cpu().numpy().squeeze(0),
                    pred_ccf_annotations=pred_ccf_annot.cpu().numpy(),
                    gt_ccf_annotations=gt_ccf_annot.cpu().numpy(),
                    iteration=iteration,
                    pred_template_points=pred_patch.cpu(),
                    gt_template_points=gt_patch.cpu(),
                    gt_mask=(gt_ccf_annot != 0).cpu().numpy() if exclude_background_pixels else mask.cpu().bool().numpy(),
                    pred_mask=mask.cpu().bool().numpy() if exclude_background_pixels else None,
                )
                fig_filename = f"slice_{slice_indices[i]}_y_{patch_ys[i]}_x_{patch_xs[i]}_step_{iteration}.png"
                mlflow.log_figure(fig, f"inference/{"train" if is_train else "val"}/slice_{slice_indices[i]}/y_{patch_ys[i]}_x_{patch_xs[i]}/{fig_filename}")
                plt.close(fig)

            sample_idx += 1

            _update_confusion_matrix(
                confusion_matrix=major_confusion_matrix,
                pred_annotations=pred_ccf_annot,
                true_annotations=gt_ccf_annot,
                class_mapping=major_class_mapping,
                ccf_id_to_index=ccf_id_to_index,
            )

            _update_confusion_matrix(
                confusion_matrix=small_confusion_matrix,
                pred_annotations=pred_ccf_annot,
                true_annotations=gt_ccf_annot,
                class_mapping=small_class_mapping,
                ccf_id_to_index=ccf_id_to_index,
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
    mlflow.log_metrics(metrics={
        f"eval/rmse_hemisphere_agnostic_{'train' if is_train else 'val'}": rmse,
        f"eval/rmse_direct_{'train' if is_train else 'val'}": rmse_direct,
        f"eval/rmse_flipped_{'train' if is_train else 'val'}": rmse_flipped,
    }, step=iteration)

    if exclude_background_pixels:
        tissue_mask_dice = (2 * tissue_mask_tp_sum) / (2 * tissue_mask_tp_sum + tissue_mask_fp_sum + tissue_mask_fn_sum + 1e-8)
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


def _get_ccf_id_to_index(ccf_annotations: np.ndarray | torch.Tensor):
    if isinstance(ccf_annotations, torch.Tensor):
        unique_ids = torch.unique(ccf_annotations).int()
    else:
        unique_ids = torch.from_numpy(np.unique(ccf_annotations).astype(np.int32))

    id_to_index = {int(uid.item()): idx for idx, uid in enumerate(unique_ids)}

    return id_to_index

def _build_class_mapping(region_map: dict[str, list[int]], id_to_index: dict[int, int]) -> torch.Tensor:
    """
    Build mapping from CCF structure ID to class index

    :param region_map: Mapping from region acronym to list of CCF IDs
    :return: class mapping from ccf index to class
    """

    class_mapping = torch.zeros(len(id_to_index))

    # Fill in class assignments
    for class_idx, (acronym, structure_ids) in enumerate(region_map.items(), start=1):
        for struct_id in structure_ids:
            if struct_id in id_to_index:
                class_mapping[id_to_index[struct_id]] = class_idx

    return class_mapping


def _update_confusion_matrix(
        confusion_matrix: torch.Tensor,
        pred_annotations: torch.Tensor,
        true_annotations: torch.Tensor,
        class_mapping: torch.Tensor,
        ccf_id_to_index: dict[int, int],
):
    """
    Update confusion matrix with predictions from a patch

    :param confusion_matrix: Confusion matrix to update (modified in-place, torch tensor)
    :param pred_annotations: Predicted CCF annotations (torch tensor)
    :param true_annotations: Ground truth CCF annotations (torch tensor)
    :param class_mapping: Dense array where class_mapping[i] = class for unique_ids[i]
    """
    device = pred_annotations.device
    class_mapping = class_mapping.to(device)

    # flatten
    pred_flat = pred_annotations.flatten().cpu().numpy()
    true_flat = true_annotations.flatten().cpu().numpy()

    pred_ccf_idx = torch.tensor([ccf_id_to_index[x] for x in pred_flat])
    pred_class = class_mapping[pred_ccf_idx]

    true_ccf_idx = torch.tensor([ccf_id_to_index[x] for x in true_flat])
    true_class = class_mapping[true_ccf_idx]

    num_classes = confusion_matrix.shape[0]
    combined = true_class.long() * num_classes + pred_class.long()
    bins = torch.bincount(combined, minlength=num_classes * num_classes)
    update = bins.reshape(num_classes, num_classes)
    confusion_matrix += update


def _calc_dice_from_confusion_matrix(
        confusion_matrix: torch.Tensor,
        region_map: dict[str, list[int]]
) -> dict[str, float]:
    """
    Calculate Dice scores from confusion matrix

    :param confusion_matrix: Confusion matrix (n_classes x n_classes, torch tensor)
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

        result[acronym] = float(dice.item())

    return result


def viz_sample(
    ls_template_parameters: AntsImageParameters,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    ccf_annotations: np.ndarray,
    iteration: int,
    pad_mask: torch.Tensor,
    input_image: torch.Tensor,
    slice_idx: int,
    errors: np.ndarray,
    pred_tissue_mask: Optional[torch.Tensor] = None,

):
    pred_index_space = convert_from_ants_space_tensor(template_parameters=ls_template_parameters,
                                                      physical_pts=pred_coords.permute(
                                                          (1, 2, 0)).reshape((-1, 3)))
    gt_index_space = convert_from_ants_space_tensor(template_parameters=ls_template_parameters,
                                                    physical_pts=gt_coords.permute(
                                                        (1, 2, 0)).reshape((-1, 3)))

    pred_ccf_annot = get_ccf_annotations(ccf_annotations, pred_index_space,
                                         return_np=False).reshape(
        pred_coords.shape[1:])
    pred_ccf_annot[pad_mask] = 0
    gt_ccf_annot = get_ccf_annotations(ccf_annotations, gt_index_space, return_np=False).reshape(
        gt_coords.shape[1:])
    gt_ccf_annot[pad_mask] = 0


    fig = create_diagnostic_image(
        input_image=input_image,
        slice_idx=slice_idx,
        errors=errors,
        pred_ccf_annotations=pred_ccf_annot.cpu().numpy(),
        gt_ccf_annotations=gt_ccf_annot.cpu().numpy(),
        iteration=iteration,
        pred_template_points=pred_coords,
        gt_template_points=gt_coords,
        gt_mask=pad_mask.cpu().bool().numpy(),
        pred_mask=pred_tissue_mask.cpu().bool().numpy() if pred_tissue_mask is not None else None,
    )
    return fig

def evaluate_batch(
    model: UNet,
    val_loader: DataLoader,
    device: str,
    ls_template: np.ndarray,
    ccf_annotations: np.ndarray,
    ls_template_parameters: AntsImageParameters,
    viz_indices: list[int],
    iteration: int,
    is_train: bool,
    autocast_context: ContextManager = nullcontext(),
    predict_tissue_mask: bool = True,
):
    model.eval()

    rmse = MeanSquaredError(squared=False).to(device)
    rmse_tissue_only = MeanSquaredError(squared=False).to(device)
    tissue_mask_dice = DiceScore(
        num_classes=1,
        include_background=False,
    ).to(device)

    sample_count = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluation")):
        input_images, target_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, tissue_masks, pad_masks = batch
        input_images, target_template_points, pad_masks, tissue_masks = input_images.to(device), target_template_points.to(device), pad_masks.to(device), tissue_masks.to(device)

        pad_masks = pad_masks.bool()
        tissue_masks = tissue_masks.bool()

        with autocast_context:
            model_out = model(input_images)

        if predict_tissue_mask:
            pred_coords, segmentation_logits = model_out[:, :-1], model_out[:, -1]
        else:
            pred_coords, segmentation_logits = model_out, None

        error = (pred_coords - target_template_points) ** 2

        # flip pred along AP axis if closer to GT for sagittal slices only
        pred_flipped = mirror_points(
            points=pred_coords,
            ml_dim_size=ls_template.shape[0],
            template_parameters=ls_template_parameters
        )
        flipped_error = (pred_flipped - target_template_points) ** 2

        sagittal_mask = torch.tensor(
            [o == SliceOrientation.SAGITTAL.value for o in orientations],
            device=device, dtype=torch.bool
        )
        use_flipped = (flipped_error < error) & sagittal_mask.view(-1, 1, 1, 1)
        pred_coords = torch.where(use_flipped, pred_flipped, pred_coords)

        rmse.update(
            preds=pred_coords[torch.stack([pad_masks]*3, dim=1)],
            target=target_template_points[torch.stack([pad_masks]*3, dim=1)]
        )
        rmse_tissue_only.update(
            preds=pred_coords[torch.stack([tissue_masks]*3, dim=1)],
            target=target_template_points[torch.stack([tissue_masks]*3, dim=1)]
        )
        if predict_tissue_mask:
            pred_tissue_masks = (F.sigmoid(segmentation_logits) > 0.5).to(torch.uint8)
            tissue_mask_dice.update(preds=pred_tissue_masks, target=tissue_masks)
        else:
            pred_tissue_masks = None

        B = input_images.shape[0]
        for sample_idx in range(B):
            for viz_index in viz_indices:
                if sample_count == viz_index:
                    fig = viz_sample(
                        input_image=input_images[sample_idx].cpu().squeeze(0),
                        slice_idx=slice_indices[sample_idx],
                        errors=error[sample_idx].cpu().numpy(),
                        iteration=iteration,
                        pred_coords=pred_coords[sample_idx].cpu(),
                        gt_coords=target_template_points[sample_idx].cpu(),
                        pad_mask=pad_masks[sample_idx],
                        pred_tissue_mask=pred_tissue_masks[sample_idx] if pred_tissue_masks is not None else None,
                        ls_template_parameters=ls_template_parameters,
                        ccf_annotations=ccf_annotations,
                    )
                    fig_filename = f"slice_{slice_indices[sample_idx]}_y_{patch_ys[sample_idx]}_x_{patch_xs[sample_idx]}_step_{iteration}.png"
                    mlflow.log_figure(fig,
                                      f"inference/{"train" if is_train else "val"}/slice_{slice_indices[sample_idx]}/y_{patch_ys[sample_idx]}_x_{patch_xs[sample_idx]}/{fig_filename}")
                    plt.close(fig)
            sample_count += 1

    # *1000 to convert to micron
    rmse = rmse.compute().item() * 1000
    rmse_tissue_only = rmse_tissue_only.compute().item() * 1000

    if predict_tissue_mask:
        tissue_mask_dice = tissue_mask_dice.compute().item()
    else:
        tissue_mask_dice = None

    return rmse, rmse_tissue_only, tissue_mask_dice