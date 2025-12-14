from contextlib import nullcontext
from itertools import islice
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
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
import  torch.nn.functional as F

from deep_ccf_registration.datasets.slice_dataset import SliceDataset
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.metrics.point_wise_rmse import PointwiseRMSE
from deep_ccf_registration.utils.logging_utils import ProgressLogger
from deep_ccf_registration.utils.transforms import convert_from_ants_space_tensor, mirror_points
from deep_ccf_registration.utils.utils import get_ccf_annotations
from deep_ccf_registration.utils.visualization import create_diagnostic_image
from deep_ccf_registration.utils.dataloading import BatchPrefetcher


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
    tissue_mask: Optional[torch.Tensor] = None,
    exclude_background: bool = False
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
    if exclude_background:
        pred_ccf_annot[~tissue_mask] = 0
    else:
        pred_ccf_annot[~pad_mask] = 0
    gt_ccf_annot = get_ccf_annotations(ccf_annotations, gt_index_space, return_np=False).reshape(
        gt_coords.shape[1:])

    if exclude_background:
        gt_ccf_annot[~tissue_mask] = 0
    else:
        gt_ccf_annot[~pad_mask] = 0


    fig = create_diagnostic_image(
        input_image=input_image,
        slice_idx=slice_idx,
        errors=errors,
        pred_ccf_annotations=pred_ccf_annot.cpu().numpy(),
        gt_ccf_annotations=gt_ccf_annot.cpu().numpy(),
        iteration=iteration,
        pred_template_points=pred_coords,
        gt_template_points=gt_coords,
        pad_mask=pad_mask.cpu().bool().numpy(),
        exclude_background=exclude_background,
        tissue_mask=tissue_mask.cpu().bool().numpy(),
        pred_mask=pred_tissue_mask.cpu().bool().numpy() if pred_tissue_mask is not None else None,
    )
    return fig

def evaluate_batch(
    model: UNet,
    dataloader: DataLoader,
    device: str,
    ccf_annotations: np.ndarray,
    ls_template_parameters: AntsImageParameters,
    iteration: int,
    is_train: bool,
    autocast_context: ContextManager = nullcontext(),
    predict_tissue_mask: bool = True,
    exclude_background_pixels: bool = False,
    max_iters: int = 200,
    log_every: int = 20
):
    viz_indices = np.arange(len(dataloader.dataset))
    np.random.shuffle(viz_indices)
    viz_indices = viz_indices[:10]

    model.eval()

    rmse = PointwiseRMSE().to(device)
    rmse_tissue_only = PointwiseRMSE().to(device)
    tissue_mask_dice = DiceScore(
        num_classes=1,
        include_background=False,
    ).to(device)

    sample_count = 0

    limited_batches = islice(dataloader, max_iters)
    total_iters = min(len(dataloader), max_iters)
    progress = ProgressLogger(desc='Evaluation', total=total_iters, log_every=min(total_iters, log_every))

    for batch_idx, batch in enumerate(limited_batches):
        input_images, target_template_points, dataset_indices, slice_indices, patch_ys, patch_xs, orientations, input_image_transforms, tissue_masks, pad_masks, subject_ids = batch
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

        rmse.update(
            preds=pred_coords,
            target=target_template_points,
            mask=pad_masks,
        )
        if predict_tissue_mask:
            rmse_tissue_only.update(
                preds=pred_coords,
                target=target_template_points,
                mask=tissue_masks,
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
                        exclude_background=exclude_background_pixels,
                        tissue_mask=tissue_masks[sample_idx]
                    )
                    subject_id = subject_ids[sample_idx]
                    fig_filename = f"subject_{subject_id}_slice_{slice_indices[sample_idx]}_y_{patch_ys[sample_idx]}_x_{patch_xs[sample_idx]}_step_{iteration}.png"
                    mlflow.log_figure(fig,
                                      f"inference/{"train" if is_train else "val"}/iteration/{iteration}/{fig_filename}")
                    plt.close(fig)
            sample_count += 1
        progress.log_progress()

    # *1000 to convert to micron
    rmse = rmse.compute().item() * 1000

    if predict_tissue_mask:
        rmse_tissue_only = rmse_tissue_only.compute().item() * 1000
    else:
        rmse_tissue_only = None

    if predict_tissue_mask:
        tissue_mask_dice = tissue_mask_dice.compute().item()
    else:
        tissue_mask_dice = None

    return rmse, rmse_tissue_only, tissue_mask_dice