import multiprocessing
multiprocessing.set_start_method('spawn', force=True)   # tensorstore complains "fork" not allowed
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typing import Optional
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters

from torch.utils.data import DataLoader
from loguru import logger
import json
import ants
from tqdm import tqdm
from scipy.ndimage import map_coordinates

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.collation import collate_patch_samples
from deep_ccf_registration.datasets.slice_dataset import SubjectSliceDataset
from deep_ccf_registration.datasets.transforms import build_transform
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, RotationAngles, \
    SubjectRotationAngle
from deep_ccf_registration.models import UNetWithRegressionHeads
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.utils.metrics import _calc_lowest_err_sagittal_orientation
from deep_ccf_registration.datasets.transforms import get_template_point_normalization_inverse, \
    physical_to_index_space
import matplotlib.pyplot as plt
import seaborn as sns
from deep_ccf_registration.utils.metrics import SparseDiceMetric


logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)

np.random.seed(1234)

with open('/root/capsule/data/smartspim_dataset/subject_metadata.json') as f:
    subject_metadata = json.load(f)
subject_metadata = [SubjectMetadata.model_validate(x) for x in subject_metadata]

val = np.load('/scratch/val.npy')
val_subjects = np.unique(val[:, 0])
val_subjects_sample = np.random.choice(val_subjects, size=12, replace=False).tolist()

def create_dataloader(
    subjects: list[SubjectMetadata],
    samples: np.ndarray,
    rotation_angles,
    tissue_bboxes_path: Path,
    batch_size: int,
    num_workers: int,
    config: TrainConfig,
    ls_template_parameters: TemplateParameters,
    ccf_annotations_path: str,
    include_tissue_mask: bool = False,
):
    template_parameters = TemplateParameters(
        origin=ls_template_parameters.origin,
        scale=ls_template_parameters.scale,
        direction=ls_template_parameters.direction,
        shape=ls_template_parameters.shape,
        orientation=ls_template_parameters.orientation,
    )

    transform = build_transform(
        config=config,
        template_parameters=template_parameters,
        square_symmetry=False,
        resample_to_fixed_resolution=30,
        rotate_slices=False,
        normalize_template_points=False,
        apply_grid_distortion=False,
        rotation_angles=None,
    )

    dataset = SubjectSliceDataset(
        subjects=subjects,
        samples=samples,
        template_parameters=template_parameters,
        is_train=False,
        tissue_bboxes_path=tissue_bboxes_path,
        rotation_angles=rotation_angles,
        orientations=[SliceOrientation.SAGITTAL],
        crop_size=[512, 512],
        transform=transform,
        include_tissue_mask=include_tissue_mask,
        ccf_annotations_path=ccf_annotations_path,
        rotate_slices=False,
        is_debug=False,
        aws_credentials_method='ecs',
        tissue_bbox_area_rejection_percentile=0.0
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        collate_fn=collate_patch_samples,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return dataloader

def create_val_samples(
    subjects_meta: list[SubjectMetadata],
    val_subjects: list[str],
    orientation: SliceOrientation,
    tissue_bboxes: pd.DataFrame,
):
    samples = []
    for subject in tqdm(val_subjects):
        subject_meta = [x for x in subjects_meta if x.subject_id == subject][0]
        slice_axis = subject_meta.get_slice_axis(orientation=orientation)
        slice_dim = subject_meta.registered_shape[slice_axis.dimension]
        subject_bboxes = tissue_bboxes.loc[subject]
        indices = list(range(subject_bboxes['index'].min(), subject_bboxes['index'].max()+1))
        subject_samples = np.array(list(zip(
            [subject_meta.subject_id] * slice_dim,
            indices,
            [orientation.value] * slice_dim
            )))
        samples.append(subject_samples)
    samples = np.concatenate(samples)
    return samples

ccf_annotations = ants.image_read('/scratch/ccf_annotation_to_template_moved_25.nii.gz').numpy()
np.save('/scratch/ccf_annotation_to_template_moved_25.npy', ccf_annotations)

ls_template = ants.image_read(filename='/root/capsule/scratch/smartspim_lca_template_25.nii.gz')
ls_template_ants_parameters = AntsImageParameters.from_ants_image(image=ls_template)
ls_template_parameters = TemplateParameters(
    origin=ls_template_ants_parameters.origin,
    scale=ls_template_ants_parameters.scale,
    direction=ls_template_ants_parameters.direction,
    shape=ls_template.shape,
    orientation=ls_template_ants_parameters.orientation,
)


with open('/scratch/config_co.json') as f:
    config = json.load(f)
config = TrainConfig(**config)

model = UNetWithRegressionHeads(
    in_channels=1,
    out_coords=3,
    include_tissue_mask=True,
    use_positional_encoding=False,
    feature_channels=256,
    input_dims=[512, 512],
    coord_head_channels=[256],
    encoder_name='resnet34',
    encoder_weights='imagenet',
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_use_norm='batchnorm',
)

checkpoint = torch.load(f='/scratch/51000.pt')
model.load_state_dict(state_dict=checkpoint['model_state_dict'])
model = model.to('cuda')

# not acutally used, but must pass
rotation_angles = pd.read_csv(config.rotation_angles_path).set_index('subject_id')
rotation_angles = RotationAngles(
    rotation_angles={x.Index: SubjectRotationAngle(AP_rot=x.AP_rotation, ML_rot=x.ML_rotation, SI_rot=x.SI_rotation) for x in rotation_angles.itertuples(index=True)},
    SI_range=(
        rotation_angles['SI_rotation'].min(),
        rotation_angles['SI_rotation'].max()
    ),
    ML_range=(
        rotation_angles['ML_rotation'].min(),
        rotation_angles['ML_rotation'].max()
    ),
    AP_range=(
        rotation_angles['AP_rotation'].min(),
        rotation_angles['AP_rotation'].max()
    ),
)

def calc_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    orientations,
    reduce: str = 'median'
):
    assert len(pred.shape) == 4 and pred.shape[1] == 3
    assert len(target.shape) == 4 and target.shape[1] == 3
    if mask is not None:
        assert len(mask.shape) == 3 and list(mask.shape) == [pred.shape[0]] + list(pred.shape[-2:])
    pred = _calc_lowest_err_sagittal_orientation(
        pred=pred, target=target, template_parameters=ls_template_parameters,
        orientations=orientations,
    )
    error = torch.sqrt(((pred - target) ** 2).sum(dim=1)) * 1000

    if reduce == 'median':
        n_slices = error.shape[0]
        error = torch.stack([error[x][mask.bool()[x]].median() for x in range(n_slices)])

    return error

def _update_dice_metric(
    dice_metric: SparseDiceMetric,
    ccf_annotations: np.ndarray,
    gt_physical_space_points: np.ndarray,
    pred_physical_space: np.ndarray,
    template_parameters: TemplateParameters,
    pred_mask: Optional[np.ndarray] = None,
):
    gt_points_index_space = physical_to_index_space(
        physical_pts=np.transpose(gt_physical_space_points, (1, 2, 0)),
        template_parameters=template_parameters,
    )

    pred_points_index_space = physical_to_index_space(
        physical_pts=np.transpose(pred_physical_space, (1, 2, 0)),
        template_parameters=template_parameters,
    )

    gt_ccf_annotations = map_coordinates(
        input=ccf_annotations,
        # 3, N
        coordinates=np.transpose(gt_points_index_space, (2, 0, 1)).reshape(3, -1),
        order=0,
        mode='nearest'
    )
    pred_ccf_annotations = map_coordinates(
        input=ccf_annotations,
        coordinates=np.transpose(pred_points_index_space, (2, 0, 1)).reshape(3, -1),
        order=0,
        mode='nearest'
    )

    if pred_mask is not None:
        pred_ccf_annotations[np.where(~pred_mask.flatten().astype('bool'))] = 0

    dice_metric.update(pred=pred_ccf_annotations, target=gt_ccf_annotations)

@torch.no_grad()
def run_inference(device='cuda', orientation: SliceOrientation = SliceOrientation.SAGITTAL):
    model.eval()

    per_slice_median_err_mins = []
    per_slice_median_err_maxs = []
    subject_err_medians = []
    subject_errs = []
    per_point_template_indexs = []
    per_point_errs = []
    subject_large_structure_dice_per_slice = []
    subject_small_structure_dice_per_slice = []

    large_structure_names = ['CTX', 'CP', 'HPF', 'BS', 'CB', 'CBX']
    small_structure_names = ['act', 'fr', 'mtt', 'IPN', 'MH', 'LH']

    for subject in tqdm([val_subjects_sample[0]], desc=f'running inference'):
        logger.info(f'Run inference for {subject}')

        subject_val_samples = val_samples[val_samples[:, 0] == subject]

        ccf_annotations_large_structure_dice_metric = SparseDiceMetric(class_ids=[688, 672, 1089, 343, 512, 528], terminology_path=)
        ccf_annotations_small_structure_dice_metric = SparseDiceMetric(class_ids=[908, 595, 690, 100, 483, 186])

        val_dataloader = create_dataloader(
            subjects=subject_metadata,
            samples=subject_val_samples,
            tissue_bboxes_path=Path('/root/capsule/data/tissue_bounding_boxes/tissue_bounding_boxes.parquet'),
            batch_size=16,
            num_workers=4,
            ls_template_parameters=ls_template_parameters,
            ccf_annotations_path='/scratch/ccf_annotation_to_template_moved_25.npy',
            include_tissue_mask=True,
            config=config,
            rotation_angles=rotation_angles
        )

        subject_slice_median_errs = []
        subject_gt_template_coords_index_space = []

        for batch in tqdm(val_dataloader):
            input_images = batch["input_images"].to(device)
            target_template_points = batch["target_template_points"].to(device)
            tissue_masks = batch["tissue_masks"].to(device)

            model_out = model(input_images)
            pred_template_points = model_out[:, :-1]
            pred_tissue_masks = model_out[:, -1]

            pred_template_points = pred_template_points.cpu()
            target_template_points = target_template_points.cpu()
            tissue_masks = tissue_masks.cpu()
            pred_tissue_masks = pred_tissue_masks.cpu()
            n = len(input_images)

            batch_slice_median_err = calc_error(
                pred=pred_template_points,
                target=target_template_points,
                mask=tissue_masks,
                orientations=[orientation.value] * n
            )
            subject_slice_median_errs.append(batch_slice_median_err)

            batch_slice_err = calc_error(
                pred=pred_template_points,
                target=target_template_points,
                mask=tissue_masks,
                orientations=[orientation.value] * n,
                reduce=None
            )

            batch_coords_index = physical_to_index_space(
                physical_pts=target_template_points,
                template_parameters=ls_template_parameters,
                channel_dim=1
            )
            subject_gt_template_coords_index_space.append(batch_coords_index)

            batch_point_err = batch_slice_err[tissue_masks.bool()]

            if orientation == SliceOrientation.SAGITTAL:
                template_slice_idx = 0
            else:
                raise NotImplementedError

            batch_point_idx = batch_coords_index[:, template_slice_idx][tissue_masks.bool()].flatten()

            n_points = len(batch_point_idx)
            sample = np.random.choice(n_points, size=int(n_points * 0.001), replace=False)
            per_point_template_indexs.append(batch_point_idx[sample])
            per_point_errs.append(batch_point_err[sample])

            for sample_idx in range(n):
                for dice_metric in (ccf_annotations_large_structure_dice_metric, ccf_annotations_small_structure_dice_metric):
                    _update_dice_metric(
                        dice_metric=dice_metric,
                        ccf_annotations=ccf_annotations,
                        gt_physical_space_points=target_template_points[sample_idx].numpy(),
                        pred_physical_space=pred_template_points[sample_idx].numpy(),
                        template_parameters=ls_template_parameters,
                        pred_mask=pred_tissue_masks[sample_idx].numpy(),
                    )
        subject_gt_template_coords_index_space = torch.cat(subject_gt_template_coords_index_space)
        subject_ccf_annotations = map_coordinates(
            input=ccf_annotations,
            # 3, N
            coordinates=np.transpose(subject_gt_template_coords_index_space, (2, 0, 1)).reshape(3, -1),
            order=0,
            mode='nearest'
        )
        subject_slice_median_errs = torch.cat(subject_slice_median_errs)

        # nans excluded. tissue mask empty in some rare cases. Not sure why
        per_slice_median_err_mins.append(subject_slice_median_errs[~torch.isnan(subject_slice_median_errs)].min().item())
        per_slice_median_err_maxs.append(subject_slice_median_errs[~torch.isnan(subject_slice_median_errs)].max().item())

        subject_err_medians.append(subject_slice_median_errs.nanmedian().item())

        subject_err = pd.DataFrame(subject_slice_median_errs.numpy().flatten(), columns=['err'])
        subject_err['subject_id'] = subject
        subject_errs.append(subject_err)

        large_structure_dice_per_slice = np.stack(ccf_annotations_large_structure_dice_metric._sample_scores)
        subject_large_structure_dice_per_slice.append(large_structure_dice_per_slice)

        small_structure_dice_per_slice = np.stack(ccf_annotations_small_structure_dice_metric._sample_scores)
        subject_small_structure_dice_per_slice.append(small_structure_dice_per_slice)

    per_slice_median_err_min = min(per_slice_median_err_mins)
    per_slice_median_err_max = max(per_slice_median_err_maxs)
    subject_err_mean = torch.tensor(subject_err_medians).mean().item()
    subject_err_std = torch.tensor(subject_err_medians).std().item()
    subject_errs = pd.concat(subject_errs, ignore_index=True)

    subject_median_large_structure_dice = np.stack([
        np.nanmedian(s, axis=0) for s in subject_large_structure_dice_per_slice
    ])
    subject_median_small_structure_dice = np.stack([
        np.nanmedian(s, axis=0) for s in subject_small_structure_dice_per_slice
    ])

    print(f'per_slice_median_err_min: {per_slice_median_err_min}')
    print(f'per_slice_median_err_max: {per_slice_median_err_max}')
    print(f'subject_err_mean: {subject_err_mean}')
    print(f'subject_err_std: {subject_err_std}')

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=subject_errs, x='subject_id', y='err', ax=ax)
    ax.set_ylabel('Error (micron)')
    fig.savefig('/scratch/subject_distance_err.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=torch.cat(per_point_template_indexs).numpy(), y=torch.cat(per_point_errs).numpy(), ax=ax, s=1, alpha=0.3)
    ax.set_ylabel('Error (micron)')

    if orientation == SliceOrientation.SAGITTAL:
        ax.set_xlabel('ML position (index space)')
    else:
        raise NotImplementedError
    fig.savefig('/scratch/template_position_err.png')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.boxplot(data=subject_median_large_structure_dice, ax=ax[0])
    ax[0].set_xticklabels(large_structure_names)
    ax[0].set_ylabel('Dice score')
    ax[0].set_title('Region-wise median Dice score')

    sns.boxplot(y=np.nanmean(subject_median_large_structure_dice, axis=1), ax=ax[1])
    ax[1].set_ylabel('Dice score')
    ax[1].set_title('Average median Dice score')

    fig.savefig('/scratch/dice_large_structure.png')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.boxplot(data=subject_median_small_structure_dice, ax=ax[0])
    ax[0].set_xticklabels(small_structure_names)
    ax[0].set_ylabel('Dice score')
    ax[0].set_title('Region-wise median Dice score')

    sns.boxplot(y=np.nanmean(subject_median_small_structure_dice, axis=1), ax=ax[1])
    ax[1].set_ylabel('Dice score')
    ax[1].set_title('Average median Dice score')

    fig.savefig('/scratch/dice_small_structure.png')

if __name__ == '__main__':
    tissue_bboxes = pd.read_parquet('/root/capsule/data/tissue_bounding_boxes/tissue_bounding_boxes.parquet').set_index('subject_id')

    val_samples = create_val_samples(
        subjects_meta=subject_metadata,
        val_subjects=val_subjects_sample,
        orientation=SliceOrientation.SAGITTAL,
        tissue_bboxes=tissue_bboxes)

    run_inference()

