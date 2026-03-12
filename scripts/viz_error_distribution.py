import multiprocessing
from typing import Optional

import cv2
import plotly
import tensorstore
from matplotlib.pyplot import title
from skimage.measure import find_contours

from deep_ccf_registration.utils.losses import MSE

multiprocessing.set_start_method('spawn', force=True)  # tensorstore complains "fork" not allowed
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters

from torch.utils.data import DataLoader
from loguru import logger
import json
import ants
from tqdm import tqdm
from scipy.ndimage import map_coordinates, gaussian_filter
import torch.nn.functional as F

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.collation import collate_patch_samples
from deep_ccf_registration.datasets.slice_dataset import SubjectSliceDataset
from deep_ccf_registration.datasets.transforms import build_transform
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, RotationAngles, \
    SubjectRotationAngle
from deep_ccf_registration.models import UNetWithRegressionHeads
from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.datasets.transforms import physical_to_index_space
from deep_ccf_registration.utils.metrics import SparseDiceMetric, \
    calc_lowest_err_sagittal_orientation

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)

np.random.seed(1234)

with open('/Users/adam.amster/smartspim-registration/subject_metadata_local.json') as f:
    subject_metadata = json.load(f)
subject_metadata = [SubjectMetadata.model_validate(x) for x in subject_metadata]

val = np.load('/Users/adam.amster/smartspim-registration/val.npy')
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
        orientation: SliceOrientation,
        tissue_bboxes: pd.DataFrame,
        subject_id: str,
):
    samples = []
    for subject in tqdm([subject_id]):
        subject_meta = [x for x in subjects_meta if x.subject_id == subject][0]
        slice_axis = subject_meta.get_slice_axis(orientation=orientation)
        slice_dim = subject_meta.registered_shape[slice_axis.dimension]
        subject_bboxes = tissue_bboxes.loc[subject]
        indices = list(range(subject_bboxes['index'].min(), subject_bboxes['index'].max() + 1))
        subject_samples = np.array(list(zip(
            [subject_meta.subject_id] * slice_dim,
            indices,
            [orientation.value] * slice_dim
        )))
        samples.append(subject_samples)
    samples = np.concatenate(samples)
    return samples


ccf_annotations = ants.image_read('/Users/adam.amster/smartspim-registration/ccf_annotation_to_template_moved_25.nii.gz').numpy()
np.save('/tmp/ccf_annotation_to_template_moved_25.npy', ccf_annotations)

ls_template = ants.image_read(filename='/Users/adam.amster/.transform_utils/transform_utils/smartspim_lca/template/smartspim_lca_template_25.nii.gz')
ls_template_ants_parameters = AntsImageParameters.from_ants_image(image=ls_template)
ls_template_parameters = TemplateParameters(
    origin=ls_template_ants_parameters.origin,
    scale=ls_template_ants_parameters.scale,
    direction=ls_template_ants_parameters.direction,
    shape=ls_template.shape,
    orientation=ls_template_ants_parameters.orientation,
)

with open('/Users/adam.amster/smartspim-registration/config_dev.json') as f:
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

checkpoint = torch.load(f='/Users/adam.amster/Downloads/58000.pt', map_location='cpu')
model.load_state_dict(state_dict=checkpoint['model_state_dict'])

# not acutally used, but must pass
rotation_angles = pd.read_csv(config.rotation_angles_path).set_index('subject_id')
rotation_angles = RotationAngles(
    rotation_angles={x.Index: SubjectRotationAngle(AP_rot=x.AP_rotation, ML_rot=x.ML_rotation,
                                                   SI_rot=x.SI_rotation) for x in
                     rotation_angles.itertuples(index=True)},
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
        reduce: Optional[str] = 'median'
):
    assert len(pred.shape) == 4 and pred.shape[1] == 3
    assert len(target.shape) == 4 and target.shape[1] == 3
    if mask is not None:
        assert len(mask.shape) == 3 and list(mask.shape) == [pred.shape[0]] + list(pred.shape[-2:])
    pred = calc_lowest_err_sagittal_orientation(
        pred=pred, target=target, template_parameters=ls_template_parameters,
        orientations=orientations,
    )
    error = torch.sqrt(((pred - target) ** 2).sum(dim=1)) * 1000

    if reduce == 'median':
        n_slices = error.shape[0]
        error = torch.stack([error[x][mask.bool()[x]].median() for x in range(n_slices)])

    return error


def plot_error_distribution(
        coords: torch.Tensor,
        error: torch.Tensor,
        mask: torch.Tensor,
        template_parameters,
        point_size: float = 2.0,
        stride: int = 400,
) -> plotly.graph_objs.Figure:
    coords_index = physical_to_index_space(
        physical_pts=coords,
        template_parameters=template_parameters,
        channel_dim=1,
    )

    mask_np = mask.bool().numpy()
    x = coords_index[:, 0].numpy()[mask_np][::stride]
    y = coords_index[:, 1].numpy()[mask_np][::stride]
    z = coords_index[:, 2].numpy()[mask_np][::stride]
    errors = error.numpy()[mask_np][::stride]

    fig = plotly.graph_objs.Figure(
        data=plotly.graph_objs.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=point_size,
                color=errors,
                colorscale='turbo',
                opacity=0.6,
                colorbar=dict(title="error (microns)", thickness=20),
                cmin=np.percentile(errors, 1),
                cmax=np.percentile(errors, 99),
            ),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, ls_template.shape[0]]),
            yaxis=dict(range=[0, ls_template.shape[0]]),
            zaxis=dict(range=[0, ls_template.shape[0]]),
            xaxis_title="ML", yaxis_title="AP", zaxis_title="SI"
        ),
        margin=dict(b=120),
    )

    return fig


@torch.no_grad()
def viz_error_distribution(
    val_samples: np.ndarray,
    subject_id: str,
    structure_id: int,
    device='cpu',
    stride=1,
    orientation: SliceOrientation = SliceOrientation.SAGITTAL,
    resampled_resolution: float = 30,
):
    model.eval()

    gt_coords = []
    pred_coords = []
    tissue_masks = []

    logger.info(f'Run inference for {subject_id}')

    subject_meta = [x for x in subject_metadata if x.subject_id == subject_id][0]

    subject_val_samples = val_samples[val_samples[:, 0] == subject_id]

    val_dataloader = create_dataloader(
        subjects=subject_metadata,
        samples=subject_val_samples,
        tissue_bboxes_path=Path(
            '/Users/adam.amster/smartspim-registration/tissue_bounding_boxes.parquet'),
        batch_size=16,
        num_workers=4,
        ls_template_parameters=ls_template_parameters,
        ccf_annotations_path='/tmp/ccf_annotation_to_template_moved_25.npy',
        include_tissue_mask=True,
        config=config,
        rotation_angles=rotation_angles
    )

    for batch in tqdm(val_dataloader):
        input_images = batch["input_images"].to(device)
        target_template_points = batch["target_template_points"].to(device)
        batch_tissue_masks = batch["tissue_masks"].to(device)

        model_out = model(input_images)
        pred_template_points = model_out[:, :-1]

        pred_template_points = pred_template_points.cpu()
        target_template_points = target_template_points.cpu()

        pred_template_points = calc_lowest_err_sagittal_orientation(
            pred=pred_template_points,
            target=target_template_points,
            template_parameters=ls_template_parameters,
            orientations=[batch["orientations"][i] for i in
                          range(pred_template_points.shape[0])],
        )

        gt_coords.append(target_template_points)
        pred_coords.append(pred_template_points)
        tissue_masks.append(batch_tissue_masks)

    subject_gt_coords = torch.cat(gt_coords)
    subject_pred_coords = torch.cat(pred_coords)
    subject_tissue_masks = torch.cat(tissue_masks)

    error = calc_error(
        pred=subject_pred_coords,
        target=subject_gt_coords,
        mask=subject_tissue_masks,
        orientations=[orientation],
        reduce=None,
    )

    fig = plot_error_distribution(
        coords=subject_pred_coords,
        template_parameters=ls_template_parameters,
        error=error,
        mask=subject_tissue_masks,
    )
    fig.write_html(f'/tmp/{subject_id}_error.html')


def main(subject_id: str, subject_metadata: list[SubjectMetadata]):
    tissue_bboxes = pd.read_parquet(
        '/Users/adam.amster/smartspim-registration/tissue_bounding_boxes.parquet').set_index(
        'subject_id')

    val_samples = create_val_samples(
        subjects_meta=subject_metadata,
        orientation=SliceOrientation.SAGITTAL,
        tissue_bboxes=tissue_bboxes,
        subject_id=subject_id,
    )

    viz_error_distribution(val_samples=val_samples, structure_id=908, subject_id=subject_id)

if __name__ == '__main__':
    main(subject_id=val_subjects_sample[0], subject_metadata=subject_metadata,)
