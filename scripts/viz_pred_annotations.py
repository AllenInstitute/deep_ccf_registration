import multiprocessing

import cv2
import plotly
import tensorstore
from matplotlib.pyplot import title
from skimage.measure import find_contours

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


def unpad_and_resize(
    annotations: np.ndarray,
    orig_h: int,
    orig_w: int,
    native_res_um: tuple[float, float],
    resampled_resolution: float = 30.0,
):
    scale_y = native_res_um[0] / resampled_resolution
    scale_x = native_res_um[1] / resampled_resolution

    h_resized = round(orig_h * scale_y)
    w_resized = round(orig_w * scale_x)

    # crop padding (top-left aligned → padding is on right/bottom)
    cropped = annotations[:h_resized, :w_resized]

    # resize to original image space
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def seg_plot(
    gt_points: tuple[np.ndarray, ...],
    pred_points: tuple[np.ndarray, ...],
    fn_points: tuple[np.ndarray, ...],
    fn_pred_ids: np.ndarray,
    subject_meta: SubjectMetadata,
    terminology: pd.DataFrame,
    stride: int = 1,
):
    fig = plotly.graph_objects.Figure()
    fig.add_trace(plotly.graph_objects.Scatter3d(
        x=gt_points[0][::stride], y=gt_points[1][::stride], z=gt_points[2][::stride],
        mode='markers',
        marker=dict(size=1, color='green', opacity=1.0),
        name='gt'
    ))
    fig.add_trace(plotly.graph_objects.Scatter3d(
        x=pred_points[0][::stride], y=pred_points[1][::stride], z=pred_points[2][::stride],
        mode='markers',
        marker=dict(size=1, color='blue', opacity=1.0),
        name='pred'
    ))

    # FN trace
    id_to_name = dict(zip(terminology['annotation_value'], terminology['abbreviation']))

    fn_names = np.array([
        'background' if int(i) == 0 else id_to_name[i]
        for i in fn_pred_ids[::stride]
    ])

    fig.add_trace(plotly.graph_objects.Scatter3d(
        x=fn_points[0][::stride], y=fn_points[1][::stride], z=fn_points[2][::stride],
        mode='markers',
        marker=dict(size=1, color=fn_pred_ids[::stride], colorscale='Viridis', opacity=1.0, colorbar=dict(title='Pred CCF ID')),
        name='FN',
        text=fn_names,
        hoverinfo='text',
    ))

    # opacity sliders
    opacity_steps = np.round(np.arange(0, 1.05, 0.1), 1)

    gt_slider = dict(
        steps=[
            dict(method='restyle', args=[{'marker.opacity': [v]}, [0]], label=str(v))
            for v in opacity_steps
        ],
        currentvalue=dict(prefix='GT opacity: '),
        x=0.0, len=0.45, y=0.0,
        active=10,
    )

    pred_slider = dict(
        steps=[
            dict(method='restyle', args=[{'marker.opacity': [v]}, [1]], label=str(v))
            for v in opacity_steps
        ],
        currentvalue=dict(prefix='Pred opacity: '),
        x=0.55, len=0.45, y=0.0,
        active=10,
    )

    fn_slider = dict(
        steps=[
            dict(method='restyle', args=[{'marker.opacity': [v]}, [2]], label=str(v))
            for v in opacity_steps
        ],
        currentvalue=dict(prefix='FN opacity: '),
        x=0.275, len=0.45, y=0.1,
        active=10,
    )

    fig.update_layout(
        sliders=[gt_slider, pred_slider, fn_slider],
        scene=dict(
            xaxis=dict(range=[0, ls_template.shape[0]]),
            yaxis=dict(range=[0, ls_template.shape[0]]),
            zaxis=dict(range=[0, ls_template.shape[0]]),
        ),
        margin=dict(b=120),
    )

    return fig

@torch.no_grad()
def viz_ccf_annotations(
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
    pred_tissue_masks = []

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

        model_out = model(input_images)
        pred_template_points = model_out[:, :-1]
        batch_pred_tissue_masks = F.sigmoid(model_out[:, -1]) > 0.5

        pred_template_points = pred_template_points.cpu()
        target_template_points = target_template_points.cpu()

        batch_gt_coords_index = physical_to_index_space(
            physical_pts=target_template_points,
            template_parameters=ls_template_parameters,
            channel_dim=1
        )

        pred_template_points = calc_lowest_err_sagittal_orientation(
            pred=pred_template_points,
            target=target_template_points,
            template_parameters=ls_template_parameters,
            orientations=[batch["orientations"][i] for i in
                          range(pred_template_points.shape[0])],
        )

        batch_pred_coords_index = physical_to_index_space(
            physical_pts=pred_template_points,
            template_parameters=ls_template_parameters,
            channel_dim=1
        )

        gt_coords.append(batch_gt_coords_index)
        pred_coords.append(batch_pred_coords_index)
        pred_tissue_masks.append(batch_pred_tissue_masks)

    subject_gt_coords = np.concatenate(gt_coords)
    subject_pred_coords = np.concatenate(pred_coords)
    subject_pred_tissue_masks = np.concatenate(pred_tissue_masks)

    subject_gt_ccf_annotations = np.stack([
        map_coordinates(
            input=ccf_annotations,
            coordinates=subject_gt_coords[i].reshape(3, -1),
            order=0,
            mode='nearest'
        ).reshape(subject_gt_coords.shape[2:])
        for i in range(subject_pred_coords.shape[0])
    ])

    dice_metric = SparseDiceMetric(
        class_ids=np.array([structure_id]),
        terminology_path=Path('/Users/adam.amster/smartspim-registration/terminology.csv'),
        terminology_correction_path=Path('/Users/adam.amster/smartspim-registration/missing_ccf_ids_mapped.json')
    )
    ccf_ids = dice_metric._parent_to_children_ids[structure_id]

    subject_pred_ccf_annotations = np.stack([
        map_coordinates(
            input=ccf_annotations,
            coordinates=subject_pred_coords[i].reshape(3, -1),
            order=0,
            mode='nearest'
        ).reshape(subject_pred_coords.shape[2:])
        for i in range(subject_pred_coords.shape[0])
    ])

    subject_pred_ccf_annotations[~subject_pred_tissue_masks] = 0

    dice_metric.update(pred=subject_pred_ccf_annotations, target=subject_gt_ccf_annotations)
    dice_metric = dice_metric.compute()
    print(f'dice: {dice_metric}')

    gt_mask = np.isin(subject_gt_ccf_annotations, ccf_ids)
    pred_mask = np.isin(subject_pred_ccf_annotations, ccf_ids)

    gt_points = tuple(
        subject_gt_coords[:, dim][gt_mask] for dim in range(3)
    )
    pred_points = tuple(
        subject_pred_coords[:, dim][pred_mask] for dim in range(3)
    )

    fn_mask = gt_mask & ~pred_mask
    fn_points = tuple(
        subject_gt_coords[:, dim][fn_mask] for dim in range(3)
    )
    fn_pred_ids = subject_pred_ccf_annotations[fn_mask]

    terminology = pd.read_csv('~/smartspim-registration/terminology.csv')

    seg_fig = seg_plot(
        gt_points=gt_points,
        pred_points=pred_points,
        subject_meta=subject_meta,
        fn_points=fn_points,
        fn_pred_ids=fn_pred_ids,
        terminology=terminology,
    )
    seg_fig.write_html(f'/tmp/{structure_id}_seg.html')


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

    viz_ccf_annotations(val_samples=val_samples, structure_id=908, subject_id=subject_id)

if __name__ == '__main__':
    main(subject_id=val_subjects_sample[0], subject_metadata=subject_metadata,)
