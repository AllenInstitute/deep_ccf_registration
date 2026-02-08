import json
import os
import sys

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters
from aind_smartspim_transform_utils.utils.utils import convert_from_ants_space
from loguru import logger

from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.datasets.template_meta import TemplateParameters
from deep_ccf_registration.metadata import SubjectMetadata, TissueBoundingBoxes, RotationAngles, \
    SubjectRotationAngle
from deep_ccf_registration.run import create_dataloader


logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.add(sys.stderr, level=log_level)



def main(resample_to_fixed_resolution: int = None):
    with open('/Users/adam.amster/smartspim-registration/dataset_meta-test.json') as f:
        meta = json.load(f)
    meta = [SubjectMetadata(**x) for x in meta]

    with open('/Users/adam.amster/smartspim-registration/tissue_bboxes.json') as f:
        bboxes = json.load(f)

    bboxes = TissueBoundingBoxes(bounding_boxes=bboxes)
    with open('/Users/adam.amster/smartspim-registration/config_dev.json') as f:
        config = json.load(f)
    config = TrainConfig(**config)

    ls_template = ants.image_read(filename="/Users/adam.amster/.transform_utils/transform_utils/smartspim_lca/template/smartspim_lca_template_25.nii.gz")
    ls_template_ants_parameters = AntsImageParameters.from_ants_image(image=ls_template)
    ls_template_parameters = TemplateParameters(
        origin=ls_template_ants_parameters.origin,
        scale=ls_template_ants_parameters.scale,
        direction=ls_template_ants_parameters.direction,
        shape=ls_template.shape,
        orientation=ls_template_ants_parameters.orientation,
    )
    ccf_annotations = ants.image_read(str(config.ccf_annotations_path)).numpy()

    with open(config.rotation_angles_path) as f:
        rotation_angles = pd.read_csv(f).set_index('subject_id')
    rotation_angles = RotationAngles(
        rotation_angles={x.Index: SubjectRotationAngle(AP_rot=x.AP_rotation, ML_rot=x.ML_rotation, SI_rot=x.SI_rotation) for x in rotation_angles.itertuples(index=True)},
        SI_range=(
            rotation_angles['SI_rotation'].mean() - rotation_angles['SI_rotation'].std()*2,
            rotation_angles['SI_rotation'].mean() + rotation_angles['SI_rotation'].std()*2
        ),
        ML_range=(
            rotation_angles['ML_rotation'].mean() - rotation_angles['ML_rotation'].std()*2,
            rotation_angles['ML_rotation'].mean() + rotation_angles['ML_rotation'].std()*2
        ),
        AP_range=(
            rotation_angles['AP_rotation'].mean() - rotation_angles['AP_rotation'].std()*2,
            rotation_angles['AP_rotation'].mean() + rotation_angles['AP_rotation'].std()*2
        ),
    )

    config.data_augmentation.apply_square_symmetry_transform = False
    config.data_augmentation.rotate_slices = False
    config.normalize_template_points = False
    config.data_augmentation.apply_grid_distortion = False
    config.seed = None
    config.debug_slice_idx = 709

    config.patch_size = (512, 512)
    config.resample_to_fixed_resolution = resample_to_fixed_resolution

    dataloader = create_dataloader(
        metadata=meta,
        tissue_bboxes=bboxes,
        config=config,
        is_train=True,
        batch_size=1,
        ls_template_parameters=ls_template_parameters,
        num_workers=0,
        device='cpu',
        ccf_annotations=ccf_annotations,
        rotation_angles=rotation_angles,
        include_tissue_mask=config.predict_tissue_mask
    )

    fig = plt.figure(figsize=(20, 10))

    # Create 2 rows: first row for 2D images, second row for 3D plots
    ax1 = []
    ax2 = []

    for i in range(6):
        # First row: 2D image subplots
        ax1.append(fig.add_subplot(2, 6, i + 1))
        # Second row: 3D scatter subplots
        ax2.append(fig.add_subplot(2, 6, i + 7, projection='3d'))

    for i in range(6):
        if i == 1:
            #config.data_augmentation.rotate_slices = True
            #config.data_augmentation.apply_square_symmetry_transform = True
            config.data_augmentation.apply_grid_distortion = True

            dataloader = create_dataloader(
                metadata=meta,
                tissue_bboxes=bboxes,
                config=config,
                is_train=True,
                batch_size=1,
                ls_template_parameters=ls_template_parameters,
                num_workers=0,
                device='cpu',
                ccf_annotations=ccf_annotations,
                rotation_angles=rotation_angles,
                include_tissue_mask=config.predict_tissue_mask,
            )

        batch = next(iter(dataloader))
        img = batch['input_images'][0].squeeze()
        tissue_mask = batch['tissue_masks'][0]

        ax1[i].imshow(img, cmap='gray')
        ax1[i].axis('off')

        template_points = batch["target_template_points"][0]

        gt_template_points_index_space = convert_from_ants_space(
            template_parameters=ls_template_parameters,
            physical_pts=template_points.permute((1, 2, 0)).view(-1,3).numpy()
        )

        logger.info(f'ML range: {gt_template_points_index_space[:, 0].min():.3f}-{gt_template_points_index_space[:, 0].max():.3f}')

        Y, X = np.meshgrid(np.arange(template_points.shape[1]),
                           np.arange(template_points.shape[2]),
                           indexing='ij')
        ax1[i].contour(X, Y, tissue_mask, levels=[0.5], colors='yellow', linewidths=2)

        ML_axis, AP_axis, SI_axis = 0, 1, 2

        ml_lim = [0, ls_template.shape[ML_axis]]
        ap_lim = [0, ls_template.shape[AP_axis]]
        si_lim = [0, ls_template.shape[SI_axis]]

        intensities = batch['input_images'][0].flatten()
        idx = np.arange(0, len(intensities))

        ax2[i].scatter(gt_template_points_index_space[idx, ML_axis],
                       gt_template_points_index_space[idx, SI_axis],
                       gt_template_points_index_space[idx, AP_axis],
                       c=intensities[idx], cmap='gray', s=0.5)
        ax2[i].view_init(elev=20, azim=80)
        ax2[i].set_xlim(ml_lim)
        ax2[i].set_ylim(si_lim)
        ax2[i].set_zlim(ap_lim)
        ax2[i].set_xlabel('ML')
        ax2[i].set_ylabel('SI')
        ax2[i].set_zlabel('AP')
        ax2[i].set_box_aspect([ls_template.shape[ML_axis],
                               ls_template.shape[SI_axis],
                               ls_template.shape[AP_axis]])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(resample_to_fixed_resolution=30)