from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, Literal, Union
from pathlib import Path

from deep_ccf_registration.metadata import SliceOrientation
from deep_ccf_registration.models import PositionalEmbeddingType, PositionalEmbeddingPlacement


class LRScheduler(Enum):
    ReduceLROnPlateau = "ReduceLROnPlateau"
    CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    CosineAnnealingLR = "CosineAnnealingLR"

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # SMP Unet backbone configuration
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    encoder_depth: int = 5
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)
    decoder_use_norm: Union[bool, str] = "batchnorm"

    feature_channels: int = 64
    pos_encoding_channels: int = 16
    positional_embedding_type: Optional[PositionalEmbeddingType] = None
    positional_embedding_placement: Optional[PositionalEmbeddingPlacement] = None
    coord_head_channels: tuple[int, ...]

    @model_validator(mode='after')
    def validate_decoder_channels_length(self):
        if len(self.decoder_channels) != self.encoder_depth:
            raise ValueError(
                f"decoder_channels length ({len(self.decoder_channels)}) "
                f"must equal encoder_depth ({self.encoder_depth})"
            )
        return self

class DataAugmentationConfig(BaseModel):
    # extract rotated slices
    rotate_slices: bool = False
    apply_square_symmetry_transform: bool = False
    apply_grid_distortion: bool = False

class TrainConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dataset_meta_path: Path
    train_val_split: float = Field(0.8, ge=0.0, le=1.0)
    ls_template_path: Path
    # ccf annotations moved to ls template space
    ccf_annotations_path: Path = Path("/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/ccf_annotation_to_template_moved_25.nii.gz")
    orientation: Optional[SliceOrientation] = None
    registration_downsample_factor: int = 3
    tensorstore_aws_credentials_method: str = "default"
    patch_size: tuple[int, int]

    # make all input images range from [0, 1] using lower and upper percentiles as thresholds
    normalize_input_image: bool = True
    # normalize template points to [-1,1] (background points will be outside this range)
    normalize_template_points: bool = True
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(0, ge=0)
    # Fraction of slices to randomly sample per subject per epoch (0.0 < x <= 1.0)
    subject_slice_fraction: float = Field(0.25, gt=0.0, le=1.0)
    # Number of subjects to group together (None = no grouping, use all subjects)
    subject_group_size: Optional[int] = Field(None, gt=0)
    exclude_background_pixels: bool = False
    predict_tissue_mask: bool = True
    dataloader_prefetch_factor: Optional[int] = None

    load_checkpoint: Optional[Path] = None
    # MLflow run ID to resume (for spot instance recovery)
    resume_mlflow_run_id: Optional[str] = None

    max_iters: int
    learning_rate: float = Field(0.001, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    eval_iters: int = 50
    val_viz_samples: int = Field(10, ge=0)

    eval_interval: int = Field(500, gt=0)
    patience: int = Field(10, gt=0)
    min_delta: float = Field(1e-4, ge=0.0)
    region_ccf_ids_map_path: Path

    model_weights_out_dir: Path = Path("./checkpoints")
    log_file: Optional[Path] = None

    device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    mixed_precision: bool = False
    seed: Optional[int] = 1234
    debug: bool = False

    # path to bounding boxes conforming to schema: dict[str, Optional[TissueBoundingBox]]
    # mapping subject id to list of optional bounding boxes. bbox may be null if no tissue in slice
    # bboxes ordered by index across a certain axis
    # TODO handle different orientations. Currently only sagittal
    tissue_bounding_boxes_path: Path

    # path to rotation angles to align input to template calculated from affine matrix
    # conforms to _SubjectRotationAngle
    rotation_angles_path: Path

    mlflow_experiment_name: str = "slice_registration"
    mlflow_tracking_uri: Optional[str] = None
    use_mlflow: bool = True

    model: ModelConfig

    lr_scheduler: Optional[LRScheduler] = None

    # Gradient clipping (max gradient norm). Set to None to disable.
    grad_clip_max_norm: Optional[float] = 1.0

    # Learning rate warmup steps. Linearly ramps LR from 0 to learning_rate over this many steps.
    warmup_steps: int = 0

    use_positional_encoding: bool = False
    gradient_accumulation_steps: int = 1

    # microns/px to resample to
    resample_to_fixed_resolution: Optional[int] = None
    debug_start_y: Optional[int] = None
    debug_start_x: Optional[int] = None
    debug_slice_idx: Optional[int] = None

    # base directory containing /volumes and /warps caches
    cache_dir: Path = Path('/data')

    tmp_path: Path = Path('/tmp')

    data_augmentation: DataAugmentationConfig