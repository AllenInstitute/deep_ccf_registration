from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from pathlib import Path

from deep_ccf_registration.metadata import SliceOrientation


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # number of channels in each layer of the unet in the encoder
    unet_channels: tuple[int, ...]
    unet_stride: tuple[int, ...]
    unet_dropout: float = 0.0
    unet_feature_channels: int = 64
    unet_head_size: str = "small"
    unet_use_positional_encoding: bool = True
    unet_pos_encoding_channels: int = 16

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
    crop_warp_to_bounding_box: bool = True
    patch_size: Optional[tuple[int, int]] = None

    # downsample input image and target points so that longest max size is this
    longest_max_size: Optional[int] = None

    # make all input images have uniform orientation
    normalize_orientation: bool = True
    # make all input images range from [0, 1] using lower and upper percentiles as thresholds
    normalize_input_image: bool = True
    # normalize template points to [-1,1] (background points will be outside this range)
    normalize_template_points: bool = True
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(0, ge=0)
    exclude_background_pixels: bool = False
    predict_tissue_mask: bool = True
    dataloader_prefetch_factor: Optional[int] = None

    load_checkpoint: Optional[Path] = None

    max_iters: int
    learning_rate: float = Field(0.001, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    decay_learning_rate: bool = True
    warmup_iters: int = Field(1000, ge=0)
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
    seed: int = 1234
    debug: bool = False

    # path to bounding boxes conforming to schema: dict[str, Optional[TissueBoundingBox]]
    # mapping subject id to list of optional bounding boxes. bbox may be null if no tissue in slice
    # bboxes ordered by index across a certain axis
    # TODO handle different orientations. Currently only sagittal
    tissue_bounding_boxes_path: Path

    mlflow_experiment_name: str = "slice_registration"

    mlflow_tracking_uri: Optional[str] = None

    use_mlflow: bool = True

    model: ModelConfig