from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Tuple
from pathlib import Path

from deep_ccf_registration.metadata import SliceOrientation


class TrainConfig(BaseModel):
    dataset_meta_path: Path
    train_val_split: float = Field(0.8, ge=0.0, le=1.0)
    ls_template_path: Path
    ccf_annotations_path: Path
    orientation: Optional[SliceOrientation] = None
    registration_downsample_factor: int = 3
    tensorstore_aws_credentials_method: str = "default"
    crop_warp_to_bounding_box: bool = True
    patch_size: Optional[tuple[int, int]] = None
    normalize_orientation_map: Optional[dict[SliceOrientation, list[AcquisitionDirection]]] = {
        SliceOrientation.SAGITTAL: [AcquisitionDirection.SUPERIOR_TO_INFERIOR,
                                    AcquisitionDirection.ANTERIOR_TO_POSTERIOR]
    }
    limit_sagittal_slices_to_hemisphere: bool = False
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(0, ge=0)
    exclude_background_pixels: bool = True

    unet_init_features: int = Field(64, gt=0)
    unet_depth: int = Field(4, gt=0)
    load_checkpoint: Optional[Path] = None

    n_epochs: int = Field(100, gt=0)
    learning_rate: float = Field(0.001, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    decay_learning_rate: bool = True
    warmup_iters: int = Field(1000, ge=0)

    loss_eval_interval: int = Field(500, gt=0)
    eval_iters: int = Field(100, gt=0)
    patience: int = Field(10, gt=0)
    min_delta: float = Field(1e-4, ge=0.0)
    ls_template_to_ccf_affine_path: Path
    ls_template_to_ccf_inverse_warp_path: Path
    ccf_template_path: Path
    region_ccf_ids_map_path: Path

    model_weights_out_dir: Path = Path("./checkpoints")
    log_file: Optional[Path] = None

    device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    mixed_precision: bool = False
    seed: int = 1234
    debug: bool = False
