from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Tuple
from pathlib import Path

from deep_ccf_registration.metadata import SliceOrientation


class TrainConfig(BaseModel):
    """Complete training configuration"""
    dataset_meta_path: Path
    train_val_split: float = Field(0.8, ge=0.0, le=1.0)
    ls_template_path: Path
    ccf_annotations_path: Path
    orientation: Optional[SliceOrientation] = None
    registration_downsample_factor: int = 3
    tensorstore_aws_credentials_method: str = "default"
    crop_warp_to_bounding_box: bool = True
    patch_size: Optional[Tuple[int, int]] = (256, 256)
    normalize_orientation_map_path: Optional[Path] = None
    limit_sagittal_slices_to_hemisphere: bool = False
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(4, ge=0)
    eval_frac: float = Field(1.0, ge=0.0, le=1.0)

    # Model
    unet_init_features: int = Field(64, gt=0)
    load_checkpoint: Optional[Path] = None

    # Training
    n_epochs: int = Field(100, gt=0)
    learning_rate: float = Field(0.001, gt=0.0)
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    weight_decay: float = Field(0.0, ge=0.0)
    decay_learning_rate: bool = True
    warmup_iters: int = Field(1000, ge=0)

    # Evaluation
    loss_eval_interval: int = Field(500, gt=0)
    eval_iters: int = Field(100, gt=0)
    patience: int = Field(10, gt=0)
    min_delta: float = Field(1e-4, ge=0.0)

    # Output
    model_weights_out_dir: Path = Path("./checkpoints")
    log_file: Optional[Path] = None

    device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    mixed_precision: bool = False
    seed: int = 1234

    @field_validator('patch_size', mode='before')
    @classmethod
    def parse_patch_size(cls, v):
        if v is None or v == 'None':
            return None
        if isinstance(v, str):
            parts = v.split(',')
            return tuple(int(x.strip()) for x in parts)
        return v
