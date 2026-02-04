import numpy as np
import torch

from deep_ccf_registration.datasets.iterable_slice_dataset import PatchSample


def collate_patch_samples(samples: list[PatchSample]) -> dict:
    """
    Collate a list of PatchSample into a batched dictionary.
    Assumes all samples have been padded/cropped to the same size by transforms.
    Returns a dict with:
        - input_images: (B, 1, H, W) tensor
        - target_template_points: (B, 3, H, W) tensor
        - tissue_masks: (B, H, W) tensor (if present)
        - pad_masks: (B, H, W) tensor indicating valid (non-padded) pixels
        - dataset_indices: list of dataset identifiers (strings)
        - slice_indices: (B,) tensor
        - patch_ys: (B,) tensor
        - patch_xs: (B,) tensor
        - orientations: list of str
        - subject_ids: list of str
    """
    batch_size = len(samples)

    images = []
    for s in samples:
        img = s.data
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # (1, H, W)
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        images.append(img)
    images = np.stack(images, axis=0)  # (B, C, H, W)

    template_points = np.stack(
        [np.transpose(s.template_points, (2, 0, 1)) for s in samples],
        axis=0
    )  # (B, 3, H, W)

    # Get output dimensions
    _, _, out_h, out_w = images.shape

    # Reconstruct pad_masks from padding info
    pad_masks = np.zeros((batch_size, out_h, out_w), dtype=np.float32)
    for idx, sample in enumerate(samples):
        valid_h = out_h - sample.pad_top - sample.pad_bottom
        valid_w = out_w - sample.pad_left - sample.pad_right
        pad_masks[idx, sample.pad_top:sample.pad_top + valid_h,
                  sample.pad_left:sample.pad_left + valid_w] = True

    tissue_masks = None
    if samples[0].tissue_mask is not None:
        tissue_masks = np.stack([s.tissue_mask.astype(np.float32) for s in samples], axis=0)

    result = {
        "input_images": torch.from_numpy(images),
        "target_template_points": torch.from_numpy(template_points),
        "pad_masks": torch.from_numpy(pad_masks),
        "dataset_indices": [s.dataset_idx for s in samples],
        "slice_indices": torch.tensor([s.slice_idx for s in samples]),
        "patch_ys": torch.tensor([s.start_y for s in samples]),
        "patch_xs": torch.tensor([s.start_x for s in samples]),
        "orientations": [s.orientation for s in samples],
        "subject_ids": [s.subject_id for s in samples],
    }

    if tissue_masks is not None:
        result["tissue_masks"] = torch.from_numpy(tissue_masks)

    # Handle eval template points (at original resolution, no interpolation)
    # These may have different sizes per sample, so we pad to max size
    has_eval_points = any(s.eval_template_points is not None for s in samples)
    if has_eval_points:
        eval_samples = [s for s in samples if s.eval_template_points is not None]
        if eval_samples:
            max_eval_h = max(s.eval_template_points.shape[0] for s in eval_samples)
            max_eval_w = max(s.eval_template_points.shape[1] for s in eval_samples)
            template_dtype = samples[0].template_points.dtype

            eval_template_points = np.zeros((batch_size, 3, max_eval_h, max_eval_w), dtype=template_dtype)
            eval_pad_masks = np.zeros((batch_size, max_eval_h, max_eval_w), dtype=np.uint8)

            for idx, sample in enumerate(samples):
                if sample.eval_template_points is None:
                    continue
                etp = np.transpose(sample.eval_template_points, (2, 0, 1))
                eval_template_points[idx, :, :etp.shape[1], :etp.shape[2]] = etp
                eval_pad_masks[idx, :sample.eval_template_points.shape[0], :sample.eval_template_points.shape[1]] = 1

            result["eval_template_points"] = torch.from_numpy(eval_template_points)
            result["eval_pad_masks"] = torch.from_numpy(eval_pad_masks.astype(bool))
            result["eval_shapes"] = [s.eval_shape for s in samples]

            if samples[0].eval_tissue_mask is not None:
                eval_tissue_masks = np.zeros((batch_size, max_eval_h, max_eval_w), dtype=np.float32)
                for idx, sample in enumerate(samples):
                    if sample.eval_tissue_mask is not None:
                        eval_tissue_masks[idx, :sample.eval_tissue_mask.shape[0], :sample.eval_tissue_mask.shape[1]] = sample.eval_tissue_mask
                result["eval_tissue_masks"] = torch.from_numpy(eval_tissue_masks)

    return result
