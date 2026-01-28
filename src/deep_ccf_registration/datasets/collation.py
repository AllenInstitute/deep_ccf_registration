import numpy as np
import torch

from deep_ccf_registration.datasets.iterable_slice_dataset import PatchSample


def collate_patch_samples(samples: list[PatchSample], pad_dim: int = 512) -> dict:
    """
    Collate a list of PatchSample into a batched dictionary.
    Returns a dict with:
        - input_images: (B, 1, H, W) tensor
        - dataset_indices: list of dataset identifiers (strings)
        - slice_indices: (B,) tensor
        - patch_ys: (B,) tensor
        - patch_xs: (B,) tensor
        - orientations: list of str
        - subject_ids: list of str
        - pad_masks: (B, H, W) tensor indicating valid (non-padded) pixels
        - pad_mask_heights: (B,) tensor of valid pixel counts along H
        - pad_mask_widths: (B,) tensor of valid pixel counts along W
    """
    batch_size = len(samples)
    image_dtype = samples[0].data.dtype
    template_dtype = samples[0].template_points.dtype if samples[0].template_points is not None else np.float32

    images = np.zeros((batch_size, 1, pad_dim, pad_dim), dtype=image_dtype)
    template_points = np.zeros((batch_size, 3, pad_dim, pad_dim), dtype=template_dtype)
    pad_masks = np.zeros((batch_size, pad_dim, pad_dim), dtype=np.uint8)
    tissue_masks = np.zeros((batch_size, pad_dim, pad_dim), dtype=np.float32)
    pad_mask_heights = np.zeros(batch_size, dtype=np.int32)
    pad_mask_widths = np.zeros(batch_size, dtype=np.int32)

    for idx, sample in enumerate(samples):
        img = sample.data
        img_h, img_w = img.shape
        template_points_h, template_points_w = sample.template_points.shape[:-1]
        images[idx, 0, :img_h, :img_w] = img

        if sample.template_points is None:
            raise ValueError("PatchSample missing template_points; cannot collate")
        sample_template_points = np.transpose(sample.template_points, (2, 0, 1))
        template_points[idx, :, :sample_template_points.shape[1], :sample_template_points.shape[2]] = sample_template_points

        pad_masks[idx, :template_points_h, :template_points_w] = 1
        pad_mask_heights[idx] = template_points_h
        pad_mask_widths[idx] = template_points_w

        if sample.tissue_mask is not None:
            tissue_masks[idx, :sample_template_points.shape[1], :sample_template_points.shape[2]] = sample.tissue_mask

    result = {
        "input_images": torch.from_numpy(images),
        "target_template_points": torch.from_numpy(template_points),
        "tissue_masks": torch.from_numpy(tissue_masks),
        "pad_masks": torch.from_numpy(pad_masks.astype(bool)),
        "pad_mask_heights": torch.from_numpy(pad_mask_heights),
        "pad_mask_widths": torch.from_numpy(pad_mask_widths),
        "dataset_indices": [s.dataset_idx for s in samples],
        "slice_indices": torch.tensor([s.slice_idx for s in samples]),
        "patch_ys": torch.tensor([s.start_y for s in samples]),
        "patch_xs": torch.tensor([s.start_x for s in samples]),
        "orientations": [s.orientation for s in samples],
        "subject_ids": [s.subject_id for s in samples],
    }

    has_eval_points = any(s.eval_template_points is not None for s in samples)
    if has_eval_points:
        max_eval_h = max(s.eval_template_points.shape[0] for s in samples if s.eval_template_points is not None)
        max_eval_w = max(s.eval_template_points.shape[1] for s in samples if s.eval_template_points is not None)
        eval_template_points = np.zeros((batch_size, 3, max_eval_h, max_eval_w), dtype=template_dtype)
        eval_pad_masks = np.zeros((batch_size, max_eval_h, max_eval_w), dtype=np.uint8)
        eval_tissue_masks = np.zeros((batch_size, max_eval_h, max_eval_w), dtype=np.float32)

        for idx, sample in enumerate(samples):
            if sample.eval_template_points is None:
                continue
            etp = np.transpose(sample.eval_template_points, (2, 0, 1))
            eval_template_points[idx, :, :etp.shape[1], :etp.shape[2]] = etp
            eval_pad_masks[idx, :sample.eval_template_points.shape[0], :sample.eval_template_points.shape[1]] = 1
            if sample.eval_tissue_mask is not None:
                eval_tissue_masks[idx, :sample.eval_tissue_mask.shape[0], :sample.eval_tissue_mask.shape[1]] = sample.eval_tissue_mask

        result["eval_target_template_points"] = torch.from_numpy(eval_template_points)
        result["eval_pad_masks"] = torch.from_numpy(eval_pad_masks.astype(bool))
        result["eval_tissue_masks"] = torch.from_numpy(eval_tissue_masks)

    return result
