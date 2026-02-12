from deep_ccf_registration.configs.train_config import TrainConfig
from deep_ccf_registration.metadata import TissueBoundingBoxes


def _convert_tissue_bboxes_to_parquet(config: TrainConfig, bboxes: TissueBoundingBoxes):
    rows = []
    for key, boxes in bboxes.bounding_boxes.items():
        # Find the contiguous range [first_valid, last_valid] and fill any
        # gaps with a dummy 1x1 bbox. Only one subject has a gap in practice.
        valid_indices = [i for i, box in enumerate(boxes) if box is not None]
        if not valid_indices:
            continue
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        valid_set = set(valid_indices)
        for i in range(first_valid, last_valid + 1):
            box = boxes[i] if i in valid_set else None
            if box is not None:
                rows.append({
                    "subject_id": key,
                    "index": i,
                    "y": box.y,
                    "x": box.x,
                    "width": box.width,
                    "height": box.height,
                })
            else:
                # Dummy bbox to keep slice range contiguous
                rows.append({
                    "subject_id": key,
                    "index": i,
                    "y": 0,
                    "x": 0,
                    "width": 1,
                    "height": 1,
                })

    df = pd.DataFrame(rows)
    path = config.tmp_path / "tissue_bounding_boxes.parquet"
    os.makedirs(path, exist_ok=True)
    df.to_parquet(path, index=False, partition_cols=["subject_id"])
    return path