import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from deep_ccf_registration.metadata import TissueBoundingBoxes


def convert_tissue_bboxes_to_parquet(out_dir: Path, bboxes: TissueBoundingBoxes):
    rows = []
    for subject_id, boxes in tqdm(bboxes.bounding_boxes.items()):
        valid_indices = [i for i, box in enumerate(boxes) if box is not None]
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        for i in range(first_valid, last_valid + 1):
            box = boxes[i]
            if box is not None:
                rows.append({
                    "subject_id": subject_id,
                    "index": i,
                    "y": box.y,
                    "x": box.x,
                    "width": box.width,
                    "height": box.height,
                })
            else:
                # Just use previous bbox if missing, this only happened in 1 case
                ii = i
                print(f'subject_id={subject_id} box {i} is missing, finding previous box')
                while box is None:
                    box = boxes[ii]
                    ii -= 1
                rows.append({
                    "subject_id": subject_id,
                    "index": i,
                    "y": box.y,
                    "x": box.x,
                    "width": box.width,
                    "height": box.height,
                })

    df = pd.DataFrame(rows)
    path = out_dir / "tissue_bounding_boxes.parquet"
    os.makedirs(path, exist_ok=True)
    df.to_parquet(path, index=False, partition_cols=["subject_id"])
    return path

if __name__ == '__main__':
    with open('/Users/adam.amster/smartspim-registration/tissue_bboxes.json') as f:
        tissue_bboxes = json.load(f)
    tissue_bboxes = TissueBoundingBoxes(bounding_boxes=tissue_bboxes)
    convert_tissue_bboxes_to_parquet(
        out_dir=Path('/tmp/'),
        bboxes=tissue_bboxes
    )