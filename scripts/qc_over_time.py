import json
import seaborn as sns
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def main():
    with open('/Users/adam.amster/smartspim-registration/metric.json') as f:
        metric = json.load(f)

    with open('/Users/adam.amster/Downloads/subject_metadata.json') as f:
        subject_metadata = json.load(f)

    for subject in metric:
        subject_meta = [x for x in subject_metadata if x['subject_id'] == subject['subject_id']][0]
        stitched_path = subject_meta['stitched_volume_path']
        stitched_path = Path(stitched_path.replace('s3://aind-open-data/', ''))
        stitched_dir = stitched_path.parent.parent.parent.name
        stitched_date = stitched_dir[stitched_dir.index('stitched_')+len('stitched_'):]
        subject['stitched_date'] = stitched_date

    metric = pd.DataFrame(metric)
    metric['stitched_date'] = pd.to_datetime(metric['stitched_date'], format="%Y-%m-%d_%H-%M-%S")
    sns.scatterplot(metric, x='stitched_date', y='dice_metric')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()