import json
import os
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import codeocean.error
import numpy as np
import pandas as pd
import requests
from PIL import Image
from codeocean import CodeOcean
from codeocean.computation import RunParams, NamedRunParam, ComputationState, Computation, \
    ComputationEndStatus
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from loguru import logger

from deep_ccf_registration.datasets.slice_dataset import AcquisitionAxis, SubjectMetadata


class CodeOceanError(Exception):
    pass

@contextmanager
def catch_code_ocean_error():
    """Context manager to catch Code Ocean errors, since the API doesn't surface them
    and instead returns just the HTTP status code."""
    try:
        yield
    except requests.HTTPError as err:
        try:
            message = err.response.json()["message"]
        except requests.JSONDecodeError:
            message = err.response.text
        raise CodeOceanError(message)

def submit_jobs(
    subjects: list[dict],
    exclude_subjects: Optional[list[dict]] = None,
    job_limit: int = 100,
    jobs: Optional[list] = None
):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=co_token)

    errors = []
    jobs = [] if jobs is None else jobs
    job_count = 0

    if exclude_subjects is None:
        exclude_subjects = []

    subjects = [x for x in subjects if x['subject_id'] not in exclude_subjects]

    for subject in tqdm(subjects):
        run_params = RunParams(
            capsule_id='7945f892-523c-4c94-b0f6-b91cfd3d660a',
            named_parameters=[
                NamedRunParam(
                    param_name='subject-id',
                    value=subject['subject_id']
                )
            ]
        )
        try:
            with catch_code_ocean_error():
                run_response = co_client.computations.run_capsule(run_params=run_params)
        except CodeOceanError:
            errors.append(subject)
            continue

        while True:
            try:
                co_client.computations.rename_computation(
                    run_response.id,
                    name=f'{subject["subject_id"]}_qc',
                )
                break
            except codeocean.error.Error:
                time.sleep(30)

        jobs.append({'subject_id': subject['subject_id'], "computation_id": run_response.id})

        job_count += 1

        if job_count == job_limit or job_count == len(subjects):
            with open('/tmp/input_space_midline_jobs.json', 'w') as f:
                f.write(json.dumps(jobs, indent=2))

            logger.info('Job count reached. Waiting until all running jobs finish')
            while True:
                job_statuses = get_job_statuses()
                running_count = sum([x.state == ComputationState.Running for _, x in job_statuses.items()])
                if running_count != 0:
                    logger.info(f'Running count: {running_count}')
                    time.sleep(30)
                else:
                    job_count = 0
                    break
    print('errors')
    print(errors)

    return jobs

def get_job_statuses() -> dict[str, Computation]:
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    with open('/Users/adam.amster/smartspim-registration/input_space_midline_jobs.json') as f:
        jobs = json.load(f)

    computation_responses: dict[str, Computation] = {}
    for job in jobs:
        computation_response = co_client.computations.get_computation(
            computation_id=job['computation_id']
        )
        computation_responses[job['subject_id']] = computation_response

    return computation_responses

def get_output(jobs):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    computations = []
    for job in tqdm(jobs, desc='getting computations'):
        computation_response = co_client.computations.get_computation(
            computation_id=job['computation_id']
        )
        computations.append(computation_response)

    subject_id_computation_map = {x.name.split('_')[0]: x.id for x in computations}

    outputs = []
    errors = []
    for subject_id, computation_id in tqdm(subject_id_computation_map.items()):
        try:
            download_url = co_client.computations.get_result_file_download_url(
                computation_id=computation_id,
                path='midline.json'
            )
            response = requests.get(download_url.url)
            response.raise_for_status()  # Check for errors


            output = response.json()
            outputs.append({'subject_id': subject_id, **output})
        except codeocean.error.Error:
            errors.append(subject_id)
            continue

    with open('/tmp/midline.json', 'w') as f:
        f.write(json.dumps(outputs, indent=2))

    print('errors')
    print(errors)

def _parse_orientation(axes: list[AcquisitionAxis]):
    axes = sorted(axes, key=lambda x: x.dimension)
    return ''.join([x.direction.value[0] for x in axes])

def create_pdf(qc_jobs, out_path: Path):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    computations = []
    for qc_job in qc_jobs:
        computation_response = co_client.computations.get_computation(
            computation_id=qc_job['computation_id']
        )
        computations.append(computation_response)

    subject_id_computation_map = {x.name.split('_')[0]: x.id for x in computations}

    images = {}
    for subject_id, computation_id in tqdm(subject_id_computation_map.items()):
        download_url = co_client.computations.get_result_file_download_url(
            computation_id=computation_id,
            path=f'{subject_id}.png'
        )
        response = requests.get(download_url.url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        images[subject_id] = img

    with open('/Users/adam.amster/Downloads/subject_metadata.json') as f:
        dataset = json.load(f)

    dataset = [SubjectMetadata.model_validate(x) for x in dataset]

    with open('/tmp/metric.json') as f:
        metrics = json.load(f)

    metrics = [x for x in metrics if x['subject_id'] in images.keys()]
    metrics = pd.DataFrame(metrics)
    metrics['orientation'] = metrics['subject_id'].apply(lambda subject_id: _parse_orientation(
        axes=[x for x in dataset if x.subject_id == subject_id][0].axes))
    metrics = metrics.sort_values('dice_metric', ascending=False)

    with PdfPages(out_path) as pdf:
        for i, row in enumerate(metrics.itertuples()):
            subject = [x for x in dataset if x.subject_id == row.subject_id][0]
            img = images[subject.subject_id]
            print(f"Adding image {i + 1}/{len(images)} to PDF")

            # Create figure with size matching image aspect ratio
            img_width, img_height = img.size

            if img_width > img_height:
                figsize = (30, 15)
            else:
                figsize = (15, 30)

            fig, ax = plt.subplots(figsize=figsize)
            orientation = _parse_orientation(axes=subject.axes)
            ax.set_title(f'subject id {subject.subject_id} orientation {orientation} dice {row.dice_metric:.3f}')

            # Display image
            ax.imshow(img)

            # Save current figure to PDF
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

def rerun_failed_jobs(subject_ids):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    with open('/Users/adam.amster/Downloads/qc_subjects.json') as f:
        qc_subjects = json.load(f)

    qc_subjects = [x for x in qc_subjects if x["subject_id"] in subject_ids]

    rerun_jobs = submit_jobs(subjects=qc_subjects)

    with open('/tmp/qc_jobs.json') as f:
        jobs = json.load(f)

    jobs = [x for x in jobs if x['subject_id'] not in subject_ids]

    jobs += rerun_jobs

    return jobs

if __name__ == '__main__':
    co_token = os.environ['CODEOCEAN_TOKEN']

    with open('/Users/adam.amster/Downloads/subject_metadata.json') as f:
        subjects = json.load(f)


    # with open('/tmp/input_space_midline_jobs.json') as f:
    #     exclude_subjects = json.load(f)

    # failed_subjects = ['776874', '761405', '730695', '730868', '718453', '796385', '752311', '716868', '790763', '781166', '792096', '765861', '758018', '707541']
    # failed_subject = [x for x in subjects if x['subject_id'] in failed_subjects]
    # with open('/Users/adam.amster/smartspim-registration/input_space_midline_jobs.json') as f:
    #     already_submitted = json.load(f)
    # jobs = submit_jobs(subjects=failed_subject, jobs=already_submitted)

    # computations = get_job_statuses()
    # failed_computations = [x for x, c in computations.items() if c.end_status != ComputationEndStatus.Succeeded or c.exit_code != 0 or c.state != ComputationState.Completed]
    # print(failed_computations)
    # jobs = rerun_failed_jobs(subject_ids=['774923'])
    #
    # logger.info('writing jobs meta to /tmp/qc_jobs.json')
    # with open('/tmp/qc_jobs.json', 'w') as f:
    #     f.write(json.dumps(jobs, indent=2))

    # with open('/tmp/qc_jobs.json') as f:
    #     qc_jobs = json.load(f)
    #
    # with open('/tmp/metric.json') as f:
    #     metrics = json.load(f)
    #
    # rng = np.random.default_rng(1234)
    # subjects = [x['subject_id'] for x in metrics if x['dice_metric'] < 0.9]
    # idxs = np.arange(len(subjects))
    # np.random.shuffle(idxs)
    # idxs = idxs[:100]
    #
    # subjects = [subjects[i] for i in idxs]
    # qc_jobs = [x for x in qc_jobs if x['subject_id'] in subjects]

    with open('/Users/adam.amster/smartspim-registration/input_space_midline_jobs.json') as f:
        jobs = json.load(f)
    get_output(jobs=jobs)
    #create_pdf(qc_jobs=qc_jobs, out_path=Path('/tmp/smartspim_qc_bad.pdf'))

    # metrics = pd.DataFrame(metrics)
    # metrics['dice_metric'].plot.hist()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # sorted_dice = np.sort(metrics['dice_metric'])
    # cdf = np.arange(1, len(sorted_dice) + 1) / len(sorted_dice)
    # ax.plot(sorted_dice, cdf)
    # ax.set_xlabel('Dice Score')
    # ax.set_ylabel('Cumulative Probability')
    # ax.grid(True)
    #
    # plt.show()
