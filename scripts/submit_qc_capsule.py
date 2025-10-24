import json
import os
import time
from contextlib import contextmanager
from io import BytesIO

import codeocean.error
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

def submit_jobs(qc_subjects: list[dict]):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=co_token)

    errors = []
    jobs = []
    for subject in tqdm(qc_subjects):
        run_params = RunParams(
            capsule_id='1c2ed940-5f63-450f-83e9-5500308c2bf6',
            named_parameters=[
                NamedRunParam(
                    param_name='subject-id',
                    value=subject['subject_id']
                ),
                NamedRunParam(
                    param_name='dataset-meta-path',
                    value='/data/smartspim_dataset/subject_metadata.json'
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
    print('errors')
    print(errors)

    return jobs

def get_job_statuses():
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    with open('/tmp/qc_jobs.json') as f:
        qc_jobs = json.load(f)

    computation_responses: dict[str, Computation] = {}
    for qc_job in qc_jobs:
        computation_response = co_client.computations.get_computation(
            computation_id=qc_job['computation_id']
        )
        computation_responses[qc_job['subject_id']] = computation_response

    running_jobs = [x for _, x in computation_responses.items() if x.state == ComputationState.Running]
    if running_jobs:
        logger.warning(f'{len(running_jobs)} still running')
        return

    failed_jobs: dict[str, str] = {}
    for subject_id, computation_response in computation_responses.items():
        if computation_response.state == ComputationState.Failed or computation_response.exit_code == 1 or computation_response.end_status != ComputationEndStatus.Succeeded:
            failed_jobs[subject_id] = computation_response.id
    if failed_jobs:
        for failed_subject_id, failed_job_id in failed_jobs.items():
            logger.error(f'{failed_subject_id} failed')
        raise RuntimeError(f'{len(failed_jobs)} jobs failed')

    assert all([x.state == ComputationState.Completed and x.exit_code == 0 for _, x in computation_responses.items()])

def get_metrics():
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    with open('/tmp/qc_jobs.json') as f:
        qc_jobs = json.load(f)

    computations = []
    for qc_job in qc_jobs:
        computation_response = co_client.computations.get_computation(
            computation_id=qc_job['computation_id']
        )
        computations.append(computation_response)

    subject_id_computation_map = {x.name.split('_')[0]: x.id for x in computations}

    metrics = []
    for subject_id, computation_id in tqdm(subject_id_computation_map.items()):
        download_url = co_client.computations.get_result_file_download_url(
            computation_id=computation_id,
            path='dice_metric.json'
        )

        response = requests.get(download_url.url)
        response.raise_for_status()  # Check for errors

        metric = response.json()
        metrics.append({'subject_id': subject_id, **metric})

    with open('/tmp/metric.json', 'w') as f:
        f.write(json.dumps(metrics, indent=2))

def _parse_orientation(axes: list[AcquisitionAxis]):
    axes = sorted(axes, key=lambda x: x.dimension)
    return ''.join([x.direction.value[0] for x in axes])

def create_pdf():
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    with open('/tmp/qc_jobs.json') as f:
        qc_jobs = json.load(f)

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

    metrics = pd.DataFrame(metrics)
    metrics['orientation'] = metrics['subject_id'].apply(lambda subject_id: _parse_orientation(
        axes=[x for x in dataset if x.subject_id == subject_id][0].axes))
    metrics = metrics.sort_values('dice_metric', ascending=False)

    with PdfPages('/tmp/smartspim_qc.pdf') as pdf:
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

    rerun_jobs = submit_jobs(qc_subjects=qc_subjects)

    with open('/tmp/qc_jobs.json') as f:
        jobs = json.load(f)

    jobs = [x for x in jobs if x['subject_id'] not in subject_ids]

    jobs += rerun_jobs

    return jobs

if __name__ == '__main__':
    co_token = os.environ['CODEOCEAN_TOKEN']

    with open('/Users/adam.amster/Downloads/qc_subjects (1).json') as f:
        qc_subjects = json.load(f)
    #jobs = submit_jobs(qc_subjects=qc_subjects)

    #get_job_statuses()

    # jobs = rerun_failed_jobs(subject_ids=['774923'])
    #
    # logger.info('writing jobs meta to /tmp/qc_jobs.json')
    # with open('/tmp/qc_jobs.json', 'w') as f:
    #     f.write(json.dumps(jobs, indent=2))

    get_metrics()
    create_pdf()
