import json
import os
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

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
        subjects: list[SubjectMetadata],
        exclude_subjects: Optional[list[str]] = None,
        job_limit: int = 100
):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=co_token)

    errors = []
    jobs = []
    job_count = 0

    if exclude_subjects is None:
        exclude_subjects = []

    subjects = [x for x in subjects if x.subject_id not in exclude_subjects]

    for subject in tqdm(subjects):
        output_path = subject.ls_to_template_inverse_warp_path.replace('/data/aind_open_data/', '')
        output_path = Path(output_path).parent / 'transforms.zarr'

        run_params = RunParams(
            capsule_id='378af6e0-a21d-496b-9184-cb0175d8e0d5',
            named_parameters=[
                NamedRunParam(
                    param_name='subject-id',
                    value=subject.subject_id
                ),
                NamedRunParam(
                    param_name='dataset-metadata-path',
                    value='/data/smartspim_dataset/subject_metadata.json'
                ),
                NamedRunParam(
                    param_name='light-sheet-template-path',
                    value='/data/aind_open_data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz'
                ),
                NamedRunParam(
                    param_name='output-path',
                    value=f's3://marmot-development-802451596237-us-west-2/transforms/{output_path}'
                ),
                NamedRunParam(
                    param_name='warp-precision',
                    value='float16'
                ),
                NamedRunParam(
                    param_name='chunk-size',
                    value='64'
                )
            ]
        )
        try:
            with catch_code_ocean_error():
                run_response = co_client.computations.run_capsule(run_params=run_params)
        except CodeOceanError:
            errors.append(subject)
            continue

        co_client.computations.rename_computation(
            run_response.id,
            name=f'{subject.subject_id}_transform_to_ome_zarr',
        )

        jobs.append({'subject_id': subject.subject_id, "computation_id": run_response.id})

        job_count += 1

        if job_count == job_limit:
            with open('/tmp/transforms_to_ome_zarr_jobs.json', 'w') as f:
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

    with open('/tmp/transforms_to_ome_zarr_jobs.json') as f:
        jobs = json.load(f)

    computation_responses: dict[str, Computation] = {}
    for job in jobs:
        computation_response = co_client.computations.get_computation(
            computation_id=job['computation_id']
        )
        computation_responses[job['subject_id']] = computation_response

    return computation_responses

def get_distance_metrics():
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

    distances = []
    for subject_id, computation_id in tqdm(subject_id_computation_map.items()):
        download_url = co_client.computations.get_result_file_download_url(
            computation_id=computation_id,
            path=f'{subject_id}_roundtrip_distance.json'
        )

        response = requests.get(download_url.url)
        response.raise_for_status()  # Check for errors

        distance = response.json()
        distances.append({'subject_id': subject_id, **distance})

    with open('/tmp/distances.json', 'w') as f:
        f.write(json.dumps(distances, indent=2))

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

    # with open('/tmp/distances.json') as f:
    #     distances = json.load(f)
    #
    # distances = pd.DataFrame(distances)
    # distances['orientation'] = distances['subject_id'].apply(lambda subject_id: _parse_orientation(
    #     axes=[x for x in dataset if x.subject_id == subject_id][0].axes))
    # distances = distances.sort_values('roundtrip_distance')

    with PdfPages('/tmp/smartspim_qc.pdf') as pdf:
        for i, (subject_id, _) in enumerate(subject_id_computation_map.items()):
            subject = [x for x in dataset if x.subject_id == subject_id][0]
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
            ax.set_title(f'subject id {subject.subject_id} orientation {orientation}')

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

def get_already_processed():
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    computations = co_client.capsules.list_computations(
        capsule_id='378af6e0-a21d-496b-9184-cb0175d8e0d5'
    )

    computations = [x for x in computations if 'transform_to_ome_zarr' in  x.name and
                    x.state == ComputationState.Completed and x.exit_code == 0 and x.end_status == ComputationEndStatus.Succeeded]

    # return subject ids
    return [x.name.split('_')[0] for x in computations]

if __name__ == '__main__':
    co_token = os.environ['CODEOCEAN_TOKEN']

    with open('/Users/adam.amster/smartspim-registration/subject_metadata.json') as f:
        dataset_meta = json.load(f)

    already_processed = get_already_processed()
    dataset_meta = [SubjectMetadata.model_validate(x) for x in dataset_meta]
    jobs = submit_jobs(
        subjects=dataset_meta,
        exclude_subjects=already_processed
    )

    #get_job_statuses()

    # jobs = rerun_failed_jobs(subject_ids=['774923'])
    #
    # logger.info('writing jobs meta to /tmp/qc_jobs.json')
    # with open('/tmp/qc_jobs.json', 'w') as f:
    #     f.write(json.dumps(jobs, indent=2))

    #get_distance_metrics()
    #create_pdf()
