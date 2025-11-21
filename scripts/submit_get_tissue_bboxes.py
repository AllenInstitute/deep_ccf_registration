import json
import os
import time
from contextlib import contextmanager
from typing import Optional

import codeocean.error
import requests
from codeocean import CodeOcean
from codeocean.computation import RunParams, NamedRunParam, ComputationState, Computation, \
    ComputationEndStatus
from tqdm import tqdm
from loguru import logger

from deep_ccf_registration.metadata import AcquisitionAxis, SubjectMetadata


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
            #capsule_id='bb83afa1-d0a3-49c7-b888-e0b1beef19ef',
            capsule_id='402472b0-67e7-48cc-b8ce-605c6cf60444',
            named_parameters=[
                NamedRunParam(
                    param_name='subject-id',
                    value=subject['subject_id']
                ),
                NamedRunParam(
                    param_name='num-workers',
                    value='2'
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
                    name=f'{subject["subject_id"]}_tissue_bbox',
                )
                break
            except codeocean.error.Error:
                time.sleep(30)

        jobs.append({'subject_id': subject['subject_id'], "computation_id": run_response.id})

        job_count += 1

        if job_count == job_limit or job_count == len(subjects):
            with open('/Users/adam.amster/smartspim-registration/tissue_bboxes_jobs.json', 'w') as f:
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

    with open('/Users/adam.amster/smartspim-registration/tissue_bboxes_jobs.json') as f:
        jobs = json.load(f)

    computation_responses: dict[str, Computation] = {}
    for job in jobs:
        computation_response = co_client.computations.get_computation(
            computation_id=job['computation_id']
        )
        computation_responses[job['subject_id']] = computation_response

    return computation_responses

def get_output(jobs: list[Computation]):
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)

    outputs = {}
    errors = []
    good = []
    for job in tqdm(jobs):
        subject_id = job.name.split('_')[0]
        if subject_id in good:
            continue
        try:
            download_url = co_client.computations.get_result_file_download_url(
                computation_id=job.id,
                path=f'{subject_id}_tissue_bboxes.json'
            )
            response = requests.get(download_url.url)
            response.raise_for_status()  # Check for errors


            output = response.json()
            outputs[subject_id] = output
            good.append(subject_id)
        except codeocean.error.Error:
            errors.append(subject_id)
            continue

    with open('/Users/adam.amster/smartspim-registration/tissue_bboxes.json', 'w') as f:
        f.write(json.dumps(outputs, indent=2))

    print('errors')
    print(set(errors).difference(good))

def _parse_orientation(axes: list[AcquisitionAxis]):
    axes = sorted(axes, key=lambda x: x.dimension)
    return ''.join([x.direction.value[0] for x in axes])

def get_failed_jobs():
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)
    computations = co_client.capsules.list_computations(capsule_id='bb83afa1-d0a3-49c7-b888-e0b1beef19ef')
    computations = [x for x in computations if x.name.endswith('tissue_bbox')]
    succeeded_computations = [c for c in computations if c.end_status == ComputationEndStatus.Succeeded and c.exit_code == 0 and c.state == ComputationState.Completed]
    subject_computations = [x.name.split('_')[0] for x in succeeded_computations]

    with open('/Users/adam.amster/smartspim-registration/subject_metadata.json') as f:
        subjects = json.load(f)

    subjects_need_run = list(set([x['subject_id'] for x in subjects]).difference(subject_computations))
    print(f'{len(subjects_need_run)} need rerun')
    return subjects_need_run


def get_jobs():
    co_client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org",
                          token=co_token)
    computations = co_client.capsules.list_computations(capsule_id='bb83afa1-d0a3-49c7-b888-e0b1beef19ef')
    computations += co_client.capsules.list_computations(capsule_id='402472b0-67e7-48cc-b8ce-605c6cf60444')
    computations = [x for x in computations if x.name.endswith('tissue_bbox')]

    with open('/Users/adam.amster/smartspim-registration/subject_metadata.json') as f:
        subjects = json.load(f)

    subject_ids = [x['subject_id'] for x in subjects]
    computations = [x for x in computations if x.name.split('_')[0] in subject_ids]
    return computations

if __name__ == '__main__':
    co_token = os.environ['CODEOCEAN_TOKEN']

    with open('/Users/adam.amster/smartspim-registration/subject_metadata.json') as f:
        subjects = json.load(f)


    # with open('/tmp/input_space_midline_jobs.json') as f:
    #     exclude_subjects = json.load(f)

    # failed_subjects = ['776874', '761405', '730695', '730868', '718453', '796385', '752311', '716868', '790763', '781166', '792096', '765861', '758018', '707541']
    # failed_subject = [x for x in subjects if x['subject_id'] in failed_subjects]
    # with open('/Users/adam.amster/smartspim-registration/input_space_midline_jobs.json') as f:
    #     already_submitted = json.load(f)
    #jobs = submit_jobs(subjects=subjects)

    #failed = ['701971', '746045', '751035', '805162', '734858', '730476', '751023', '725231', '734856', '793594', '731336', '751024']
    #failed = ['725231', '730476', '793594', '805162']
    #submit_jobs(subjects=[x for x in subjects if x['subject_id'] in failed])

    jobs = get_jobs()
    get_output(jobs=jobs)
    # computations = get_job_statuses()
    # failed_computations = [x for x, c in computations.items() if c.end_status != ComputationEndStatus.Succeeded or c.exit_code != 0 or c.state != ComputationState.Completed]
    # print(failed_computations)

    # with open('/Users/adam.amster/smartspim-registration/tissue_bboxes_jobs.json') as f:
    #     tissue_bboxes_jobs = json.load(f)
    # print(set([x['subject_id'] for x in subjects]).difference([x['subject_id'] for x in tissue_bboxes_jobs]))
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

    # with open('/Users/adam.amster/smartspim-registration/input_space_midline_jobs.json') as f:
    #     jobs = json.load(f)
    # get_output(jobs=jobs)
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
