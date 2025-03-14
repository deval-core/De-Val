import asyncio
import random
from dataclasses import dataclass

import math
import pickle
import time

import bittensor as bt
from compute_horde_sdk.v1 import (
    ComputeHordeClient as SDKComputeHordeClient,
    InlineInputVolume,
    HuggingfaceInputVolume,
    ExecutorClass,
    ComputeHordeJob,
    ComputeHordeJobStatus,
    ComputeHordeJobSpec,
)

from deval.model.model_state import ModelState
from deval.compute_horde_settings import (
    COMPUTE_HORDE_ARTIFACT_MODEL_COLDKEY_PATH,
    COMPUTE_HORDE_ARTIFACT_MODEL_HASH_PATH,
    COMPUTE_HORDE_ARTIFACT_MODEL_STATE_PATH,
    COMPUTE_HORDE_FACILITATOR_URL,
    COMPUTE_HORDE_JOB_MAX_ATTEMPTS,
    COMPUTE_HORDE_JOB_NAMESPACE,
    COMPUTE_HORDE_VALIDATOR_HOTKEY,
    COMPUTE_HORDE_WALLET,
    COMPUTE_HORDE_HOTKEY,
    COMPUTE_HORDE_EXECUTOR_CLASS,
    COMPUTE_HORDE_JOB_DOCKER_IMAGE,
    COMPUTE_HORDE_JOB_TIMEOUT,
    COMPUTE_HORDE_ARTIFACTS_DIR,
    COMPUTE_HORDE_VOLUME_MODEL_PATH,
    COMPUTE_HORDE_VOLUME_TASK_REPO_DIR,
    COMPUTE_HORDE_VOLUME_MINER_STATE_DIR,
    COMPUTE_HORDE_VOLUME_MINER_STATE_FILENAME,
    COMPUTE_HORDE_VOLUME_TASK_REPO_FILENAME,
)
from deval.task_repository import TaskRepository
from deval.tasks.task import TasksEnum

_REQUIRED_SETTINGS = (
    "COMPUTE_HORDE_JOB_NAMESPACE",
    "COMPUTE_HORDE_JOB_DOCKER_IMAGE",
    "COMPUTE_HORDE_VALIDATOR_HOTKEY",
)

CROSS_VALIDATION_CHANCE = 0.1
"""Currently, cross validation is random, this specifies the fraction of jobs to be cross-validated."""

TASKS_REWARD_ABS_TOL = {
    TasksEnum.RELEVANCY.value: 0.05,
    TasksEnum.HALLUCINATION.value: 0.05,
    TasksEnum.COMPLETENESS.value: 0.05,
    TasksEnum.ATTRIBUTION.value: 0.05,
}
"""Absolute tolerance for average rewards comparison."""
# Currently, require (average) rewards to be within 5% of each other for all tasks.
# Should be adjusted if needed:
# - too low: might report false positives
# - too high: might not catch cheating miners


# Note: there are classes ComputeHordeClient and ComputeHordeJobResult in the SDK, these are not the same.


@dataclass
class ComputeHordeJobResult:
    model_state: ModelState
    model_hash: str
    model_coldkey: str


class ComputeHordeClient:
    def __init__(self, keypair: bt.Keypair | None = None):
        missing_settings = [
            setting for setting in _REQUIRED_SETTINGS if not globals().get(setting)
        ]
        if missing_settings:
            raise ValueError(
                f"Required settings: {', '.join(missing_settings)} are not set. "
                "Please set them in your .env file."
            )

        if COMPUTE_HORDE_WALLET and COMPUTE_HORDE_HOTKEY:
            bt.logging.info(
                f"Using hotkey {COMPUTE_HORDE_HOTKEY} from wallet {COMPUTE_HORDE_WALLET} to sign ComputeHorde jobs."
            )
            keypair = bt.wallet(
                name=COMPUTE_HORDE_WALLET, hotkey=COMPUTE_HORDE_HOTKEY
            ).hotkey

        if keypair is None:
            raise ValueError("No keypair provided.")

        self.keypair = keypair
        self.executor_class = (
            ExecutorClass(COMPUTE_HORDE_EXECUTOR_CLASS)
            if COMPUTE_HORDE_EXECUTOR_CLASS
            else ExecutorClass.always_on__llm__a6000
        )
        self.client = SDKComputeHordeClient(
            hotkey=self.keypair,
            compute_horde_validator_hotkey=COMPUTE_HORDE_VALIDATOR_HOTKEY,
            **(
                {"facilitator_url": COMPUTE_HORDE_FACILITATOR_URL}
                if COMPUTE_HORDE_FACILITATOR_URL
                else {}
            ),
        )

    async def run_epoch_on_compute_horde(
        self,
        miner_state: ModelState,
        task_repo: TaskRepository,
    ) -> ComputeHordeJobResult:
        task_repo_pkl = pickle.dumps(task_repo)
        miner_state_pkl = pickle.dumps(miner_state)

        job_spec = ComputeHordeJobSpec(
            executor_class=self.executor_class,
            job_namespace=COMPUTE_HORDE_JOB_NAMESPACE,
            docker_image=COMPUTE_HORDE_JOB_DOCKER_IMAGE,
            args=[
                "python",
                "neurons/compute_horde_entrypoint.py",
                "--random-seed",
                str(0),
            ],
            artifacts_dir=COMPUTE_HORDE_ARTIFACTS_DIR,
            # TODO: Pin huggingface volume revision.
            input_volumes={
                COMPUTE_HORDE_VOLUME_MINER_STATE_DIR: InlineInputVolume.from_file_contents(
                    COMPUTE_HORDE_VOLUME_MINER_STATE_FILENAME, miner_state_pkl
                ),
                COMPUTE_HORDE_VOLUME_TASK_REPO_DIR: InlineInputVolume.from_file_contents(
                    COMPUTE_HORDE_VOLUME_TASK_REPO_FILENAME,
                    task_repo_pkl,
                    compress=True,
                ),
                COMPUTE_HORDE_VOLUME_MODEL_PATH: HuggingfaceInputVolume(
                    repo_id=miner_state.get_model_url()
                ),
            },
        )

        bt.logging.info("Running organic job on Compute Horde.")

        job, result = await self.run_job(job_spec)

        if self.should_cross_validate():
            # Only useful if reusing the same miner - it will reject the second job if started immediately.
            # TODO: Remove this.
            await asyncio.sleep(15)
            await self.cross_validate(job, result, job_spec)

        return result

    def should_cross_validate(self) -> bool:
        return random.random() < CROSS_VALIDATION_CHANCE

    async def cross_validate(
        self,
        job: ComputeHordeJob,
        result: ComputeHordeJobResult,
        job_spec: ComputeHordeJobSpec,
    ):
        bt.logging.info("Running cross-validation job on Compute Horde.")
        try:
            trusted_job, trusted_result = await self.run_job(
                job_spec, on_trusted_miner=True
            )
        except Exception as e:
            bt.logging.error(f"Failed to run cross-validation job: {e}")
            return

        if not self.results_similar(result, trusted_result):
            bt.logging.warning(
                f"Results are not similar, reporting cheated job {job.uuid}."
            )
            try:
                await self.client.report_cheated_job(job.uuid)
            except Exception as e:
                bt.logging.error(f"Failed to report cheated job {job.uuid}: {e}")

    async def run_job(
        self,
        job_spec: ComputeHordeJobSpec,
        on_trusted_miner: bool = False,
    ) -> tuple[ComputeHordeJob, ComputeHordeJobResult]:
        start = time.time()

        job = await self.client.run_until_complete(
            job_spec,
            on_trusted_miner=on_trusted_miner,
            job_attempt_callback=lambda job_att: bt.logging.info(
                f"Job attempt {job_att.uuid}"
            ),
            max_attempts=COMPUTE_HORDE_JOB_MAX_ATTEMPTS,
            timeout=COMPUTE_HORDE_JOB_TIMEOUT,
        )

        time_took = time.time() - start

        if job.result is None:
            raise RuntimeError(
                f"Job {job.uuid} result is None. Check logs for more details."
            )

        bt.logging.info(job.result.stdout, f"Job {job.uuid} stdout:")

        if job.status != ComputeHordeJobStatus.COMPLETED:
            raise RuntimeError(
                f"Job {job.uuid} status is {job.status}. Check logs for more details."
            )

        bt.logging.success(f"Job {job.uuid} finished in {time_took} seconds")

        model_state_pkl = job.result.artifacts[COMPUTE_HORDE_ARTIFACT_MODEL_STATE_PATH]
        model_hash = job.result.artifacts[COMPUTE_HORDE_ARTIFACT_MODEL_HASH_PATH]
        model_coldkey = job.result.artifacts[COMPUTE_HORDE_ARTIFACT_MODEL_COLDKEY_PATH]

        result = ComputeHordeJobResult(
            model_state=pickle.loads(model_state_pkl),
            model_hash=model_hash.decode("utf-8"),
            model_coldkey=model_coldkey.decode("utf-8"),
        )

        return job, result

    def results_similar(
        self, result: ComputeHordeJobResult, trusted_result: ComputeHordeJobResult
    ) -> bool:
        # Compare model hash
        if result.model_hash != trusted_result.model_hash:
            bt.logging.warning(
                f"{result.model_hash=} does not match {trusted_result.model_hash=}"
            )
            return False

        # Compare model coldkey
        if result.model_coldkey != trusted_result.model_coldkey:
            bt.logging.warning(
                f"{result.model_coldkey=} does not match {trusted_result.model_coldkey=}"
            )
            return False

        # Compare model rewards
        for task_name, trusted_rewards in trusted_result.model_state.rewards.items():
            rewards = result.model_state.rewards.get(task_name, [])
            if len(rewards) != len(trusted_rewards):
                bt.logging.warning(
                    f"Number of rewards for task {task_name} is different, miner: {len(rewards)} rewards, trusted_miner: {len(trusted_rewards)} rewards."
                )
                return False

            if len(trusted_rewards) < 3:
                # Not enough samples
                bt.logging.debug(
                    f"Only {len(trusted_rewards)} samples for {task_name}, skipping comparison."
                )
                continue

            avg_reward = sum(rewards) / len(rewards)
            avg_trusted_reward = sum(trusted_rewards) / len(trusted_rewards)

            abs_tol = TASKS_REWARD_ABS_TOL.get(task_name)
            if abs_tol is not None and not math.isclose(
                avg_reward, avg_trusted_reward, abs_tol=abs_tol
            ):
                bt.logging.warning(
                    f"Rewards for task {task_name} are too different, "
                    f"miner: avg({rewards})={avg_reward}, trusted_miner: avg({trusted_rewards})={avg_trusted_reward}, "
                    f"for {abs_tol=}."
                )
                return False

        return True
