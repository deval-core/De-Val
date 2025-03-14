import argparse
import os
import pickle
import subprocess

import bittensor as bt
import torch

from deval.api.miner_docker_client import MinerDockerClient
from deval.contest import DeValContest
from deval.model.model_state import ModelState
from deval.rewards.pipeline import RewardPipeline
from deval.compute_horde_settings import (
    COMPUTE_HORDE_ARTIFACT_MODEL_STATE_PATH,
    COMPUTE_HORDE_ARTIFACT_MODEL_COLDKEY_PATH,
    COMPUTE_HORDE_ARTIFACT_MODEL_HASH_PATH,
    COMPUTE_HORDE_VOLUME_MINER_STATE_PATH,
    COMPUTE_HORDE_VOLUME_MODEL_PATH,
    COMPUTE_HORDE_VOLUME_TASK_REPO_PATH,
)
from deval.task_repository import TaskRepository
from deval.utils.logging import WandBLogger
from deval.validator import Validator


def get_args():
    parser = argparse.ArgumentParser(description="Parse required flags.")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device (string, optional)",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=False,
        help="Timeout (int, optional)",
        default=60 * 10,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        required=False,
        help="Random seed (int, optional)",
        default=0,
    )
    return parser.parse_args()


def validate_model(
    miner_state: ModelState, model_hash: str, model_coldkey: str
) -> bool:
    if miner_state.coldkey != model_coldkey:
        print(
            "Mismatch between the Miner's coldkey and the Model's Coldkey. INVALID Model"
        )
        return False

    if miner_state.chain_model_hash != model_hash:
        print(
            "Mismatch between the model hash on the chain commit and the model hash on huggingface"
        )
        return False

    return True


def main():
    bt.logging.set_console()

    args = get_args()

    bt.logging.info("Loading task repository")
    with open(COMPUTE_HORDE_VOLUME_TASK_REPO_PATH, "rb") as f:
        task_repo: TaskRepository = pickle.load(f)

    bt.logging.info("Loading miner state")
    with open(COMPUTE_HORDE_VOLUME_MINER_STATE_PATH, "rb") as f:
        miner_state: ModelState = pickle.load(f)

    bt.logging.info("Starting miner API")
    with subprocess.Popen(
        ["poetry", "run", "uvicorn", "deval.api.miner_api:app", "--port", "8000"],
        env={
            **os.environ.copy(),
            "MODEL_VOLUME_DIR": COMPUTE_HORDE_VOLUME_MODEL_PATH,
            "PYTHONUNBUFFERED": "1",
            "RANDOM_SEED": str(args.random_seed),
        },
    ) as uvicorn_process:
        active_tasks = [task_name for task_name, _ in task_repo.get_all_tasks()]

        reward_pipeline = RewardPipeline(
            selected_tasks=active_tasks, device=args.device
        )

        contest = DeValContest(
            reward_pipeline=reward_pipeline,
            forward_start_time=0,  # Ignoring this, not needed here.
            timeout=args.timeout,
        )

        miner_docker_client = MinerDockerClient(api_url="http://localhost:8000")

        miner_docker_client._poll_service_for_readiness(500)

        model_hash = miner_docker_client.get_model_hash()
        model_coldkey = miner_docker_client.get_model_coldkey()

        with open(COMPUTE_HORDE_ARTIFACT_MODEL_HASH_PATH, "wb") as f:
            f.write(model_hash.encode("utf-8"))

        with open(COMPUTE_HORDE_ARTIFACT_MODEL_COLDKEY_PATH, "wb") as f:
            f.write(model_coldkey.encode("utf-8"))

        is_valid = validate_model(miner_state, model_hash, model_coldkey)
        if is_valid:
            wandb_logger = WandBLogger(None, None, active_tasks, None, force_off=True)

            for task_name, tasks in task_repo.get_all_tasks():
                bt.logging.debug(f"Running task {task_name}")
                miner_state = Validator.run_step(
                    task_name,
                    tasks,
                    miner_docker_client,
                    miner_state,
                    contest,
                    wandb_logger,
                )

            bt.logging.debug(miner_state.rewards)
            bt.logging.info("Completed epoch")
        else:
            bt.logging.info("Invalid model, epoch not ran")

        bt.logging.info(f"Rewards for uid {miner_state.uid} are: {miner_state.rewards}")

        with open(COMPUTE_HORDE_ARTIFACT_MODEL_STATE_PATH, "wb") as f:
            pickle.dump(miner_state, f)

        bt.logging.info("Terminating miner API")
        uvicorn_process.terminate()


if __name__ == "__main__":
    main()
