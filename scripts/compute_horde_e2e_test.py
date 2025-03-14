import argparse
import asyncio
import os
import pickle
import sys
import time
from uuid import uuid4

import bittensor
from dotenv import load_dotenv, find_dotenv

# initialize
load_dotenv(find_dotenv())

from deval.api.miner_docker_client import MinerDockerClient
from deval.contest import DeValContest
from deval.compute_horde_client import ComputeHordeClient
from deval.model.chain_metadata import ChainModelMetadataStore
from deval.model.model_state import ModelState
from deval.task_repository import TaskRepository
from deval.tasks.task import TasksEnum
from deval.utils.logging import WandBLogger


allowed_models = ["gpt-4o", "gpt-4o-mini", "mistral-7b", "claude-3.5", "command-r-plus"]


def get_args():
    parser = argparse.ArgumentParser(description="Parse required flags.")
    parser.add_argument(
        "--hf-id", type=str, required=True, help="HF ID (string, required)"
    )
    parser.add_argument(
        "--hotkey", type=str, required=True, help="Hotkey (string, required)"
    )
    parser.add_argument(
        "--coldkey", type=str, required=True, help="Coldkey (string, required)"
    )
    parser.add_argument(
        "--bittensor-wallet", type=str, default="default", help="Bittensor wallet name (string, default='default')"
    )
    parser.add_argument(
        "--bittensor-hotkey", type=str, default="default", help="Bittensor hotkey name (string, default='default')"
    )
    return parser.parse_args()

bittensor.logging.set_console()

args = get_args()


wallet = bittensor.wallet(name=args.bittensor_wallet, hotkey=args.bittensor_hotkey)

# params for chain commit
subtensor = bittensor.subtensor()


repo_id, model_id = args.hf_id.split("/")
hotkey = args.hotkey
coldkey = args.coldkey
timeout = 60 * 15
uid = 1  # Not true, but for test it doesn't matter
max_model_size_gbs = 18

job_uuid = str(uuid4())

task_sample_rate = [
    (TasksEnum.RELEVANCY.value, 1),
    (TasksEnum.HALLUCINATION.value, 1),
    (TasksEnum.ATTRIBUTION.value, 1),
    (TasksEnum.COMPLETENESS.value, 1),
]
active_tasks = [t[0] for t in task_sample_rate]

forward_start_time = int(time.time())
contest = DeValContest(
    None,
    forward_start_time,
    timeout,  # not actually used
)

miner_docker_client = MinerDockerClient()
wandb_logger = WandBLogger(None, None, active_tasks, None, force_off=True)
metadata_store = ChainModelMetadataStore(
    subtensor=subtensor, wallet=None, subnet_uid=15
)

if os.path.exists("task_repo.pkl"):
    print("Loading task repo from file")
    with open("task_repo.pkl", "rb") as f:
        task_repo = pickle.load(f)
else:
    task_repo = TaskRepository(allowed_models=allowed_models)

    print("Generating the tasks")
    task_repo.generate_all_tasks(task_probabilities=task_sample_rate)
    with open("task_repo.pkl", "wb") as f:
        pickle.dump(task_repo, f)

chain_metadata = metadata_store.retrieve_model_metadata(hotkey)
assert chain_metadata is not None, "Chain metadata not found"
miner_state = ModelState(repo_id, model_id, uid, netuid=15)
miner_state.add_miner_coldkey(coldkey)
miner_state.add_chain_metadata(chain_metadata)

print("Deciding if we should run evaluation ")
is_valid = miner_state.should_run_evaluation(
    uid, max_model_size_gbs, forward_start_time, [uid]
)

if not is_valid:
    print("Not running evaluation")
    sys.exit(1)

compute_horde_client = ComputeHordeClient(wallet.hotkey)


async def run_job():
    task_repo.refresh_models_after_load = False
    job_result = await compute_horde_client.run_epoch_on_compute_horde(miner_state, task_repo)

    print(job_result.model_hash)
    print(job_result.model_coldkey)
    print(job_result.model_state.rewards)


asyncio.run(run_job())
