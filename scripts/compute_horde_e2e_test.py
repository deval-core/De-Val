import argparse
import asyncio
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
from deval.rewards.pipeline import RewardPipeline
from deval.task_repository import TaskRepository
from deval.tasks.task import TasksEnum
from deval.utils.logging import WandBLogger

wallet = bittensor.wallet(name="default", hotkey="default")

# params for chain commit
subtensor = bittensor.subtensor()

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

    return parser.parse_args()

bittensor.logging.set_console()

args = get_args()


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
reward_pipeline = RewardPipeline(selected_tasks=active_tasks, device="cuda")

forward_start_time = int(time.time())
contest = DeValContest(
    reward_pipeline,
    forward_start_time,
    timeout,  # not actually used
)

miner_docker_client = MinerDockerClient()
wandb_logger = WandBLogger(None, None, active_tasks, None, force_off=True)
metadata_store = ChainModelMetadataStore(
    subtensor=subtensor, wallet=None, subnet_uid=15
)

print("Initializing tasks and contest")
task_repo = TaskRepository(
    allowed_models=allowed_models, refresh_models_after_load=False
)

print("Generating the tasks")
task_repo.generate_all_tasks(task_probabilities=task_sample_rate)

chain_metadata = metadata_store.retrieve_model_metadata(hotkey)
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
    new_miner_state = await compute_horde_client.run_epoch_on_compute_horde(
        contest,
        miner_state,
        task_repo,
    )

    print(new_miner_state.rewards)


asyncio.run(run_job())
