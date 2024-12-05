from deval.contest import DeValContest
from deval.validator import Validator
import time
from deval.rewards.pipeline import RewardPipeline
from deval.task_repository import TaskRepository
from dotenv import load_dotenv, find_dotenv
from deval.model.model_state import ModelState
from deval.tasks.task import TasksEnum
from deval.api.miner_docker_client import MinerDockerClient
from deval.utils.logging import WandBLogger
from deval.model.chain_metadata import ChainModelMetadataStore
import bittensor as bt 





# initialize
_ = load_dotenv(find_dotenv())
allowed_models = ["gpt-4o-mini"]

repo_id = "deval-core"
model_id = "base-eval"
timeout = 600
uid = 1
max_model_size_gbs = 18

# params for chain commit 
model_url = "deval-core/base-eval-test"
subtensor = bt.subtensor(network='test')
hotkey = "5HGiNFJApXHkMV9RoWAVYhKopuXG7VpvirABf3EyKcn1kAkn"


print("Initializing tasks and contest")
task_repo = TaskRepository(allowed_models=allowed_models)

task_sample_rate = [
    (TasksEnum.RELEVANCY.value, 1),
    #(TasksEnum.HALLUCINATION.value, 1),
    #(TasksEnum.ATTRIBUTION.value, 1),
    #(TasksEnum.COMPLETENESS.value, 1)
]
active_tasks = [t[0] for t in task_sample_rate]
reward_pipeline = RewardPipeline(
    selected_tasks=active_tasks, device="cpu"
)


forward_start_time = time.time()
contest = DeValContest(
    reward_pipeline, 
    forward_start_time, 
    timeout
)

miner_docker_client = MinerDockerClient()
wandb_logger = WandBLogger(None, None, active_tasks, None, force_off=True)
metadata_store = ChainModelMetadataStore(
    subtensor=subtensor, wallet=None, subnet_uid=202
)

print("Generating the tasks")
task_repo.generate_all_tasks(task_probabilities=task_sample_rate)

chain_metadata = metadata_store.retrieve_model_metadata(hotkey)
miner_state = ModelState(repo_id, model_id, uid, chain_metadata)

print("Deciding if we should run evaluation ")
is_valid = miner_state.should_run_evaluation(
    uid, max_model_size_gbs, forward_start_time, [uid]
)

if is_valid:
    print("Running evaluation and starting epoch")
    miner_state = Validator.run_epoch(
        contest,
        miner_state, 
        task_repo, 
        miner_docker_client,
        wandb_logger
    )
    print("Completed epoch")

print("updating contest with rewards and ranking")
contest.update_model_state_with_rewards(miner_state) 
weights = contest.rank_and_select_winners(task_sample_rate)


print(weights)