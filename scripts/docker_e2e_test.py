from deval.contest import DeValContest
from deval.validator import Validator
import time
from deval.rewards.reward import RewardResult
from deval.rewards.pipeline import RewardPipeline
from deval.task_repository import TaskRepository
from dotenv import load_dotenv, find_dotenv
from deval.model.model_state import ModelState
from deval.tasks.task import TasksEnum
from deval.api.miner_docker_client import MinerDockerClient


# initialize
_ = load_dotenv(find_dotenv())
allowed_models = ["gpt-4o-mini"]

repo_id = "deval-core"
model_id = "base-eval-test"
timeout = 10
uid = 1
max_model_size_gbs = 18

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

print("Generating the tasks")
task_repo.generate_all_tasks(task_probabilities=task_sample_rate)

miner_state = ModelState(repo_id, model_id, uid)

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
    )
    print("Completed epoch")

print("updating contest with rewards and ranking")
contest.update_model_state_with_rewards(miner_state) 
weights = contest.rank_and_select_winners(task_sample_rate)


print(weights)