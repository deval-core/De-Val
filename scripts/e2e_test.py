"""
This file runs through a complete epoch, but with a single miner
"""

from deval.model.huggingface_model import HuggingFaceModel
from deval.model.model_state import ModelState
from deval.tasks.task import TasksEnum
from dotenv import load_dotenv, find_dotenv
from deval.task_repository import TaskRepository
from deval.agent import HumanAgent
from deval.protocol import init_request_from_task, BtEvalResponse
import sys

# initialize
_ = load_dotenv(find_dotenv())


repo_id = "deval-core"
model_id = "base-eval-test"

task_sample_rate = [
    (TasksEnum.RELEVANCY.value, 1),
    (TasksEnum.HALLUCINATION.value, 1),
    (TasksEnum.ATTRIBUTION.value, 1),
    (TasksEnum.COMPLETENESS.value, 1)
]



# run through samples for miner
task_repo = TaskRepository(allowed_models=allowed_models)
miner_state = ModelState(
    repo_id=repo_id,
    model_id=model_id,
    uid=1
)
print(f"Last commit date {miner_state.last_commit_date}")
print(f"Last safetensor update {miner_state.last_safetensor_update}") 


model_url = miner_state.get_model_url()
print(f"Pulling model from {model_url}")

model_dir = HuggingFaceModel.pull_model_and_files(model_url, "../eval_llm")
print(f"Succesfully pulled model to {model_dir}")

sys.path.append(model_dir)
from model.pipeline import DeValPipeline
pipe = DeValPipeline("de_val", model_dir = model_dir)
print("Successfully generated the pipeline")

print("Generating our tasks")
task_repo.generate_all_tasks(task_sample_rate)
responses = []

for task_name, tasks in task_repo.get_all_tasks():
    print(f"Starting run for task: {task_name}")
    for task in tasks:
        agent = HumanAgent(
            task=task
        )
        request = init_request_from_task(task)
        response = HuggingFaceModel.query_hf_model(pipe, request)
        print(f"Generated response: {response}")
        bt_response = BtEvalResponse(
            uid = miner_state.uid,
            response = response,
            human_agent = agent
        )

        responses.append(bt_response)
        
