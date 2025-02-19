from collections.abc import Iterator

from deval.tasks.task import TasksEnum, Task
from deval.llms.openai_llm import OpenAILLM
from deval.llms.bedrock_llm import AWSBedrockLLM
from deval.llms.base_llm import BaseLLM
from deval.llms.config import LLMAPIs, LLMArgs, LLMFormatType, SUPPORTED_MODELS
from deval.tasks.hallucination import (
    HallucinationWikipediaTopicTask, 
    HallucinationBaseTask,
    HallucinatioGenerationTask,
    HallucinationWikipediaGenTask
)
from deval.tasks.summary_completeness import (
    CompletenessWikipediaTask,
    CompletenessBaseTask,
    CompletenessGenerationTask
)
from deval.tasks.attribution import (
    AttributionGenerationTask,
    AttributionBaseTask
)
from deval.tasks.relevancy import (
    RelevancyWikipediaTask,
    RelevancyBaseTask
)
from deval.tasks.task import Task, TasksEnum
from deval.tools import (
    WikiDataset, GenericDataset, AttributionDataset
)
import os 
import numpy as np
import random 


TASKS = {
    TasksEnum.RELEVANCY.value: {
        "base_function": RelevancyBaseTask,
        "tasks": [
            {
                "task_function": RelevancyWikipediaTask,
                "dataset": WikiDataset,
            }
        ],
        "task_p": 1,
    },
    TasksEnum.HALLUCINATION.value: {
        "base_function": HallucinationBaseTask,
        "tasks": [
            {
                "task_function": HallucinatioGenerationTask,
                "dataset": GenericDataset,
            },
            {
                "task_function": HallucinationWikipediaTopicTask,
                "dataset": WikiDataset,
            },
            {
                "task_function": HallucinationWikipediaGenTask,
                "dataset": WikiDataset,
            }
        ],
        "task_p": 1,
    },
    TasksEnum.COMPLETENESS.value: {
        "base_function": CompletenessBaseTask,
        "tasks": [
            {
                "task_function": CompletenessWikipediaTask,
                "dataset": WikiDataset,
            },
            {
                "task_function": CompletenessGenerationTask,
                "dataset": GenericDataset,
            }
        ],
        "task_p": 1,
    },
    TasksEnum.ATTRIBUTION.value: {
        'base_function': AttributionBaseTask,
        "tasks": [
            {
                "task_function": AttributionGenerationTask,
                "dataset": AttributionDataset,
            }
        ],
        "task_p": 1,
    }
}

class TaskRepository:

    def __init__(self, allowed_models: list[str] | None = None, refresh_models_after_load: bool = True):
        self.tasks: dict[TasksEnum, list[Task]] = {} 

        # initialize available models 
        self.supported_models = SUPPORTED_MODELS

        if allowed_models is not None:
            self.supported_models = self.filter_to_allowed_models(allowed_models)

        self.refresh_models_after_load = refresh_models_after_load
        self.available_models = self.get_available_models()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['available_models']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.refresh_models_after_load:
            self.available_models = self.get_available_models()

    def filter_to_allowed_models(self, allowed_models: list[str] | None) -> dict:
        filtered_dict = {}
        for key, value in self.supported_models.items():
            filtered_values = [model for model in value if model in allowed_models]
            filtered_dict[key] = filtered_values

        return filtered_dict

    def get_available_models(self) -> list[BaseLLM]:
        available_models = []

        # TODO: integrate with config settings 
        model_kwargs = LLMArgs(format = LLMFormatType.TEXT)

        # go through each of our models and store the available ones
        openai_key = os.getenv("OPENAI_API_KEY", None)
        if openai_key is not None:
            for model_id in self.supported_models.get(LLMAPIs.OPENAI, []):
                openai_llm = OpenAILLM(
                    model_id=model_id,
                    model_kwargs=model_kwargs
                )
                available_models.append(openai_llm)
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", None)
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
        if aws_access_key is not None and aws_secret_key is not None:
            for model_id in self.supported_models.get(LLMAPIs.BEDROCK, []):
                bedrock_llm = AWSBedrockLLM(
                    model_id=model_id,
                    model_kwargs=model_kwargs
                )
                is_model_accessible = bedrock_llm.check_model_id_access()
                
                if is_model_accessible:
                    available_models.append(bedrock_llm)

            
        # we only require that at least one model can be run 
        if len(available_models) == 0:
            raise ValueError("Provide at least one API token to run at least one model")

        return available_models

    def get_random_llm(self) -> BaseLLM:
        return np.random.choice(self.available_models)

    def create_task(self, llm_pipeline: BaseLLM, task_name: str) -> Task:
        
        task_extract = TASKS.get(task_name, {}).get('tasks', None)
        if task_extract is None:
            raise ValueError(f"Task {task_name} not supported. Please choose a valid task")

        selected_task = random.sample(task_extract, k = 1)[0]
        task_function = selected_task['task_function']
        dataset = selected_task['dataset']()

        task = task_function(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )

        return task

    def generate_all_tasks(
        self, 
        task_probabilities: list[tuple()],
    ) -> None:
        # loops through and stores all tasks to be evaluated against in the epoch 
        for task_name, n in task_probabilities:
            self.tasks[task_name] = []
            for i in range(n):
                print(f"Generating Task Name: {task_name}, iteration: {i}")
                llm_pipeline = self.get_random_llm()
                try:
                    task = self.create_task(llm_pipeline, task_name)
                    self.tasks[task_name].append(task)
                except:
                    continue

    def get_all_tasks(self) -> Iterator[tuple[str, list[Task]]]:
        for task_name, tasks in self.tasks.items():
            yield task_name, tasks



if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    
    _ = load_dotenv(find_dotenv())

    task_sample_rate = [
        (TasksEnum.RELEVANCY.value, 1),
        #(TasksEnum.HALLUCINATION.value, 1),
        #(TasksEnum.ATTRIBUTION.value, 1),
        #(TasksEnum.COMPLETENESS.value, 1),
    ]

    allowed_models = ["gpt-4o-mini"]
    task_repo = TaskRepository(allowed_models=allowed_models)

    task_repo.generate_all_tasks(task_sample_rate)
    for task_name, tasks in task_repo.get_all_tasks():
        print(tasks)
