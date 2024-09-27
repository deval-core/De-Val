from deval.tasks import TasksEnum, Task, TASKS
from deval.llms.openai_llm import OpenAILLM
from deval.llms.bedrock_llm import AWSBedrockLLM
from deval.llms.base_llm import BaseLLM
from deval.llms.config import LLMAPIs, LLMArgs, LLMFormatType, SUPPORTED_MODELS
import os 
import numpy as np


class TaskRepository:

    tasks: list[Task] = [] 

    def __init__(self, allowed_models: list[str] | None = None):
        # initialize available models 
        self.supported_models = SUPPORTED_MODELS

        if allowed_models is not None:
            self.supported_models = self.filter_to_allowed_models(allowed_models)

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
        
        task_extract = TASKS.get(task_name, None)
        if task_extract is None:
            raise ValueError(f"Task {task_name} not supported. Please choose a valid task")

        task_function = task_extract['task_function']
        dataset = task_extract['dataset']()

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
            for i in range(n):
                llm_pipeline = self.get_random_llm()
                task = self.create_task(llm_pipeline, task_name)
                self.tasks.append(task)

    def get_all_tasks(self) -> Task:
        for task in self.tasks:
            yield task



if __name__ == "__main__":
    from deval.llms.openai_llm import OpenAILLM
    from deval.llms.config import LLMArgs, LLMFormatType, LLMAPIs
    from dotenv import load_dotenv, find_dotenv
    
    _ = load_dotenv(find_dotenv())

    task_sample_rate = [
        (TasksEnum.RELEVANCY.value, 1),
        (TasksEnum.HALLUCINATION.value, 1),
        (TasksEnum.ATTRIBUTION.value, 1),
    ]

    allowed_models = ["gpt-4o-mini"]
    task_repo = TaskRepository(allowed_models=allowed_models)

    task_repo.generate_all_tasks(task_sample_rate)
    for t in task_repo.get_all_tasks():
        print(t)
