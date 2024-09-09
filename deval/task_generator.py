from deval.tasks import TasksEnum, Task, TASKS
from deval.llms.openai_llm import OpenAILLM
from deval.llms.bedrock_llm import AWSBedrockLLM
from deval.llms.base_llm import LLMArgs, LLMFormatType, BaseLLM, LLMAPIs
import os 
import numpy as np


class TaskGenerator:

    def __init__(self):
        # initialize available models 
        self.supported_models = {
            LLMAPIs.OPENAI : ["gpt-4o-mini", "gpt-4o-2024-08-06"],
            LLMAPIs.BEDROCK : ["anthropic.claude-3-haiku-20240307-v1:0", "cohere.command-r-plus-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0",  "mistral.mistral-small-2402-v1:0", "mistral.mistral-large-2402-v1:0"]
        }
        self.available_models = self.get_available_models()

    def get_available_models(self) -> list[BaseLLM]:
        available_models = []

        # TODO: integrate with config settings 
        model_kwargs = LLMArgs(format = LLMFormatType.TEXT)

        # go through each of our models and store the available ones
        openai_key = os.getenv("OPENAI_API_KEY", None)
        if openai_key is not None:
            for model_id in self.supported_models.get(LLMAPIs.OPENAI):
                openai_llm = OpenAILLM(
                    model_id=model_id,
                    model_kwargs=model_kwargs
                )
                available_models.append(openai_llm)
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", None)
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
        if aws_access_key is not None and aws_secret_key is not None:
            for model_id in self.supported_models.get(LLMAPIs.BEDROCK):
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

    def create_task(self, llm_pipeline, task_name: str) -> Task:
        
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



if __name__ == "__main__":
    from deval.llms.openai_llm import OpenAILLM
    from deval.llms.base_llm import LLMArgs, LLMFormatType, LLMAPIs
    from dotenv import load_dotenv, find_dotenv
    
    task_name = TasksEnum.HALLUCINATION.value
    _ = load_dotenv(find_dotenv())

    task_generator = TaskGenerator()

    llm_pipeline = [
        model for model in task_generator.available_models 
        if model.api == LLMAPIs.OPENAI 
        and model.model_id == "gpt-4o-mini"
    ][0]
 
    task = task_generator.create_task(llm_pipeline, task_name)
    print(task)
