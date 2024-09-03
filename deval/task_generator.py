from deval.tasks import TasksEnum, Task, TASKS



def create_task(llm_pipeline, task_name: str) -> Task:
    
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
    from deval.llms.base_llm import LLMArgs, LLMFormatType
    from dotenv import load_dotenv, find_dotenv
    
    task_name = TasksEnum.HALLUCINATION.value
    _ = load_dotenv(find_dotenv())

    model_kwargs = LLMArgs(format = LLMFormatType.TEXT)
    llm_pipeline = OpenAILLM(
        model_id="gpt-4o-mini",
        system_prompt="You are a helpful AI assistant",
        model_kwargs=model_kwargs
    )
 
    task = create_task(llm_pipeline, task_name)
    print(task)
