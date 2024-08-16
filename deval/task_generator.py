from deval.tasks import (
    Task,
)
from deval.tasks import TasksEnum, TASKS


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
    from deval.llms import OpenAIPipeline

    llm_pipeline = OpenAIPipeline(
        model_id="gpt-4o-mini",
        mock=False,
    )  

    task_name = TasksEnum.ATTRIBUTION.value

    task = create_task(llm_pipeline, task_name)
    print(task)
