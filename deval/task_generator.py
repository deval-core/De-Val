from deval.tasks import (
    Task,
    HallucinationTask,
    CompletenessTask
)
from deval.tools import (
    WikiDataset,
    GenericDataset
)
from deval.tasks import TasksEnum


def create_task(llm_pipeline, task_name: TasksEnum) -> Task:
    wiki_based_tasks = []
    generic_text_tasks = [TasksEnum.HALLUCINATION, TasksEnum.COMPLETENESS]
    
    if task_name in wiki_based_tasks:
        dataset = WikiDataset()

    if task_name in generic_text_tasks:
        dataset = GenericDataset()
    
    if task_name == TasksEnum.HALLUCINATION:
        task = HallucinationTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )

    if task_name == TasksEnum.COMPLETENESS:
        task = CompletenessTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )

    else:
        raise ValueError(f"Task {task_name} not supported. Please choose a valid task")

    return task



if __name__ == "__main__":
    from deval.llms import OpenAIPipeline

    llm_pipeline = OpenAIPipeline(
        model_id="gpt-3.5-turbo-0125",
        mock=False,
    )  

    task_name = TasksEnum.COMPLETENESS

    task = create_task(llm_pipeline, task_name)
    print(task)
