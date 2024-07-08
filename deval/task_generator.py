from deval.tasks import (
    Task,
    HallucinationTask,
    CompletenessTask,
    AttributionTask,
    RelevancyTask
)
from deval.tools import (
    WikiDataset,
    GenericDataset,
    AttributionDataset
)
from deval.tasks import TasksEnum


def create_task(llm_pipeline, task_name: TasksEnum) -> Task:
    wiki_based_tasks = [TasksEnum.RELEVANCY]
    generic_text_tasks = [TasksEnum.HALLUCINATION, TasksEnum.COMPLETENESS]
    attribution_text_tasks = [TasksEnum.ATTRIBUTION]
    
    if task_name in wiki_based_tasks:
        dataset = WikiDataset()

    elif task_name in generic_text_tasks:
        dataset = GenericDataset()
    
    elif task_name in attribution_text_tasks:
        dataset = AttributionDataset()
    
    if task_name == TasksEnum.HALLUCINATION:
        task = HallucinationTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )

    elif task_name == TasksEnum.COMPLETENESS:
        task = CompletenessTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )
    
    elif task_name == TasksEnum.ATTRIBUTION:
        task = AttributionTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )
    
    elif task_name == TasksEnum.RELEVANCY:
        task = RelevancyTask(
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

    task_name = TasksEnum.RELEVANCY

    task = create_task(llm_pipeline, task_name)
    print(task)
