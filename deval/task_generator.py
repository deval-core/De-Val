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


def create_task(llm_pipeline, task_name: str) -> Task:
    wiki_based_tasks = [TasksEnum.RELEVANCY.value]
    generic_text_tasks = [TasksEnum.HALLUCINATION.value, TasksEnum.COMPLETENESS.value]
    attribution_text_tasks = [TasksEnum.ATTRIBUTION.value]
    
    if task_name in wiki_based_tasks:
        dataset = WikiDataset()

    elif task_name in generic_text_tasks:
        dataset = GenericDataset()
    
    elif task_name in attribution_text_tasks:
        dataset = AttributionDataset()
    
    if task_name == TasksEnum.HALLUCINATION.value:
        task = HallucinationTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )

    elif task_name == TasksEnum.COMPLETENESS.value:
        task = CompletenessTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )
    
    elif task_name == TasksEnum.ATTRIBUTION.value:
        task = AttributionTask(
            llm_pipeline=llm_pipeline, 
            context=dataset.next(),
        )
    
    elif task_name == TasksEnum.RELEVANCY.value:
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
        model_id="gpt-4o-mini",
        mock=False,
    )  

    task_name = TasksEnum.ATTRIBUTION.value

    task = create_task(llm_pipeline, task_name)
    print(task)
