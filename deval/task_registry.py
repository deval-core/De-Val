from .tasks import Task, HallucinationTask, CompletenessTask, AttributionTask, RelevancyTask
from .tools import WikiDataset, GenericDataset, AttributionDataset

from pydantic import BaseModel


# TODO: switch completion as float with a list of these
class TaskResult(BaseModel):
    name: str
    score: int
    reasoning: str | None


hallucination_task, hallucination_dataset = HallucinationTask.name, [GenericDataset.name]
completeness_task, completeness_dataset = CompletenessTask.name, [GenericDataset.name]
attribution_task, attribution_dataset = AttributionTask.name, [AttributionDataset.name]
relevancy_task, relevancy_dataset = RelevancyTask.name, [WikiDataset.name]

# task storage management
TASK_REGISTRY = {
    hallucination_task: hallucination_dataset,
    #completeness_task: completeness_dataset,
    #attribution_task: attribution_dataset,
    #relevancy_task: relevancy_dataset
}

