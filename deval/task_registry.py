from .tasks import Task, HallucinationTask, CompletenessTask
from .tools import WikiDataset, GenericDataset
from deval.tasks import TasksEnum

from pydantic import BaseModel


class TaskResult(BaseModel):
    name: TasksEnum
    score: int
    reasoning: str | None


hallucination_task, hallucination_dataset = HallucinationTask.name, [GenericDataset.name]
completeness_task, completeness_dataset = CompletenessTask.name, [GenericDataset.name]

# task storage management
TASK_REGISTRY = {
    hallucination_task: hallucination_dataset,
    completeness_task: completeness_dataset
}

