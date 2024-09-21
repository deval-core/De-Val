from .hallucination import HallucinationTask
from .summary_completeness import CompletenessTask
from .attribution import AttributionTask
from .relevancy import RelevancyTask
from .task import Task, TasksEnum
from deval.tools import (
    WikiDataset, GenericDataset, AttributionDataset
)

# To add a new task, add a new entry here with the relevant function and dataset
TASKS = {
    TasksEnum.RELEVANCY.value: {
        "task_function": RelevancyTask,
        "dataset": WikiDataset,
        "task_p": 0.25,
    },
    TasksEnum.HALLUCINATION.value: {
        "task_function": HallucinationTask,
        "dataset": GenericDataset,
        "task_p": 0.25,
    },
    TasksEnum.COMPLETENESS.value: {
        "task_function": CompletenessTask,
        "dataset": GenericDataset,
        "task_p": 0.25,
    },
    TasksEnum.ATTRIBUTION.value: {
        "task_function": AttributionTask,
        "dataset": AttributionDataset,
        "task_p": 0.25,
    }
}