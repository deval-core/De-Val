from .hallucination import HallucinationTask
from .summary_completeness import CompletenessTask
from .attribution import AttributionTask
from .relevancy import RelevancyTask
from .task import Task, TasksEnum


TASKS = {
    TasksEnum.HALLUCINATION: HallucinationTask,
    TasksEnum.COMPLETENESS: CompletenessTask,
    TasksEnum.ATTRIBUTION: AttributionTask,
    TasksEnum.RELEVANCY: RelevancyTask
}