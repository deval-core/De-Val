from .hallucination import HallucinationTask
from .summary_completeness import CompletenessTask
from .attribution import AttributionTask
from .relevancy import RelevancyTask
from .task import Task, TasksEnum


TASKS = {
    TasksEnum.HALLUCINATION.value: HallucinationTask,
    TasksEnum.COMPLETENESS.value: CompletenessTask,
    TasksEnum.ATTRIBUTION.value: AttributionTask,
    TasksEnum.RELEVANCY.value: RelevancyTask
}