from .hallucination import HallucinationTask
from .summary_completeness import CompletenessTask
from .task import Task, TasksEnum


TASKS = {
    TasksEnum.HALLUCINATION: HallucinationTask,
    TasksEnum.COMPLETENESS: CompletenessTask
}