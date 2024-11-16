from deval.tasks.task import Task, TasksEnum
from deval.rewards.reward import RewardReferenceType


class CompletenessBaseTask(Task):
    name = TasksEnum.COMPLETENESS.value
    desc = "Generates a fake input context and associated summary for a summary completeness evaluation task"
    goal = "Estimates the comprehensiveness of a summary"

    reward_definition = [
        dict(name="float_diff", weight=0.5, reference_type = RewardReferenceType.SCORE),
        dict(name="rouge", weight=0.25, reference_type = RewardReferenceType.MISTAKES),
        dict(name="relevance", weight=0.25, reference_type = RewardReferenceType.MISTAKES),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.25, reference_type = RewardReferenceType.SCORE),
        dict(name="rouge", weight=0.125, reference_type = RewardReferenceType.MISTAKES),
        dict(name="relevance", weight=0.125, reference_type = RewardReferenceType.MISTAKES),
    ]