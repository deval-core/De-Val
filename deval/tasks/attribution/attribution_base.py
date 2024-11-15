from deval.tasks.task import Task, TasksEnum
from deval.rewards.reward import RewardReferenceType


class AttributionBaseTask(Task):
    name = TasksEnum.ATTRIBUTION.value
    desc = "Generate a context and associated action items for a misattribution evaluation task"
    goal = "Estimates the number of correctly attributed action items in a response given a RAG context"

    reward_definition = [
        dict(name="float_diff", weight=0.5, reference_type = RewardReferenceType.SCORE),
        dict(name="exact_match", weight=0.5, reference_type = RewardReferenceType.MISTAKES),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.25, reference_type = RewardReferenceType.SCORE),
        dict(name="exact_match", weight=0.5, reference_type = RewardReferenceType.MISTAKES),
    ]