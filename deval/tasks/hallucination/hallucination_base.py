from deval.tasks.task import Task, TasksEnum
from deval.rewards.reward import RewardReferenceType


class HallucinationBaseTask(Task):
    name = TasksEnum.HALLUCINATION.value
    desc = "Generates a fake input context and associated claims for a hallucination evaluation task"
    goal = "Estimates the number of hallucination in a response given a RAG context"

    reward_definition = [
        dict(name="float_diff", weight=0.6, reference_type = RewardReferenceType.SCORE),
        dict(name="exact_match", weight=0.4, reference_type = RewardReferenceType.MISTAKES),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.35, reference_type = RewardReferenceType.SCORE),
        dict(name="exact_match", weight=0.5, reference_type = RewardReferenceType.MISTAKES),
    ]
