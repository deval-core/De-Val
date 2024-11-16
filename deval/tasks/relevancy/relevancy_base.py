from deval.tasks.task import Task, TasksEnum
from deval.rewards.reward import RewardReferenceType


class RelevancyBaseTask(Task):
    name = TasksEnum.RELEVANCY.value
    desc = "Estimates the relevancy of an answer to a user query based on a provided context"
    goal = "to identify whether an answer to a user query is relevant or not"

    reward_definition = [
        dict(name="ordinal", weight=1.0, reference_type = RewardReferenceType.SCORE),
    ]
    penalty_definition = []