import time
import torch
from typing import List
from deval.rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum

# TODO: Improve to normalize scores between 0 and 1 or at least ensure consistency across runs
class FloatDiffModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "float_diff"

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def numeric_score(reference: float, pred: float) -> float:
        """Compute a score based on the difference between a reference and a prediction."""

        if pred is None or pred < 0 or pred > 1:
            return 0.0

        try:
            if pred == reference:
                return 1.0
            # Compute the difference
            diff = abs(reference - pred) 
            # Make sure the difference is between 0 and 1
            diff = min(diff, 1)
            # Clip any very small scores
            if diff > 0.999:
                diff = 1.0
            return 1.0 - diff
        except Exception:
            return 0.0

    def reward(self, reference: float, completions: List[float]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            reward = self.numeric_score(reference, completion)
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "numeric",
            },
        )
        return output
