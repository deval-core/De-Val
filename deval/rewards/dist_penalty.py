import time
import torch
from typing import List
from deval.rewards import BaseRewardModel, BatchRewardOutput


class DistPenaltyRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "dist_penalty"

    def __init__(self, **kwargs):
        super().__init__()
        self.categories = [
            (0, .15),
            (0.15, 0.3),
            (0.3, 0.5),
            (0.5, 1.0001),
        ]
            
    def dist_score(self, reference: float, completion: float, classes: list[tuple]) -> float:
        if completion < 0 or completion > 1:
            return 1.0
        try:
            diff = abs(reference - completion)
            cat = [cat for cat in classes if diff >= cat[0] and diff < cat[1]][0]
            reward = (classes.index(cat)/(len(classes)-1)) 
        except Exception as e:
            reward = 1.0

        return reward
            

    def reward(self, reference: float, completions: List[float]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.categories
        for completion in completions:
            t0 = time.time()
            reward = self.dist_score(reference, completion, classes)
            
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "dist_penalty",
            },
        )
        return output
