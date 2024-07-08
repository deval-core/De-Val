import time
import torch
from typing import List
from deval.rewards import BaseRewardModel, BatchRewardOutput


class OrdinalRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "ordinal"

    def __init__(self, **kwargs):
        super().__init__()

        # TODO: integrate support for multi-label ordinal classification
        self.binary = [ 
            1.0,
            0.0,
        ]

    def reward(self, reference: str, completions: List[float]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.binary
        for completion in completions:
            t0 = time.time()
            
            if completion in classes:
                reward = 1-abs(classes.index(reference) - classes.index(completion))/(len(classes)-1)
            else:
                reward = 0 
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "ordinal",
            },
        )
        return output
