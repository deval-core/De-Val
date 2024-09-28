from enum import Enum
import torch
from dataclasses import dataclass


class RewardModelTypeEnum(Enum):
    WEIGHTED_REWARD = "reward"
    FILTER_REWARD = "filter"
    PENALTY = "penalty"

class RewardReferenceType(Enum):
    SCORE = "score"
    MISTAKES = "mistakes"

@dataclass
class RewardEvent:
    """Contains rewards for all the responses in a batch"""

    model_name: str
    rewards: torch.FloatTensor
    rewards_normalized: torch.FloatTensor
    timings: torch.FloatTensor
    model_type: RewardModelTypeEnum
    batch_time: float
    extra_info: dict

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.model_name}_raw_{self.model_type.value}": self.rewards.tolist(),
            f"{self.model_name}_{self.model_type.value}": self.rewards_normalized.tolist(),
            f"{self.model_name}_{self.model_type.value}_timings": self.timings.tolist(),
            f"{self.model_name}_{self.model_type.value}_batch_time": self.batch_time,
            f"{self.model_name}_{self.model_type.value}_extra_info": self.extra_info,
        }

