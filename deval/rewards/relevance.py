import os
import time
import torch
from angle_emb import AnglE
from torch.nn.functional import cosine_similarity
from deval.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)


class RelevanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "relevance"

    def __init__(self, threshold=None, device=None, pooling_strategy="cls"):
        super().__init__()
        self.threshold = threshold
        model_path = os.environ.get("ANGLE_MODEL_PATH", "WhereIsAI/UAE-Large-V1")
        self.model = AnglE.from_pretrained(
            model_path, pooling_strategy=pooling_strategy, device=device
        )
        self.model.tokenizer._pad_token = self.model.tokenizer.pad_token
        if device.startswith("cuda"):
            # This line is necessary to pass the model to the device defined at its initialization
            self.model = self.model.cuda()

    def reward(self, reference: list[str], completion: list[str]) -> BatchRewardOutput:
        """Calculates the cosine similarity between sentence embeddings of the reference and completions.
        We subtract a baseline score which is what an empty string would get (a failed completion). This is usually around 0.35
        We also clip the rewards between 0 and 1. The maximum effective score is around 0.65
        """
        reference = "\n".join(r for r in reference)
        reference_embedding = self.model.encode(reference, to_numpy=False)
        rewards = []
        timings = []
        
        # baseline is the cosine similarity between the reference and an empty string
        baseline = cosine_similarity(
            reference_embedding.reshape(1, -1),
            self.model.encode("", to_numpy=False).reshape(1, -1),
        )

        t0 = time.time()
        completion = "\n".join(c for c in completion)
        emb = self.model.encode(completion, to_numpy=False)
        
        # Calculate cosine similarity between reference and completion embeddings, and subtract baseline
        score = (
            cosine_similarity(
                reference_embedding.reshape(1, -1), emb.reshape(1, -1)
            )
            - baseline
        )

        rewards.append(score)
        timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards).clip(min=0, max=1),
            timings=torch.FloatTensor(timings),
            extra_info={"threshold": self.threshold},
        )

        return output
