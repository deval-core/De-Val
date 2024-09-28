import time
import torch
from rouge import Rouge
from deval.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)


class RougeRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "rouge"

    def __init__(self, ngram="rouge-l", metric="f", avg=False, device=None, **kwargs):
        super().__init__()
        self.ngram = ngram
        self.metric = metric
        self.avg = avg
        self.rouge = Rouge(**kwargs)

    def rouge_score(self, reference, completion):
        if not completion or not reference:
            return 0.0
        return self.rouge.get_scores(reference, completion, avg=self.avg)[0][
            self.ngram
        ][self.metric]

    def reward(self, reference: list[str], completions: list[list[str]]) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        rewards = []
        timings = []
        reference = "\n".join(r for r in reference)

        for completion in completions:
            t0 = time.time()
            completion = "\n".join(c for c in completion)
            rewards.append(self.rouge_score(reference, completion))
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "ngram": self.ngram,
                "metric": self.metric,
                "avg": self.avg,
            },
        )

        return output
