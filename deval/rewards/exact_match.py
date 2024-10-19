import time
import torch
from rouge import Rouge
from deval.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)


class ExactMatchRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "exact_match"

    def __init__(self, ngram="rouge-l", metric="f", avg=False, device=None, **kwargs):
        super().__init__()
        self.ngram = ngram
        self.metric = metric
        self.avg = avg
        self.rouge = Rouge(**kwargs)

        self.exact_match_threshold = 0.85

    def rouge_score(self, reference, completion):
        if not completion or not reference:
            return 0.0
        return self.rouge.get_scores(reference, completion, avg=self.avg)[0][
            self.ngram
        ][self.metric]

    def check_match(self, reference_mistake: str, completion_mistake: str) -> bool:
        rouge_score = self.rouge_score(reference_mistake, completion_mistake)
        if rouge_score >= self.exact_match_threshold:
            return True
        else:
            return False

    def reward(self, reference: list[str], completion: list[str]) -> BatchRewardOutput:
        """Compute the number of exact matches scores given a completion and reference pair."""
        rewards = []
        timings = []

        t0 = time.time()
        matches = []
        for reference_mistake in reference:
            matched_reference = False
            for completion_mistake in completion:
                is_match = self.check_match(reference_mistake, completion_mistake)
                if is_match:
                    matched_reference = True
            
            matches.append(int(matched_reference))

        if len(matches) > 0:
            rewards.append(sum(matches) / len(matches))
        else:
            rewards.append(0)
            
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
