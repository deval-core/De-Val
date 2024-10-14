import torch
import time
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from deval.rewards.models import RewardEvent, RewardModelTypeEnum, RewardReferenceType
from deval.protocol import BtEvalResponse


class RewardResult:
    def __init__(self, reward_pipeline, responses: List[BtEvalResponse], device):
        """Passes the responses through the reward models and calculates the total reward

        Args:
            reward_pipeline (RewardPipeline): List of all loaded/ative reward models
            task (Task): Task instance which contains reward_definition (list of reward model requirements) and a reference answer (str)
            responses (list[BtEvalResponse]): Network responses to the prompt
            device (str): Device to run the reward models on
        """ 

        self.reward_pipeline = reward_pipeline
        self.responses = responses
        self.device = device
        self.rewards = []
        self.all_reward_events = []
        self.all_penalty_events = []

        for r in responses:
            reference_score = r.human_agent.reference
            reference_mistakes = r.human_agent.reference_mistakes
            reference_true_values = r.human_agent.reference_true_values

            task_rewards = r.human_agent.task.reward_definition
            task_penalties = r.human_agent.task.penalty_definition
            
            reward_events = self.reward_responses(
                miner_score=r.response.score,
                miner_extracted_items=r.response.mistakes,
                reference_score=reference_score,
                reference_extracted_items=reference_mistakes,
                models=task_rewards,
                reward_type=RewardModelTypeEnum.WEIGHTED_REWARD,
            )
            self.all_reward_events.append(reward_events)

            penalty_events = self.reward_responses(
                miner_score=r.response.score,
                miner_extracted_items=r.response.mistakes,
                reference_score=reference_score,
                reference_extracted_items=reference_true_values,
                models=task_penalties,
                reward_type=RewardModelTypeEnum.PENALTY,
            )
            self.all_penalty_events.append(penalty_events)

            reward = self.total_reward(
                reward_events,
                penalty_events,
                task_rewards,
                task_penalties
            )
            self.rewards.append(reward)


    def __state_dict__(self):
        # split by task where i = 0 is the first task computed
        state = {}
        for i in range(len(self.rewards)):
            state[i] = {"rewards": self.rewards[i]}
            for event in self.all_reward_events[i] + self.all_penalty_events[i]:
                state[i].update(event.asdict())
        return state

    def reward_responses(
        self, 
        miner_score: float, 
        miner_extracted_items: list[str],
        reference_score: float, 
        reference_extracted_items: list[str], 
        models: List[dict], 
        reward_type: RewardModelTypeEnum
    ) -> List[RewardEvent]:
        """Calculates the rewards for the responses given the task and returns a RewardEvent for each reward model
        reward_events: List[RewardEvent] = [
            RewardEvent(model_name='rouge', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
            RewardEvent(model_name='relevance', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
        ]
        """
        reward_events = []

        for reward_info in models:
            # Select the reward model from preloaded reward model pipeline
            reward_model = self.reward_pipeline.get(reward_info["name"])
            if not reward_model:
                raise ValueError(
                    f"Reward model {reward_info['name']} not supported. Please choose from {self.reward_pipeline.keys()}"
                )
            # Compute the rewards for the responses given the prompt
            reference_type = reward_info.get("reference_type")
            reference = reference_score if reference_type == RewardReferenceType.SCORE else reference_extracted_items

            if reference_type == RewardReferenceType.SCORE:
                completion = miner_score
                reference = reference_score 
        
            if reference_type == RewardReferenceType.MISTAKES:
                completion = miner_extracted_items
                reference = reference_extracted_items

            reward_event = reward_model.apply(
                reference, completion, reward_type=reward_type
            )
            reward_events.append(reward_event)

        return reward_events

    def total_reward(
        self, 
        reward_events, 
        penalty_events,
        reward_models,
        penalty_models
    ) -> torch.FloatTensor:
        """Combines the rewards from all the reward models into a single reward tensor"""
        # Compute the rewards for the responses given the prompt        
        rewards = torch.zeros(
            (1,), dtype=torch.float32, device=self.device
        )

        for event in reward_events:
            for reward_info in filter(
                lambda x: x["name"] == event.model_name, reward_models
            ):
                rewards += reward_info["weight"] * event.rewards.to(self.device)

        for event in penalty_events:
            for reward_info in filter(
                lambda x: x["name"] == event.model_name, penalty_models
            ):
                rewards *= 1 - reward_info["weight"] * event.rewards.to(self.device)

        return rewards

    def __str__(self):
        return f"{self.__class__.__name__}(reward_events={self.all_reward_events}; penalty_events={self.all_penalty_events}; rewards={self.rewards!r})"


@dataclass
class BatchRewardOutput:
    rewards: torch.FloatTensor
    timings: torch.FloatTensor
    extra_info: dict

    def __post_init__(self):
        if self.rewards.shape != self.timings.shape:
            raise ValueError(
                f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}"
            )

        self.rewards_normalized = (self.rewards - self.rewards.min()) / (
            self.rewards.max() - self.rewards.min() + 1e-6
        )


class BaseRewardModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reward(self, reference: float, completions: List[float]) -> BatchRewardOutput:
        pass

    def apply(
        self, 
        reference: float | list[str], # score or mistakes 
        completions: float | list[str], # score or mistakes 
        reward_type: RewardModelTypeEnum, 
    ) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(reference, completions)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            model_name=self.name,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            model_type=reward_type,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"



if __name__ == "__main__":
    from deval.tasks.task import TasksEnum
    from deval.task_repository import TaskRepository
    from deval.agent import HumanAgent
    from deval.protocol import BtEvalResponse
    from deval.api.models import EvalResponse
    from deval.rewards.pipeline import RewardPipeline
    from dotenv import load_dotenv, find_dotenv
    
    task_name = TasksEnum.HALLUCINATION.value
    _ = load_dotenv(find_dotenv())

    allowed_models = ["gpt-4o-mini"]
    task_repo = TaskRepository(allowed_models=allowed_models)
    llm_pipeline = task_repo.get_random_llm()
 
    task = task_repo.create_task(llm_pipeline, task_name)
    agent = HumanAgent(task=task)
    print(f"Reference score: {agent.reference}")
    print(f"Reference Mistakes: {agent.reference_mistakes}")
    print(f"Reference True Values: {agent.reference_true_values}")

    # prep fake response
    uid = 1 
    responses = [
        BtEvalResponse(response = EvalResponse(uid = uid, score = 0.5, response_time = 1.5, mistakes = agent.reference_true_values), human_agent = agent),
        BtEvalResponse(response = EvalResponse(uid = uid, score = 1.0, response_time = 1.5, mistakes = []), human_agent = agent),
        BtEvalResponse(response = EvalResponse(uid = uid, score = 0.0, response_time = 1.5, mistakes = agent.reference_mistakes), human_agent = agent)
    ]

    # reward compute
    active_tasks = [TasksEnum.ATTRIBUTION.value,  TasksEnum.HALLUCINATION.value, TasksEnum.COMPLETENESS.value, TasksEnum.RELEVANCY.value]
    
    reward_pipeline = RewardPipeline(
        selected_tasks=active_tasks, device="cpu"
    )
    rewards = RewardResult(
        reward_pipeline,
        responses=responses,
        device="cpu",
    )

    print(rewards.__state_dict__())
