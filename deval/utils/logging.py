import json
import os
import copy
import wandb
import bittensor as bt
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List
import logging
import deval
from deval.protocol import BtEvalResponse
from deval.rewards.reward import RewardResult
from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME
from pydantic import BaseModel
from deval.utils.config import config

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)

# TODO: this ishacky  - turn into class
class WandBConfig(BaseModel):
    off: bool
    run_step_length: int
    dont_save_events: bool


    def __init__(config):
        return WandBConfig(
            off = config.wandb.off,
            run_step_length = config.wandb.run_step_length,
            dont_save_events = config.neuron.dont_save_events
        )
    

#wandb_config = WandBConfig(config(None))

@dataclass
class Log:
    validator_model_id: str
    challenge: str
    challenge_prompt: str
    reference: str
    miners_ids: List[str]
    responses: List[str]
    miners_time: List[float]
    challenge_time: float
    reference_time: float
    rewards: List[float]
    task: dict
    # extra_info: dict


def export_logs(logs: List[Log]):
    bt.logging.info("📝 Exporting logs...")

    # Create logs folder if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Get the current date and time for logging purposes
    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M")

    all_logs_dict = [asdict(log) for log in logs]

    for logs in all_logs_dict:
        task_dict = logs.pop("task")
        prefixed_task_dict = {f"task_{k}": v for k, v in task_dict.items()}
        logs.update(prefixed_task_dict)

    log_file = f"./logs/{date_string}_output.json"
    with open(log_file, "w") as file:
        json.dump(all_logs_dict, file)

    return log_file


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return (
        not self.config.wandb.off
        and self.step
        and self.step % self.config.wandb.run_step_length == 0
    )


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    tags = [
        self.wallet.hotkey.ss58_address,
        deval.__version__,
        str(deval.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
    ]

    if self.config.mock:
        tags.append("mock")
    for task in self.active_tasks:
        tags.append(task)
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=self.config.wandb.project_name,
        entity=self.config.wandb.entity,
        config=wandb_config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
    )
    bt.logging.success(f"Started a new wandb run <blue> {self.wandb.name} </blue>")


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def log_event(self, responses: list[BtEvalResponse], reward_result: RewardResult):
    reward_dict = reward_result.__state_dict__()
    for i, response in enumerate(responses):
        agent = response.human_agent

        event = {
            **agent.__state_dict__(),
            **response.__state_dict__(),
            **reward_dict[i],
        }
    
        if not wandb_config.dont_save_events:
            logger.log(38, event)

        if wandb_config.off:
            return

        if not getattr(self, "wandb", None):
            init_wandb(self)

        # Log the event to wandb.
        self.wandb.log(event)