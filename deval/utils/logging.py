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
from deval.utils.config import config as get_config, add_args

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)


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

class WandBLogger:
    
    # TODO: fix -  the storage of settings is janky
    def __init__(self, hotkey_address, netuid, active_tasks, config = None, force_off = False):
        self.hotkey_address = hotkey_address
        self.netuid = netuid
        self.active_tasks = active_tasks

        if not config:
            self.config = get_config(self)
        else:
            self.config = config

        if force_off:
            self.config.wandb.off = True

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)
    

    def export_logs(self, logs: List[Log]):
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

    def init_wandb(self, reinit=False):
        """Starts a new wandb run."""
        tags = [
            self.hotkey_address,
            deval.__version__,
            str(deval.__spec_version__),
            f"netuid_{self.netuid}",
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
            project="subnet",
            entity="deval-ai",
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
        self.init_wandb(reinit=True)


    def log_event(self, responses: list[BtEvalResponse], reward_result: RewardResult):
        reward_dict = reward_result.__state_dict__()
        for i, response in enumerate(responses):
            agent = response.human_agent

            event = {
                **agent.__state_dict__(),
                **response.__state_dict__(),
                **reward_dict[i],
            }
        
            if not self.config.neuron.dont_save_events:
                logger.log(38, event)

            if self.config.wandb.off:
                return

            if not getattr(self, "wandb", None):
                self.init_wandb()

            # Log the event to wandb.
            self.wandb.log(event)