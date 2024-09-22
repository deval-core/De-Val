import bittensor as bt
from deval.forward import forward
from deval.llms.openai_llm import OpenAILLM
from deval.base.validator import BaseValidatorNeuron
from deval.rewards.pipeline import RewardPipeline
from deval.task_generator import TaskGenerator
from dotenv import load_dotenv, find_dotenv
import os


class Validator(BaseValidatorNeuron):
    """
    Text prompt validator neuron.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # load all of our environment variables for easy access
        _ = load_dotenv(find_dotenv())

        # get allowed model ids and init generator
        allowed_models = self.config.neuron.model_ids.split(",")
        self.task_generator = TaskGenerator(allowed_models=allowed_models)

        # define how often tasks should run 
        if abs(1-sum(self.config.neuron.task_p)) > 0.001:
            raise ValueError("Task probabilities do not sum to 1.")

        # Filter out tasks with 0 probability
        self.active_tasks = [
            task
            for task, p in zip(self.config.neuron.tasks, self.config.neuron.task_p)
            if p > 0
        ]
        # Load the reward pipeline
        self.reward_pipeline = RewardPipeline(
            selected_tasks=self.active_tasks, device=self.device
        )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)

    def __enter__(self):
        if self.config.no_background_thread:
            bt.logging.warning("Running validator in main thread.")
            self.run()
        else:
            self.run_in_background_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")
