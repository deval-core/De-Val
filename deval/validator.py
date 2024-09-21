import bittensor as bt
import time
from deval.base.validator import BaseValidatorNeuron
from deval.rewards import RewardPipeline, RewardResult
from deval.task_repository import TaskRepository
from dotenv import load_dotenv, find_dotenv
from deval.utils.uids import get_top_incentive_uids, get_candidate_uids
import os
from deval.model.model_state import ModelState
from deval.model.huggingface_model import HuggingFaceModel
from deval.contest import DeValContest
from deval.responses import get_metadata_from_miner, DendriteModelQueryEvent
from deval.agent import HumanAgent
from deval.protocol import EvalRequest
from deval.api.docker_client import DockerClient
from deval.tasks import Task



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

        self.num_uids_total = 256
        self.max_model_size_gbs = 20 # TODO: figure out actual number

        # get allowed model ids and init generator
        allowed_models = self.config.neuron.model_ids.split(",")
        self.task_repo = TaskRepository(allowed_models=allowed_models)

        # define how often tasks should run 
        if abs(1-sum(self.config.neuron.task_p)) > 0.001:
            raise ValueError("Task probabilities do not sum to 1.")

        # Filter out tasks with 0 probability
        self.task_sample_rate = [
            (task, int(self.config.neuron.num_task_examples * p))
            for task, p in zip(self.config.neuron.tasks, self.config.neuron.task_p)
            if p > 0
        ]
        # Load the reward pipeline
        active_tasks = [t[0] for t in self.task_sample_rate]
        self.reward_pipeline = RewardPipeline(
            selected_tasks=active_tasks, device=self.device
        )

    # TODO: these are only staticmethods to enable early testability while bringing similar components to same place
    # remove staticmethod once we have more mature testing suite
    # because we don't pass self, we have to pass a lot of variables around
    @staticmethod
    async def forward(self):
        """
        Validator forward pass. Consists of:
            * collect all UIDs
            * generate all task samples
            * iterate through all miners (run_epoch) 
                * pull model
                * verify not duplicate 
                * call run_step 
                    * run queries against model 
                    * store results 
                * update scores in model state
                * cleanup 
            * rank models 
            * update scores and set weights 
        """
        
        bt.logging.info("🚀 Starting forward loop...")
        forward_start_time = time.time()

        # init this rounds contest 
        contest = DeValContest(
            forward_start_time, 
            self.reward_pipeline, 
            self.config.neuron.timeout
        )

        # step 1: collect the top incentive uids
        top_incentive_uids = get_top_incentive_uids(self, k=self.miner_incentive_threshold, num_uids=self.num_uids_total).to(self.device)
        available_uids = get_candidate_uids(k = self.num_uids_total)

        # step 2: generate all tasks for miners to be evaluated on
        self.task_repo.generate_task_examples(task_probabilities=self.task_sample_rate)


        for uid in available_uids:

            # get the model metadata information from miner
            responses = get_metadata_from_miner(self, uid)
            response_event = DendriteModelQueryEvent(responses)
            bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}") 

            miner_state = ModelState(response_event.repo_id, response_event.model_id, uid)

            is_valid = miner_state.should_run_evaluation(
                uid, self.max_model_size_gbs, forward_start_time, top_incentive_uids
            )

            if is_valid:
                miner_state = Validator.run_epoch(
                    contest,
                    miner_state, 
                    self.task_repo, 
                )

            # update contest
            contest.update_model_state_with_rewards(miner_state) 

        # rank, select winners, and score
        contest.rank()
        self.update_scores(reward_result.rewards, uids)

        
    @staticmethod
    def run_epoch(
        contest: DeValContest, 
        miner_state: ModelState, 
        task_repo: TaskRepository, 
        
    ):
        #pull model, update contest, and validate model 
        model_dir = HuggingFaceModel.pull_model_and_files(miner_state)
        contest.add_new_model_state(miner_state)


        # TODO: Integration with new docker container occurs here? 
        docker_client = DockerClient()
        docker_client.init_miner_docker(model_dir)

        # run through all tasks
        for task in task_repo.get_all_tasks():
            miner_state = Validator.run_step(task, docker_client, miner_state, contest)


        # cleanup docker and model 
        miner_state.cleanup_all() 
        
        return miner_state

    @staticmethod
    def run_step(
        task: Task, 
        docker_client: DockerClient,
        miner_state: ModelState,
        contest: DeValContest
    ):
        # query docker container with task 
        agent = HumanAgent(
            task=task
        )
        request = EvalRequest.init_from_task(task)
        response = docker_client.invoke(request, miner_state.uid)
        
        # generate and store reward  
        reward_result = RewardResult(
            contest.reward_pipeline,
            agent=agent,
            response_event=response,
            device="cpu" # self.device,
        )
        
        miner_state.add_reward(reward_result)

        # TODO: add logging for specific event here 

        return miner_state




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
