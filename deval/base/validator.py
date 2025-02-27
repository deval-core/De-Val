# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import copy
import torch
import asyncio
import argparse
import threading
import bittensor as bt
import numpy as np
import pytz

from traceback import print_exception

from deval.base.neuron import BaseNeuron
from deval.mock import MockDendrite
from deval.utils.config import add_validator_args
from deval.utils.exceptions import MaxRetryError
import pickle
import os
from datetime import datetime, timedelta


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        if not self.config.neuron.axon_off:
            bt.logging.info(
                f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
            )
        else:
            bt.logging.info(
                f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
            )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # TODO: update to 24 hours?
                forward_timeout = self.config.neuron.forward_max_time
                try:
                    task = self.loop.create_task(self.forward())
                    self.loop.run_until_complete(
                        asyncio.wait_for(task, timeout=forward_timeout)
                    )
                except torch.cuda.OutOfMemoryError as e:
                    bt.logging.error(f"Out of memory error: {e}")
                    continue
                except MaxRetryError as e:
                    bt.logging.error(f"MaxRetryError: {e}")
                    continue
                except asyncio.TimeoutError as e:
                    bt.logging.error(
                        f"Forward timeout: Task execution exceeded {forward_timeout} seconds and was cancelled.: {e}"
                    )
                    continue

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            if hasattr(self, "axon"):
                self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            sys.exit()

        # In case of unforeseen errors, the validator will log the error and quit
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
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

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.weights contains any NaN values and log a warning if it does.
        if np.isnan(self.weights).any():
            bt.logging.warning(
                "Weights contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        if not self.weights:
            bt.logging.warning(
                "Attempting to set weights, but weights are empty"
            )
            return 


        # split our uids and weights 
        final_weights = np.zeros(self.metagraph.n.item())
        for uid, weight in self.weights:
            final_weights[uid] = weight

        uids = np.indices(final_weights.shape)[0]

        bt.logging.info("raw_weights", final_weights)
        bt.logging.info("raw_weight_uids", uids)
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=uids,
            weights=final_weights,
            netuid=self.metagraph.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.info("processed_weights", processed_weights)
        bt.logging.info("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        #bt.logging.info("uint_weights", uint_weights)
        #bt.logging.info("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        num_attempts = 0
        result = False
        while result is False and num_attempts < 3:
            num_attempts += 1
            result, reason = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.metagraph.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )
            if result is True:
                bt.logging.info("set_weights on chain successfully!")
            else:
                bt.logging.error(f"set_weights failed with reason: {reason}")



    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        
    def save_state(self, save_weights = False):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        save_path = self.config.neuron.full_path 
        with open(os.path.join(save_path, "contest.pkl"), "wb") as f: 
            pickle.dump(self.contest, f)

        with open(os.path.join(save_path, "task_repo.pkl"), "wb") as f: 
            pickle.dump(self.task_repo, f)

        torch.save(
            {
                "start_over": self.start_over,
                "queried_uids": self.queried_uids,
                "hotkeys": self.hotkeys,
            },
            os.path.join(save_path, "state.pt"),
        )

        if save_weights:
            torch.save(
                {
                    "past_weights": self.weights,
                    "save_time": datetime.now()
                },
                os.path.join(save_path, "weights.pt"),
            )


    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        load_path = self.config.neuron.full_path 

        state_path = os.path.join(load_path, "state.pt")
        contest_path = os.path.join(load_path, "contest.pkl")
        task_repo_path = os.path.join(load_path, "task_repo.pkl")
        if (
            not os.path.exists(state_path) or 
            not os.path.exists(contest_path) or
            not os.path.exists(task_repo_path)
        ):
            bt.logging.info("one of the key components are unavailable. Skipping load")
            return None

        state = torch.load(state_path)
        self.start_over = state["start_over"]
        self.queried_uids = state["queried_uids"]
        self.hotkeys = state["hotkeys"]
        self.weights = state.get("past_weights", [])

        # load historical contest and task repository
        try:

            with open(contest_path, "rb") as f: 
                self.contest = pickle.load(f)

            # we put a time lock on how long a contest will take 
            max_time = 12
            now = datetime.now(tz=pytz.UTC)
            if now - timedelta(hours=max_time) <= self.contest.start_time_datetime:
                with open(task_repo_path, "rb") as f: 
                    self.task_repo = pickle.load(f)
            else:
                # if we exceed the max time then we opt to start
                self.start_over = True
                self.queried_uids = set()

        except Exception as e:
            bt.logging.warning(f"Unable to load the task repository or contest state, restarting contest: {e}")
            self.start_over = True

        # try loading in historical weights
        try:
            weight_path = os.path.join(load_path, "weights.pt")
            past_weights = torch.load(weight_path)
            weight_save_time = past_weights.get("save_time")
            if (datetime.now() - timedelta(hours=48)) <= weight_save_time:
                self.weights = past_weights.get("past_weights", [])
            else:
                self.weights = []
        
        except Exception as e:
            bt.logging.warning(f"Unable to load weights data with error: {e}")
            self.weights = []


    def reset(self):
        # self.weights = [] # no longer reset weights 
        self.task_repo = None
        self.queried_uids = set()
        
        # delete save states 
        load_path = self.config.neuron.full_path 
        files = os.listdir(load_path)
        for f in files:
            file_path = os.path.join(load_path, f)

            # we want to maintain weights data over epochs 
            if "weights.pt" in f:
                continue

            # otherwise we delete all save files 
            if os.path.isfile(file_path):
                os.remove(file_path)

    def get_uid_coldkey(self, uid: int) -> str:
        return self.metagraph.axons[uid].coldkey
