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
import bittensor as bt
from deval.tasks.task import Task
#from deval.cleaners.cleaner import CleanerPipeline


class HumanAgent:
    "Agent that impersonates a human user and makes queries based on its goal."

    @property
    def progress(self):
        return int(self.task.complete)

    @property
    def finished(self):
        return self.progress == 1


    def __init__(
        self,
        task: Task,
    ):

        self.task = task
        self.store_challenge()
        self.top_response = -1.0

   
    def store_challenge(self) -> str:
        "Store relevant information needed to generate the synapse for miners"
        self.tasks_challenge = self.task.name
        self.rag_context = self.task.rag_context
        self.llm_response = self.task.llm_response
        self.query = self.task.query
        self.reference = self.task.reference
        self.reference_mistakes = self.task.reference_mistakes
        self.reference_true_values = self.task.reference_true_values
    

    def __state_dict__(self, full=False):
        return {
            "tasks": self.tasks_challenge,
            "rag_context": self.rag_context,
            "query": self.query,
            "llm_response": self.llm_response,
            #"reference": self.reference,
            #"reference_mistakes": self.reference_mistakes,
            #"reference_true_values": self.reference_true_values,
            **self.task.__state_dict__(full=full),
        }

    def __repr__(self):
        return str(self)

    
    def update_progress(
        self, top_reward: float, top_response: str
    ):
        if top_reward > self.task.reward_threshold:
            self.task.complete = True
            self.top_response = top_response

            bt.logging.info("Agent finished its goal")
            return

        