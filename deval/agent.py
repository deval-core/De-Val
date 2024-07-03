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
from deval.tasks import Task
#from deval.cleaners.cleaner import CleanerPipeline
from pydantic import BaseModel


class HumanAgent(BaseModel):
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

   
    def store_challenge(self) -> str:
        "Store relevant information needed to generate the synapse for miners"
        self.tasks_challenge = self.task.name
        self.context_input = self.task.query
        self.response = self.task.reference
    

    def __state_dict__(self, full=False):
        return {
            "tasks": self.tasks_challenge,
            "context_input": self.context_input,
            "response": self.response,
            **self.task.__state_dict__(full=full),
        }

    def __repr__(self):
        return str(self)

    
    def update_progress(
        self, top_reward: float, top_response: str
    ):
        if top_reward > self.task.reward_threshold:
            self.task.complete = True
            self.messages.append({"content": top_response, "role": "user"})

            bt.logging.info("Agent finished its goal")
            return

        