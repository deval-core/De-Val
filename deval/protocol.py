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


from pydantic import BaseModel, Field, ConfigDict
import bittensor as bt

from typing import List
from deval.tasks.task import Task
import bittensor as bt
from typing import List
from deval.utils.misc import async_log
from deval.agent import HumanAgent



@async_log
async def execute_dendrite_call(dendrite_call):
    responses = await dendrite_call
    return responses


async def get_metadata_from_miner(validator, uid: int) -> list[bt.synapse]:
    axons = [validator.metagraph.axons[uid]]
    dendrite_call_task = execute_dendrite_call(validator.dendrite(axons=axons, synapse=ModelQuerySynapse, timeout=5))
    responses = await dendrite_call_task 

    return responses


class ModelQuerySynapse(bt.Synapse):
    """
    A simple synapse to query a miner for specific model metadata
    """

    class Config:
        """
        Pydantic model configuration class for EvalSynapse. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True

    def deserialize(self) -> "ModelQuerySynapse":
        """
        Returns the instance of the current ModelQuerySynapse object.

        This method is intended to be potentially overridden by subclasses for custom deserialization logic.
        In the context of the ModelQuerySynapse class, it simply returns the instance itself. However, for subclasses
        inheriting from this class, it might give a custom implementation for deserialization if need be.

        Returns:
            ModelQuerySynapse: The current instance of ModelQuerySynapse class.
        """
        return self


    repo_id: str = Field(
        ...,
        title="HuggingFace Repo ID",
        description="The miner's repo name for HuggingFace. Mutable",
        allow_mutation=True,
    )

    model_id: str = Field(
        ...,
        title="HuggingFace Model ID",
        description="The miner's model name for HuggingFace. Mutable",
        allow_mutation=True,
    )

    required_hash_fields: List[str] = Field(
        [],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

class EvalRequest(BaseModel):
    tasks: list[str]
    rag_context: str
    query: str | None = ""
    llm_response: str


    @staticmethod
    def init_from_task(task: Task) -> "EvalRequest":
        return EvalRequest(
            tasks = [task.name],
            rag_context = task.rag_context,
            query = task.query,
            llm_response = task.llm_response
        )

class DendriteModelQueryEvent:
    def __init__(
        self, responses: list[bt.synapse], 
    ):
        for synapse in responses:
            self.model_id = synapse.model_id
            self.repo_id = synapse.repo_id

            
class EvalResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float 
    mistakes: list[str] | None
    response_time: float
    uid: int | None = None
    human_agent: HumanAgent | None = None


