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

import pydantic
import bittensor as bt

from typing import List
from deval.tasks import Task


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


    repo_id: str = pydantic.Field(
        ...,
        title="HuggingFace Repo ID",
        description="The miner's repo name for HuggingFace. Mutable",
        allow_mutation=True,
    )

    model_id: str = pydantic.Field(
        ...,
        title="HuggingFace Model ID",
        description="The miner's model name for HuggingFace. Mutable",
        allow_mutation=True,
    )

    required_hash_fields: List[str] = pydantic.Field(
        [],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

class EvalResponse(pydantic.BaseModel):
    completion: float 
    response_time: int
    uid: int | None

class EvalRequest(pydantic.BaseModel):
    tasks: list[str]
    rag_context: str
    query: str | None = ""
    llm_response: str


    @staticmethod
    def init_from_task(self, task: Task) -> "EvalRequest":
        return EvalRequest(
            tasks = [task.name],
            rag_context = task.rag_context,
            query = task.query,
            llm_response = task.llm_response
        )


