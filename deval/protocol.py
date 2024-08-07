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



class EvalSynapse(bt.Synapse):
    # TODO: documentation 
    """
    The EvalSynapse subclass of the Synapse class encapsulates the functionalities related to evaluation scenarios.

    It specifies n fields - `tasks`, `rag_context`, `query`, `llm_response` and `completion` - that define the state of the EvalSynapse object.
    

    The Config inner class specifies that assignment validation should occur on this class (validate_assignment = True),
    meaning value assignments to the instance fields are checked against their defined types for correctness.

    Attributes:
        tasks (TaskEnum): The requested task to be completed by the miner. This field is both mandatory and immutable.
        rag_context (str): A simulated context in the evaluation scenario. This field is both mandatory and immutable.
        query (str): A simulated user query in a QA system. This field is optional and immutable.
        llm_response (str): A simulated response in the evaluation scenario. This field is both mandatory and immutable.
        completion (float): A float that captures completion of the evaluation. This field is mutable.
        required_hash_fields List[str]: A list of fields that are required for the hash.

    Methods:
        deserialize() -> "EvalSynapse": Returns the instance of the current object.


    The `EvalSynapse` class also overrides the `deserialize` method, returning the
    instance itself when this method is invoked. Additionally, it provides a `Config`
    inner class that enforces the validation of assignments (`validate_assignment = True`).
    """

    class Config:
        """
        Pydantic model configuration class for EvalSynapse. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True

    def deserialize(self) -> "EvalSynapse":
        """
        Returns the instance of the current EvalSynapse object.

        This method is intended to be potentially overridden by subclasses for custom deserialization logic.
        In the context of the EvalSynapse class, it simply returns the instance itself. However, for subclasses
        inheriting from this class, it might give a custom implementation for deserialization if need be.

        Returns:
            EvalSynapse: The current instance of the EvalSynapse class.
        """
        return self

    tasks: list[str] = pydantic.Field(
        ...,
        title="Tasks",
        description="A list of tasks in the EvalSynapse scenario. Immuatable.",
        allow_mutation=False,
    )

    rag_context: str = pydantic.Field(
        ...,
        title="Rag Context",
        description="Provided RAG context for responses in the EvalSynapse scenario. Immutable.",
        allow_mutation=False,
    )

    query: str = pydantic.Field(
        "",
        title="Query",
        description="A user generated query in the EvalSynapse scenario. Immutable.",
        allow_mutation=False,
    )

    llm_response: str = pydantic.Field(
        ...,
        title="LLM Response",
        description="LLM Responses generated from RAG context in the EvalSynapse scenario. Immutable.",
        allow_mutation=False,
    )

    completion: float = pydantic.Field(
        -1.0,
        title="Completion",
        description="Scored fields for each task. This attribute is mutable and can be updated.",
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["tasks", "rag_context", "llm_response"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


