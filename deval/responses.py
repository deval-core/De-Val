import torch
import bittensor as bt
from typing import List
import math
from deval.utils.misc import async_log
from deval.protocol import ModelQuerySynapse, EvalResponse



@async_log
async def execute_dendrite_call(dendrite_call):
    responses = await dendrite_call
    return responses


async def get_metadata_from_miner(validator, uid: int) -> list[bt.synapse]:
    axons = [validator.metagraph.axons[uid]]
    dendrite_call_task = execute_dendrite_call(validator.dendrite(axons=axons, synapse=ModelQuerySynapse, timeout=5))
    responses = await dendrite_call_task 

    return responses

class DendriteModelQueryEvent:
    def __init__(
        self, responses: list[bt.synapse], 
    ):
        for synapse in responses:
            self.model_id = synapse.model_id
            self.repo_id = synapse.repo_id

            

