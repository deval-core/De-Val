import torch
import bittensor as bt
from typing import List
import math

class DendriteResponseEvent:
    def __init__(
        self, responses: List[bt.Synapse], uids: torch.LongTensor, timeout: float
    ):
        self.uids = uids
        self.completions = []
        self.mistakes = []
        self.status_messages = []
        self.status_codes = []
        self.timings = []

        for synapse in responses:
            self.completions.append(synapse.completion)
            self.mistakes.append(synapse.mistakes)

            self.status_messages.append(synapse.dendrite.status_message)

            if math.isnan(synapse.completion) and synapse.dendrite.status_code == 200:
                synapse.dendrite.status_code = 204

            self.status_codes.append(synapse.dendrite.status_code)

            if (synapse.dendrite.process_time) and (
                synapse.dendrite.status_code == 200
                or synapse.dendrite.status_code == 204
            ):
                self.timings.append(synapse.dendrite.process_time)
            elif synapse.dendrite.status_code == 408:
                self.timings.append(timeout)
            else:
                self.timings.append(0)  # situation where miner is not alive

        self.completions = [synapse.completion for synapse in responses]
        self.timings = [
            synapse.dendrite.process_time or timeout for synapse in responses
        ]
        self.status_messages = [
            synapse.dendrite.status_message for synapse in responses
        ]
        self.status_codes = [synapse.dendrite.status_code for synapse in responses]

    def __state_dict__(self):
        return {
            "uids": self.uids.tolist(),
            "completions": self.completions,
            "mistakes": self.mistakes,
            "timings": self.timings,
            "status_messages": self.status_messages,
            "status_codes": self.status_codes,
        }

    def __repr__(self):
        return f"DendriteResponseEvent(uids={self.uids}, completions={self.completions}, mistakes={self.mistakes}, timings={self.timings}, status_messages={self.status_messages}, status_codes={self.status_codes})"
