import time
import bittensor as bt
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Union, Dict
from deval.llms import OpenAILLM as ValidatorLLM, BasePipeline
#from deval.cleaners.cleaner import CleanerPipeline
import json
from enum import Enum

class TasksEnum(Enum):
    HALLUCINATION = "hallucination"
    COMPLETENESS = "summary completeness"
    UNKNOWN = "unknown"


@dataclass
class Task(ABC):
    # topics: dict
    name: str
    desc: str
    goal: str
    query: str
    topic: str
    subtopic: str
    tags: List[str]
    context: dict
    response: float
    reward_definition: List[dict]
    penalty_definition: List[dict] = None
    reward_threshold: float = 0.0
    reference: Union[str, List[str]] = ""
    complete: bool = False
    cleaner = None
    clean_reference = False

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, desc={self.desc!r}, goal={self.goal!r}, query={self.query!r}, reference={self.reference!r}, topic={self.topic!r}, subtopic={self.subtopic!r}, tags={self.tags!r}, responses={self.responses!r}, reference={self.reference!r})"

    def __repr__(self):
        return str(self)

    def __state_dict__(self, full=False):
        state = {
            "task": self.name,
            "desc": self.desc,
            "goal": self.goal,
            "query": self.query,  # For now we just use the raw query but should add delimiters again
            "query_time": getattr(self, "query_time", 0),
            "reference": self.reference,
            "reference_time": getattr(self, "reference_time", 0),
            "topic": self.topic,
            "subtopic": self.subtopic,
            "context_time": self.context.stats.get("fetch_time", 0.0),
        }
        if full:
            state.update(asdict(self.context))

        return state
 
    def generate(
        self, system: str, prompt: str, pipeline: BasePipeline, clean=True
    ) -> str:
        """Uses the llm to generate a response to a prompt"""

        #cleaner = (
        #    CleanerPipeline(cleaning_pipeline=self.cleaning_pipeline) if clean else None
        #)

        # TODO: find a better place to define temperature
        return ValidatorLLM(pipeline, system_prompt=system, temperature=1).query(
            prompt=prompt
        )

    def generate_query(self, pipeline: BasePipeline, prompt, system_prompt) -> str:
        """Generates a query to be used for generating the challenge"""
        t0 = time.time()
        bt.logging.info("🤖 Generating query...")
        self.query = self.generate(
            system=system_prompt, 
            prompt=prompt,
            pipeline=pipeline,
            clean=False,
        )

        self.query_time = time.time() - t0
        return self.query

    def parse_llm_query(self, query) -> dict:
        json_query = json.loads(query)
        return json_query

    def format_challenge(self, challenge) -> str:
        """Formats the challenge to be used for the conversation"""
        return challenge
