import time
import bittensor as bt
from abc import ABC
from dataclasses import dataclass, asdict
from enum import Enum
from deval.llms.base_llm import BaseLLM
import json
from enum import Enum

class TasksEnum(Enum):
    HALLUCINATION = "hallucination"
    COMPLETENESS = "summary_completeness"
    ATTRIBUTION = "attribution"
    RELEVANCY = "relevancy"
    UNKNOWN = "unknown"


@dataclass
class Task(ABC):
    # topics: dict
    name: str
    desc: str
    goal: str
    topic: str
    subtopic: str
    tags: list[str]
    context: dict
    rag_context: str
    llm_response: str
    reference: float
    reference_mistakes: list[str]
    reference_true_values: list[str]
    reward_definition: list[dict]
    api: str
    model_id: str
    query: str = ""
    reward_threshold: float = 0.0
    penalty_definition: list[dict] = None
    joiners = ["\n", " ", "  ", "\t", "\n\n", "", "..."]
    complete: bool = False

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, desc={self.desc!r}, goal={self.goal!r}, rag_context={self.rag_context!r}, query={self.query}, topic={self.topic!r}, subtopic={self.subtopic!r}, tags={self.tags!r}, responses={self.llm_response!r}, reference={self.reference!r}, reference_mistakes={self.reference_mistakes}, reference_true_values={self.reference_true_values}, api={self.api!r}, model_id={self.model_id!r})"

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
            "api": self.api,
            "model_id": self.model_id,
        }
        if full:
            state.update(asdict(self.context))

        return state


    def generate_input(self, llm_pipeline: BaseLLM, prompt: str, system_prompt: str, tool_schema: dict) -> str:
        """Generates a query to be used for generating the challenge"""
        t0 = time.time()
        bt.logging.info("🤖 Generating query...")
        input = llm_pipeline.query(
            prompt=prompt,
            system_prompt=system_prompt,
            tool_schema = tool_schema
        )

        self.query_time = time.time() - t0
        return input

    def parse_llm_query(self, query) -> dict:
        json_query = json.loads(query)
        return json_query

    def format_challenge(self, challenge) -> str:
        """Formats the challenge to be used for the conversation"""
        return challenge
