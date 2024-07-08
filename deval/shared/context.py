from typing import List
from dataclasses import dataclass
import random


@dataclass
class Context:
    # TODO: Pydantic model
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: List[str]
    external_links: List[str]
    source: str
    context_type: str | None = None
    difficulty: str = random.choice(['hard', 'medium', 'easy'])
    tags: List[str] = None
    extra: dict = None  # additional non-essential information
    stats: dict = None  # retrieval stats such as fetch time, number of tries, etc.
