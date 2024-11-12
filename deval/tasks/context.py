from typing import List
from dataclasses import dataclass
import random


@dataclass
class Context:
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: list[str]
    external_links: list[str]
    source: str
    sections: list[str] | None = None
    context_type: str | None = None
    difficulty: str = random.choice(['hard', 'medium', 'easy'])
    tags: list[str] = None
    extra: dict = None  # additional non-essential information
    stats: dict = None  # retrieval stats such as fetch time, number of tries, etc.
