from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from enum import Enum


class LLMAPIs(Enum):
    OPENAI="openai"

class LLMFormatType(Enum):
    JSON="json_object"
    TEXT="text"

class LLMArgs(BaseModel):
    max_new_tokens:int = 500
    temperature: float = 0.7
    top_p: float = 0.97
    format:LLMFormatType


class BaseLLM(ABC):

    def __init__(
        self,
        api: LLMAPIs,
        model_id: str,
        model_kwargs: LLMArgs,
    ):
        self.api = api
        self.model_kwargs = model_kwargs.dict()
        self.model_id = model_id
        self.messages = []
        self.times = [0]


    @abstractmethod
    def query(
        self,
        prompt: str,
        system_prompt: str,
    ) -> str:
        ...


    @abstractmethod
    def forward(
        self, 
        messages: list[dict[str, str]]
    ) -> str:
        ...

    @abstractmethod
    def load(self, model_id: str):
        ...
    