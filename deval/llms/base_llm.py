from abc import ABC, abstractmethod
from deval.llms.llm_config import LLMAPIs, LLMArgs


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
        tool_schema: dict | None = None
    ) -> str:
        ...


    @abstractmethod
    def forward(
        self, 
        messages: list[dict[str, str]]
    ) -> str:
        ...

    @abstractmethod
    def parse_response(
        self, 
        output
    ) -> str:
        # types unknown and dependent on input API 
        ...

    @abstractmethod
    def load(self):
        ...
    