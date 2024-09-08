# supports tool use generation across all supported APIs
from deval.llms.base_llm import BaseLLM, LLMAPIs

class ToolSchemaGenerator:

    def __init__(
        self,
        name: str, 
        description: str, 
        properties: dict, 
        required: list[str]
    ):
        self.name = name
        self.description = description
        self.properties = properties
        self.required = required

    def get_schema(self, llm_pipeline: BaseLLM) -> dict:
        api = llm_pipeline.api

        if api == LLMAPIs.OPENAI:
            return self.generate_openai_schema()
        
        else:
            raise ValueError(f"Unsupported API for tool use {api} - please implement a schema for this API")


    def generate_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                    "additionalProperties": False,
                },
            }
        }