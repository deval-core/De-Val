# configuration for LLMs 
from pydantic import BaseModel
from enum import Enum

class LLMAPIs(Enum):
    OPENAI="openai"
    BEDROCK="aws_bedrock"


class LLMFormatType(Enum):
    JSON="json_object"
    TEXT="text"

class LLMArgs(BaseModel):
    max_new_tokens:int = 500
    temperature: float = 0.7
    top_p: float = 0.97
    format:LLMFormatType

SUPPORTED_MODELS = {
    LLMAPIs.OPENAI : ["gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4o-mini"],
    LLMAPIs.BEDROCK : ["anthropic.claude-3-haiku-20240307-v1:0", "cohere.command-r-plus-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0",  "mistral.mistral-small-2402-v1:0", "mistral.mistral-large-2402-v1:0"]
}