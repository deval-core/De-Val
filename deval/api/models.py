from pydantic import BaseModel
from enum import Enum

class APIStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"

class EvalRequest(BaseModel):
    tasks: list[str]
    rag_context: str
    query: str | None = ""
    llm_response: str

class EvalResponse(BaseModel):
    score: float 
    mistakes: list[str] | None
    response_time: float | None
    status_message: APIStatus | None = None
    

class ModelHashResponse(BaseModel):
    hash: str

class ModelColdkeyResponse(BaseModel):
    coldkey: str | None



