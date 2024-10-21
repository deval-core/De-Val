from pydantic import BaseModel

class EvalRequest(BaseModel):
    tasks: list[str]
    rag_context: str
    query: str | None = ""
    llm_response: str

class EvalResponse(BaseModel):
    score: float 
    mistakes: list[str] | None
    response_time: float
    

class ModelHashResponse(BaseModel):
    hash: str
    




