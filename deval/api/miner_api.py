from fastapi import FastAPI
import os
import time
from deval.api.models import EvalRequest, EvalResponse
import sys

app = FastAPI()

model_dir = os.getenv("MODEL_DIR")
sys.path.append("/app/eval_model") # matches to the location of the mounted directory
from model.pipeline import DeValPipeline


pipe = DeValPipeline("de_val", model_dir = model_dir)

@app.post("/eval_query")
async def query_model(request: EvalRequest) -> EvalResponse:
    """Process a user query through the miner's model."""
    start_time = time.time()
    tasks = request.tasks
    rag_context = request.rag_context
    query = request.query
    llm_response = request.llm_response

    completion = pipe("", tasks=tasks, rag_context=rag_context, query=query, llm_response=llm_response)
    print(f"Completion: {completion}")
    score = completion.get("score_completion", None)
    if not score:
        score = -1

    mistakes = completion.get("mistakes_completion", None)
    
    process_time = time.time() - start_time

    
    return EvalResponse(
        score = score,
        mistakes = mistakes,
        response_time = process_time,
    )

