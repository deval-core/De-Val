from fastapi import FastAPI
import os
from neurons.miners.pipeline import DeValPipeline
from deval.model.huggingface_model import HuggingFaceModel
from deval.protocol import EvalRequest
import time
from deval.protocol import EvalResponse

app = FastAPI()

repo_id = os.getenv("REPO_ID")
model_dir = os.getenv("MODEL_DIR")


def download_model_and_pipeline():
    """Download model and pipeline from HuggingFace."""
    
    # TODO: enable downloading to specific model directory
    #model_dir = HuggingFaceModel.pull_model_and_files(repo_id)
    model_dir = "../model"
    print(f"Model and pipeline downloaded to {model_dir}")

    return DeValPipeline("de_val", model_dir = model_dir)

pipe = download_model_and_pipeline()


@app.post("/eval_query")
async def query_model(request: EvalRequest) -> EvalResponse:
    """Process a user query through the miner's model."""
    
    start_time = time.time()
    tasks = request.tasks
    rag_context = request.rag_context
    query = request.query
    llm_response = request.llm_response

    completion = pipe("", tasks=tasks, rag_context=rag_context, query=query, llm_response=llm_response)
    score = completion.get("score_completion")
    mistakes = completion.get("mistakes_completion")
    
    process_time = time.time() - start_time

    
    return EvalResponse(
        score = score,
        mistakes = mistakes,
        response_time = process_time,
    )
