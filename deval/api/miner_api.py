from fastapi import FastAPI
import os
import time
from deval.api.models import EvalRequest, EvalResponse, ModelHashResponse, APIStatus, ModelColdkeyResponse
from deval.model.huggingface_model import HuggingFaceModel
import sys
import hashlib
from deval.model.utils import compute_model_hash

app = FastAPI()

model_dir = "/app/eval_llm"
sys.path.append(model_dir) # matches to the location of the mounted directory
model_url = os.getenv("MODEL_URL", "")
model_volume_dir = os.getenv("MODEL_VOLUME_DIR", "")

if model_url != "" or model_volume_dir:
    if model_url:
        model_dir = HuggingFaceModel.pull_model_and_files(model_url)
    else:
        print("Loading model from volume dir")
        model_dir = model_volume_dir
        sys.path.append(model_dir)

    from model.pipeline import DeValPipeline
    pipe = DeValPipeline("de_val", model_dir = model_dir)
    print("SUCCESFULLY LOADED PIPELINE")



@app.post("/eval_query")
async def query_model(request: EvalRequest) -> EvalResponse:
    """Process a user query through the miner's model."""
    start_time = time.time()
    tasks = request.tasks
    rag_context = request.rag_context
    query = request.query
    llm_response = request.llm_response

    completion = pipe("", tasks=tasks, rag_context=rag_context, query=query, llm_response=llm_response)
    process_time = time.time() - start_time
    try:
        print(f"Completion: {completion}")
        score = completion.get("score_completion", None)
        if score is None:
            score = -1

        mistakes = completion.get("mistakes_completion", None)

        
        return EvalResponse(
            score = score,
            mistakes = mistakes,
            response_time = process_time,
            status_message = APIStatus.SUCCESS
        )
    except Exception as e:
        print(f"Failed with error: {e}")
        return EvalResponse(
            score = -1.0,
            mistakes = [],
            response_time = process_time,
            status_message = APIStatus.ERROR
        )
        


@app.get("/get_model_hash")
async def get_model_hash()-> ModelHashResponse:
    hash_value = compute_model_hash(model_dir)
    print(f"Hash of model: {hash_value}")
    return ModelHashResponse(hash =hash_value)


@app.get("/get_model_coldkey")
async def get_model_coldkey() -> ModelColdkeyResponse:
    print("pulling coldkey")
    try:
        coldkey = open(os.path.join(model_dir, "coldkey.txt"), "r").read()
    except:
        coldkey = None
    
    print(f"Returning coldkey as: {coldkey}")
    return ModelColdkeyResponse(
        coldkey=coldkey
    )

@app.get("/health")
async def health()->bool:
    return True
