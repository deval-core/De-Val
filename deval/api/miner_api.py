from fastapi import FastAPI
import os
import shutil
from deval.model.huggingface_model import HuggingFaceModel
from deval.protocol import EvalRequest, EvalResponse
import time

app = FastAPI()

repo_id = os.getenv("REPO_ID")
model_dir = os.getenv("MODEL_DIR")

def download_model_and_pipeline(repo_id, model_dir):
    """Download model and pipeline from HuggingFace."""
    
    # TODO: enable downloading to specific model directory
    model_dir = HuggingFaceModel.pull_model_and_files(repo_id)
    return model_dir

@app.on_event("startup")
async def startup_event():
    """Download model and pipeline at startup."""
    print(f"Downloading model and pipeline for {repo_id}...")
    model_dir = download_model_and_pipeline(repo_id)
    print(f"Model and pipeline downloaded to {model_dir}")

@app.post("/eval_query")
async def query_model(request: EvalRequest) -> EvalResponse:
    """Process a user query through the miner's model."""
    
    start_time = time.time()
    # TODO: replace with pipe through custom pipeline pointing at model directory 
    completion = -1
    process_time = time.time() - start_time


    # Dummy response - real implementation would call the participant's model and pipeline
    
    return EvalResponse(
        completion = completion,
        response_time = process_time
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up model files after the container shuts down."""
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    if os.path.exists(pipeline_path):
        shutil.rmtree(pipeline_path)
    print("Model and pipeline files cleaned up.")
