from fastapi import FastAPI
import os
from deval.model.huggingface_model import HuggingFaceModel
from deval.protocol import EvalRequest, EvalResponse
import sys

app = FastAPI()

repo_id = os.getenv("REPO_ID")
model_dir = os.getenv("MODEL_DIR")
sys.path.append(model_dir)
from model.pipeline import DeValPipeline


def download_model_and_pipeline():
    """Download model and pipeline from HuggingFace."""
    
    # TODO: enable downloading to specific model directory
    #model_dir = HuggingFaceModel.pull_model_and_files(repo_id)
    model_dir = "../model"
    print(f"Model and pipeline downloaded to {model_dir}")

    #TODO: Update to obfuscated pipeline
    return DeValPipeline("de_val", model_dir = model_dir)

pipe = download_model_and_pipeline()


@app.post("/eval_query")
async def query_model(request: EvalRequest) -> EvalResponse:
    """Process a user query through the miner's model."""
    return HuggingFaceModel.query_hf_model(pipe, request)
    
