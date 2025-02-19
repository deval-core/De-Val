import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from deval.api.models import EvalRequest, EvalResponse
import time

class HuggingFaceModel:

    @staticmethod
    def get_hf_token()-> str:
        hf_token = os.getenv('HUGGINGFACE_TOKEN', None)
        
        if not hf_token:
            _ = load_dotenv(find_dotenv())
            hf_token = os.getenv('HUGGINGFACE_TOKEN', None)
        
        if not hf_token:
            raise ValueError("Please provide a HuggingFace Token within you .env file at HUGGINGFACE_TOKEN")

        return hf_token

    @staticmethod
    def pull_model_and_files(model_url: str, download_dir: str | None = None) -> str:
        hf_token = HuggingFaceModel.get_hf_token()
        download_dir = download_dir or "/app/eval_llm"

        print(f"Beggining the download of model data at {model_url}")
        local_dir = snapshot_download(
            repo_id=model_url,
            repo_type="model",
            revision="main",
            token = hf_token,
            local_dir = download_dir
        )
        print(f"Downloaded model and files to {download_dir}")
        return download_dir

    @staticmethod
    def query_hf_model(pipe: pipeline, request: EvalRequest) -> EvalResponse:
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
