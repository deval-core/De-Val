import bittensor as bt
import os
from huggingface_hub import HfApi, snapshot_download
from deval.model.obfuscate import Obfuscator
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
    def pull_model_and_files(model_url: str) -> str:
        api = HfApi()
        hf_token = HuggingFaceModel.get_hf_token()

        download_dir = f"{os.get_env('HOME')}/eval_llm"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        bt.logging.info(f"Beggining the download of model data at {model_url}")
        files = api.list_repo_files(repo_id=model_url)

        # Download specific files manually
        for file in files:
            api.hf_hub_download(
                repo_id=model_url,
                filename=file,
                token = hf_token,
                local_dir = download_dir,
                repo_type = "model"
            )
        bt.logging.info(f"Downloaded model and files to {download_dir}")
        return download_dir

    @staticmethod
    def submit_model(
        model_dir: str, 
        pipeline_dir: str, 
        repo_id: str, 
        upload_model: bool = False,
        upload_pipeline: bool = True
    ):
        """
        Obfuscates the pipeline code and uploads the pipeline and model to your Huggingface repo

        Note this does not take advantage of HF version control. 
        """
        # Validate args
        if not upload_model and not upload_pipeline:
            raise ValueError(f"Either set upload_model or upload_pipeline to True")
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory '{model_dir}' does not exist.")
        
        if not os.path.exists(pipeline_dir):
            raise ValueError(f"Pipeline directory '{pipeline_dir}' does not exist.")
        
        hf_token = HuggingFaceModel.get_hf_token()

        # Check if the Hugging Face repository exists
        api = HfApi(token = hf_token)

        print(f"Checking if repository {repo_id} exists...")
        try:
            # This will raise an exception if the repository doesn't exist
            api.repo_info(repo_id=repo_id, token=hf_token)
            print(f"Repository '{repo_id}' already exists.")
        except:
            raise ValueError(f"repository at {repo_id} does not exist, please create.")


        # Run the obfuscator on the pipeline directory
        print("Running PyArmor obfuscation on pipeline directory...")
        Obfuscator.obfuscate(pipeline_dir)

        if upload_pipeline:
            print("Starting Pipeline upload")
            api.upload_folder(
                folder_path=os.path.join(pipeline_dir, '../obfuscated_pipeline'),
                repo_id=repo_id,
                repo_type="model"
            )
            print("Completed pipeline upload")
        
        if upload_model:
            print("Starting model upload")
            api.upload_large_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
            )
            print("Completed model upload")

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