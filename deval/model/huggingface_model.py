from transformers import Pipeline
import bittensor as bt
from deval.protocol import EvalRequest
import os
from huggingface_hub import HfApi, HfFolder, Repository, snapshot_download
from deval.model.obfuscate import Obfuscator

class HuggingFaceModel:

    @staticmethod
    def pull_model_and_files(repo_id: str) -> str:
        # TODO: pass in the token somewhere 
        local_dir = snapshot_download(repo_id=repo_id, repo_type="model", revision="main")
        bt.logging.info(f"Downloading model and files to {local_dir}")
        return local_dir

    @staticmethod
    def submit_model(
        model_dir: str, 
        pipeline_dir: str, 
        repo_id: str, 
        token: str = None,
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
        
        if token is None:
            raise ValueError(f"Pass in a valid HuggingFace Token")

        # Step 2: Check if the Hugging Face repository exists
        api = HfApi()

        print(f"Checking if repository {repo_id} exists...")
        try:
            # This will raise an exception if the repository doesn't exist
            api.repo_info(repo_id=repo_id, token=token)
            print(f"Repository '{repo_id}' already exists.")
        except:
            raise ValueError(f"repository at {repo_id} does not exist, please create.")


        # Step 1: Run the obfuscator on the pipeline directory
        print("Running PyArmor obfuscation on pipeline directory...")
        Obfuscator.obfuscate(pipeline_dir)

        if upload_pipeline:
            print("Starting Pipeline upload")
            api.upload_folder(
                folder_path=pipeline_dir,
                repo_id=repo_id,
                repo_type="model",
                token= token
            )
            print("Completed pipeline upload")
        
        if upload_model:
            print("Starting model upload")
            api.upload_large_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                token= token
            )
            print("Completed model upload")
