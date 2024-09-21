from transformers import Pipeline
from huggingface_hub import snapshot_download
import bittensor as bt
from deval.protocol import EvalRequest

class HuggingFaceModel:

    @staticmethod
    def pull_model_and_files(repo_id: str) -> str:
        # TODO: define model path ourselves 
        # TODO: pass in the token somewhere 
        local_dir = snapshot_download(repo_id=repo_id, repo_type="model", revision="main")
        bt.logging.info(f"Downloading model and files to {local_dir}")
        return local_dir

    @staticmethod
    def submit_model():
        pass

    @staticmethod
    def run(request: EvalRequest, pipe: Pipeline):
        pass
        #return pipe()