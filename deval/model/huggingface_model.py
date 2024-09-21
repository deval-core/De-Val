from transformers import Pipeline
from deval.protocol import EvalSynapse
from deval.model.model_state import ModelState
from huggingface_hub import snapshot_download
import bittensor as bt

class HuggingFaceModel:

    @staticmethod
    def pull_model_and_files(miner_state: ModelState) -> str:
        local_dir = snapshot_download(repo_id=miner_state.get_model_url(), repo_type="model", revision="main")
        bt.logging.info(f"Downloading model and files to {local_dir}")
        return local_dir

    @staticmethod
    def submit_model():
        pass

    @staticmethod
    def run(eval_synapse: EvalSynapse, pipe: Pipeline):
        return pipe(eval_synapse)