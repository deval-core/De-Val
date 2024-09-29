import argparse
from deval.model.huggingface_model import HuggingFaceModel
from dotenv import load_dotenv, find_dotenv
import os




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dir",
        type=str,
        help="The path where your LLM is stored",
    )

    parser.add_argument(
        "--pipeline_dir",
        type=str,
        help="The path where your Huggingface pipeline is stored. Unobfuscated.",
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        help="The Huggingface repo to store your pipeline and model",
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        help="Your huggingface token. If null, we check your environment for HUGGINGFACE_TOKEN",
        default=None
    )

    parser.add_argument(
        "--upload_model",
        type='store_true',
        help="Boolean value on whether to submit a new version of the model. Defaults to False",
        default=False
    )

    parser.add_argument(
        "--upload_pipeline",
        type='store_true',
        help="Boolean value on whether to submit a new version of your custom pipeline folder. Defaults to True",
        default=True
    )


    # pull out args
    model_dir = parser.model_dir
    pipeline_dir = parser.pipeline_dir
    repo_id = parser.repo_id
    hf_token = parser.hf_token 

    if not hf_token:
        _ = load_dotenv(find_dotenv())
        hf_token = os.getenv('HUGGINGFACE_TOKEN', None)
        if not hf_token:
            raise ValueError("Please provide a HuggingFace Token")

    upload_model = parser.upload_model
    upload_pipeline = parser.upload_pipeline

    print("Beginning modle and pipeline submission process")
    HuggingFaceModel.submit_model(
        model_dir,
        pipeline_dir,
        repo_id,
        hf_token,
        upload_model,
        upload_pipeline
    )
    print("Completed submitting model.")








