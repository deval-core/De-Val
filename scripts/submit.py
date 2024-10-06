import argparse
from deval.model.huggingface_model import HuggingFaceModel
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
        action='store_true',
        help="Boolean value on whether to submit a new version of the model. Defaults to False",
        default=False
    )

    parser.add_argument(
        "--upload_pipeline",
        action='store_true',
        help="Boolean value on whether to submit a new version of your custom pipeline folder. Defaults to True",
        default=False
    )

    args = parser.parse_args()

    # pull out args
    model_dir = args.model_dir
    pipeline_dir = args.pipeline_dir
    repo_id = args.repo_id
    hf_token = args.hf_token 

    if not hf_token:
        hf_token = HuggingFaceModel.get_hf_token()
       

    upload_model = args.upload_model
    upload_pipeline = args.upload_pipeline
    print("Upload pipeline: ", upload_pipeline)

    print("Beginning modle and pipeline submission process")
    HuggingFaceModel.submit_model(
        model_dir,
        pipeline_dir,
        repo_id,
        upload_model,
        upload_pipeline
    )
    print("Completed submitting model.")








