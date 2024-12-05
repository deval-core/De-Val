import argparse
from deval.model.huggingface_model import HuggingFaceModel
import os
import bittensor as bt
from deval.model.chain_metadata import ChainModelMetadataStore
from deval.model.utils import compute_model_hash
from deval.model.obfuscate import Obfuscator
from huggingface_hub import HfApi

def submit_model(
    model_dir: str, 
    pipeline_dir: str, 
    repo_id: str, 
    wallet_name: str,
    hotkey_name: str,
    upload_model: bool = False,
    upload_pipeline: bool = True,
    test_network: bool = False,
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
    
    if upload_pipeline:
        print("Running PyArmor obfuscation on pipeline directory...")
        Obfuscator.obfuscate(pipeline_dir)

        print("Starting Pipeline upload")
        api.upload_folder(
            folder_path=os.path.join(pipeline_dir, '../obfuscated_pipeline'),
            repo_id=repo_id,
            repo_type="model"
        )
        print("Completed pipeline upload")
    
    if upload_model:
        print("Compute Hash")
        model_hash = compute_model_hash(model_dir)

        print("Generating on chain commit")
        if test_network:
            netuid = 202
            subtensor = bt.subtensor(network = 'test')
        else:
            netuid = 15
            subtensor = bt.subtensor()

        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)

        metadata_store = ChainModelMetadataStore(
            subtensor=subtensor, wallet=wallet, subnet_uid=netuid
        )

        metadata_store.store_model_metadata(model_url=repo_id, model_hash=model_hash)

        print("Starting model upload")
        api.upload_large_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print("Completed model upload")


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
        "--wallet_name",
        type=str,
        help="The name of the wallet associated with your miner",
    )

    parser.add_argument(
        "--hotkey_name",
        type=str,
        help="The name of the hotkey associated with your miner",
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

    parser.add_argument(
        "--test",
        action='store_true',
        help="Boolean value on whether to submit to test network",
        default=False
    )

    args = parser.parse_args()

    # pull out args
    model_dir = args.model_dir
    pipeline_dir = args.pipeline_dir
    repo_id = args.repo_id
    wallet_name = args.wallet_name
    hotkey_name = args.hotkey_name
    
    hf_token = args.hf_token 
    if not hf_token:
        hf_token = HuggingFaceModel.get_hf_token()
       

    upload_model = args.upload_model
    upload_pipeline = args.upload_pipeline
    print("Upload pipeline: ", upload_pipeline)

    test_network = args.test

    print("Beginning modle and pipeline submission process")
    submit_model(
        model_dir,
        pipeline_dir,
        repo_id,
        wallet_name,
        hotkey_name,
        upload_model,
        upload_pipeline,
        test_network=test_network,
    )
    print("Completed submitting model.")








