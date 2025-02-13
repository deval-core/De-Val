from pydantic import BaseModel
import bittensor as bt
import os
from deval.utils.constants import constants
from typing import Optional
import json


class ChainModelMetadataParsed(BaseModel):
    model_url: str | None
    model_hash: str | None
    block: int

class ChainModelMetadataStore:
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        wallet: Optional[bt.wallet] = None,
        subnet_uid: int = 202,
    ):
        self.subtensor = subtensor
        self.wallet = (
            wallet  # Wallet is only needed to write to the chain, not to read.
        )
        self.subnet_uid = subnet_uid

    def store_model_metadata(
        self, 
        model_url: str,
        model_hash: str
    ):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        
        commit_payload = {
            "model_url": model_url,
            "model_hash": model_hash,
        }
        commit_msg = json.dumps(commit_payload)

        self.subtensor.commit(self.wallet,self.subnet_uid,commit_msg)

    def retrieve_model_metadata(self, hotkey: str) -> ChainModelMetadataParsed:
        """Retrieves model metadata on this subnet for specific hotkey"""
        metadata = bt.core.extrinsics.serving.get_metadata(self.subtensor, self.subnet_uid, hotkey)

        if not metadata:
            return None
        else:
            return self.parse_chain_data(metadata)

    def parse_chain_data(self, metadata) -> ChainModelMetadataParsed:
        # decode the encoded string
        try:
            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            chain_str = bytes.fromhex(hex_data).decode()

            # format our output
            parsed_metadata = json.loads(chain_str)
        except:
            parsed_metadata = {"model_url": None, "model_hash": None}
        parsed_metadata['block'] = metadata["block"]
        return ChainModelMetadataParsed(**parsed_metadata)


# Can only commit data every ~20 minutes.
def test_store_model_metadata():
    """Verifies that the ChainModelMetadataStore can store data on the chain."""
    model_url = "deval-core/base-eval-test"
    hash = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    subtensor = bt.subtensor(network = 'test')

    # Uses .env configured wallet/hotkey/uid for the test.
    wallet_name = os.getenv("TEST_WALLET_NAME")
    hotkey_name = os.getenv("TEST_HOTKEY_NAME")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    metadata_store.store_model_metadata(model_url=model_url, model_hash=hash)

    print(f"Finished storing {model_url} on the chain.")


def test_retrieve_model_metadata():
    """Verifies that the ChainModelMetadataStore can retrieve data from the chain."""
    model_url = "deval-core/base-eval-test"

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor(network='test')

    # Uses .env configured hotkey/uid for the test.
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    hotkey_address = os.getenv("TEST_HOTKEY_ADDRESS")

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=None, subnet_uid=net_uid
    )

    # Retrieve the metadata from the chain.
    model_metadata = metadata_store.retrieve_model_metadata(hotkey_address)
    print(model_metadata)
    assert model_url == model_metadata.model_url
    assert model_metadata.model_hash is not None



if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    
    _ = load_dotenv(find_dotenv())
    # Can only commit data every ~20 minutes.
    #test_store_model_metadata()
    test_retrieve_model_metadata()