import hashlib
import os

def compute_model_hash(model_dir: str):
    print("Computing Hash of model")
    sha256_hash = hashlib.sha256()
    safetensor_files =  [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    safetensor_files = sorted(safetensor_files)

    # Open the file in binary mode
    for model_path in safetensor_files:
        with open(os.path.join(model_dir, model_path), "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

    # Return the hex digest of the file
    hash_value = sha256_hash.hexdigest()
    print(f"Hash of model: {hash_value}")
    return hash_value