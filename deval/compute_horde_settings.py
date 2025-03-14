import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

COMPUTE_HORDE_VALIDATOR_HOTKEY = os.environ.get("COMPUTE_HORDE_VALIDATOR_HOTKEY")
"""
Hotkey ss58 address of the ComputeHorde validator who will do the job.
Note: not the same as ``COMPUTE_HORDE_HOTKEY``, although will usually be the ss58 address of it.
"""

COMPUTE_HORDE_WALLET = os.environ.get("COMPUTE_HORDE_WALLET")
"""Wallet to sign ComputeHorde messages with. If left empty, the validator's wallet will be used."""

COMPUTE_HORDE_HOTKEY = os.environ.get("COMPUTE_HORDE_HOTKEY")
"""Hotkey to sign ComputeHorde messages with. If left empty, the validator's hotkey will be used."""

COMPUTE_HORDE_EXECUTOR_CLASS = os.environ.get("COMPUTE_HORDE_EXECUTOR_CLASS")
COMPUTE_HORDE_FACILITATOR_URL = os.environ.get("COMPUTE_HORDE_FACILITATOR_URL")
COMPUTE_HORDE_JOB_DOCKER_IMAGE = os.environ.get("COMPUTE_HORDE_JOB_DOCKER_IMAGE")
COMPUTE_HORDE_JOB_NAMESPACE = os.environ.get("COMPUTE_HORDE_JOB_NAMESPACE")
COMPUTE_HORDE_JOB_TIMEOUT = int(os.environ.get("COMPUTE_HORDE_JOB_TIMEOUT", 10 * 60))
COMPUTE_HORDE_JOB_MAX_ATTEMPTS = int(
    os.environ.get("COMPUTE_HORDE_JOB_MAX_ATTEMPTS", 3)
)

COMPUTE_HORDE_VOLUME_MINER_STATE_DIR = "/volume/miner_state"
COMPUTE_HORDE_VOLUME_MINER_STATE_FILENAME = "miner_state.pkl"
COMPUTE_HORDE_VOLUME_MINER_STATE_PATH = os.path.join(
    COMPUTE_HORDE_VOLUME_MINER_STATE_DIR, COMPUTE_HORDE_VOLUME_MINER_STATE_FILENAME
)
COMPUTE_HORDE_VOLUME_TASK_REPO_DIR = "/volume/task_repo"
COMPUTE_HORDE_VOLUME_TASK_REPO_FILENAME = "task_repo.pkl"
COMPUTE_HORDE_VOLUME_TASK_REPO_PATH = os.path.join(
    COMPUTE_HORDE_VOLUME_TASK_REPO_DIR, COMPUTE_HORDE_VOLUME_TASK_REPO_FILENAME
)
COMPUTE_HORDE_VOLUME_MODEL_PATH = "/volume/model"
COMPUTE_HORDE_ARTIFACTS_DIR = "/artifacts"
COMPUTE_HORDE_ARTIFACT_MODEL_STATE_PATH = os.path.join(
    COMPUTE_HORDE_ARTIFACTS_DIR, "miner_state.pkl"
)
COMPUTE_HORDE_ARTIFACT_MODEL_HASH_PATH = os.path.join(
    COMPUTE_HORDE_ARTIFACTS_DIR, "model_hash.txt"
)
COMPUTE_HORDE_ARTIFACT_MODEL_COLDKEY_PATH = os.path.join(
    COMPUTE_HORDE_ARTIFACTS_DIR, "model_coldkey.txt"
)
