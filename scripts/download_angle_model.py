import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from deval.model.huggingface_model import HuggingFaceModel  # noqa

ANGLE_MODEL_PATH = os.environ["ANGLE_MODEL_PATH"]

HuggingFaceModel.pull_model_and_files("WhereIsAI/UAE-Large-V1", ANGLE_MODEL_PATH)
