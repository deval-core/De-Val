#TODO delete - only keeping in case new approach does not work 
import requests
from deval.protocol import init_request_from_task
from deval.api.models import EvalRequest, EvalResponse
import time
import docker
import bittensor as bt

class MinerDockerClient:

    def __init__(self):
        self.host = "http://0.0.0.0" 
        self.port = 8000
        self.api_url = f"{self.host}:{self.port}"
        self.dockerfile_dir = "deval/api"

        self.model_dir = model_dir

        self.container_name = "miner-api"
        self.docker_client = docker.from_env()

    def build_image(self, force_rebuild=False):
        """
        Build the Docker image from the Dockerfile.
        :param force_rebuild: If True, rebuild the image even if it already exists.
        """
        try:
            # Check if the image already exists
            existing_image = self.client.images.get(self.container_name)
            if existing_image and not force_rebuild:
                bt.logging.warning(f"Image '{self.container_name}' already exists. Skipping build.")
                return None
        except docker.errors.ImageNotFound:
            bt.logging.info(f"Image '{self.container_name}' not found. Building it now...")

        # first stop and remove any other ones running
        self.cleanup()

        # Build the Docker image
        print(f"Building Docker image from {self.dockerfile_dir}...")
        self.client.images.build(path=self.dockerfile_dir, tag=self.container_name)
        print(f"Docker image '{self.container_name}' built successfully!")

    def run_miner_docker(self, model_dir: str) -> None:
        """Initialize the nested Docker container with mounted model and pipeline."""
        bt.logging.info(f"Starting nested Docker container: {self.container_name}...")
        
        # Create the container
        container = self.docker_client.containers.run(
            image="nested-docker-image:latest",  
            name=self.container_name,
            detach=True,
            environment={
                'MODEL_DIR': model_dir
            },
            volumes={
                model_dir: {'bind': '/app/eval_model', 'mode': 'ro'},  # Read-only mount for the model directory
            },
            network_disabled=True,  # Ensure no network access
            user="miner",  # Non-root user
            security_opt=['no-new-privileges'],  
            read_only=True  # Make filesystem read-only
        )
        
        # TODO: may need a poll here
        time.sleep(5)
        return container 

    def query_eval(self, request: EvalRequest, timeout: int) -> EvalResponse:
        """Invoke the API running in the nested Docker container with queries."""
        bt.logging.info(f"Querying API on container {self.container_name}...")
        try:
            response = requests.post(
                f"{self.api_url}/query_eval",
                json=request,
                timeout=timeout
            )
            resp = response.json()
            return EvalResponse(**resp)

        except Exception as e:
            bt.logging.error(f"Failed to query API: {e}")
            return EvalResponse(
                score = -1, 
                mistakes = [],
                response_time = timeout + 1
            )

    def cleanup(self):
        """Stop and remove the container, and delete the model directory."""
        bt.logging.info(f"Cleaning up container {self.container_name} and model files...")
        try:
            container = self.docker_client.containers.get(self.container_name)
            container.stop()
            container.remove()
            bt.logging.info(f"Container {self.container_name} removed successfully.")
        except docker.errors.NotFound:
            bt.logging.info(f"Container {self.container_name} not found.")
        
        


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    from deval.task_repository import TaskRepository
    from deval.tasks.task import TasksEnum
    
    model_dir = "../model"
    _ = load_dotenv(find_dotenv())
    task_name = TasksEnum.HALLUCINATION.value

    # create our docker client
    docker_client = MinerDockerClient(model_dir)
    docker_client.init_miner_docker()

    # initialize a task 
    allowed_models = ["gpt-4o-mini"]
    task_repo = TaskRepository(allowed_models=allowed_models)
    llm_pipeline = task_repo.get_random_llm()
 
    task = task_repo.create_task(llm_pipeline, task_name)
    request = init_request_from_task(task)

    # query the client and cleanup
    response = docker_client.query_eval(request, 10)
    docker_client.cleanup()


    