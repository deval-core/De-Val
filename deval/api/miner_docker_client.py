import requests
from deval.protocol import init_request_from_task
from deval.api.models import EvalRequest, EvalResponse
import time
import subprocess
import bittensor as bt
import os 

class MinerDockerClient:

    def __init__(self):
        self.service_name = "miner-api"
        self.host = "http://0.0.0.0" 
        self.port = 8000
        self.api_url = f"{self.host}:{self.port}"

    def start_service(self, model_dir: str):
        """Start the miner-api service using Docker Compose."""
        # ensure that our model_dir is accessible to the image during build time
        os.environ['MODEL_DIR'] = model_dir
        os.environ['COMPOSE_HTTP_TIMEOUT']='300'


        print(f"Starting {self.service_name} service...")
        subprocess.run(["docker-compose", "up", "--build", "-d", self.service_name], check=True)

        # Wait for the service to start
        time.sleep(25) 

    def stop_service(self):
        """Stop and clean up the miner-api service without affecting the validator service."""
        print(f"Stopping {self.service_name} service...")
        
        # Stop the miner-api service
        subprocess.run(["docker-compose", "stop", self.service_name], check=True)
        
        # Remove the container
        subprocess.run(["docker-compose", "rm", "-f", self.service_name], check=True)
        
        # Remove the Docker image
        self.remove_image()

    def remove_image(self):
        """Remove the Docker image associated with the miner-api service."""
        try:
            image_name = f"de-val-{self.service_name}:latest" 
            print(f"Removing Docker image: {image_name}...")
            subprocess.run(["docker", "rmi", image_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error removing image: {e}. It may not exist or is in use.")

    def query_eval(self, request: EvalRequest, timeout: int) -> EvalResponse:
        """Invoke the API running in the nested Docker container with queries."""
        bt.logging.info(f"Querying API on container {self.service_name}...")
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

        


if __name__ == "__main__":
    #TODO: update - but probably remove 
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


    