import requests
from deval.protocol import init_request_from_task
from deval.api.models import EvalRequest, EvalResponse, APIStatus
import time
import subprocess
import bittensor as bt
import os
from requests.exceptions import Timeout
import json

class MinerDockerClient:

    def __init__(self, api_url: str = "http://0.0.0.0:8000"):
        self.service_name = "miner-api"
        self.api_url = api_url

    def _poll_service_for_readiness(self, max_wait_time: int) -> bool:
        #TODO: check for errors to stop polling when we know we failed 
        num_checks = 50
        sleep_interval = int(max_wait_time/num_checks)

        for i in range(num_checks):
            time.sleep(sleep_interval)
            try:
                response = requests.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    bt.logging.info("Successful connection to miner-api...")
                    return True

            except requests.ConnectionError as e:
                # Only log a warning every 10th check (or any interval you want)
                if i % 10 == 0:
                    bt.logging.warning(f"Unable to connect to miner api (attempt {i+1}/{num_checks}): {e}")

        # If all checks = no success:
        bt.logging.error("Failed to connect to miner-api after all attempts.")
        return False

    def _is_container_running(self):
        # Checks if a Docker container is running.
        try:
            result = subprocess.run(
                ['docker-compose', 'ps', '-q', self.service_name],
                capture_output=True, text=True
            )
            bt.logging.info(f"Container is running: {result}")
            
            # If the result contains output, the container is running
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            bt.logging.warning(f"Error checking container status: {e}")
            return False

    def start_service(self):
        """Start the miner-api service using Docker Compose."""
        bt.logging.info(f"Starting {self.service_name} service...")
        subprocess.run(["docker-compose", "up", "--build", "--timeout", "300", "-d", self.service_name], check=True)

    def restart_service(self, model_url: str):
        try:
            # Restart the miner-api container
            my_env = os.environ.copy()
            my_env["MODEL_URL"] = model_url
            subprocess.run(["docker", "compose", "up", "--force-recreate", "-d", self.service_name], env=my_env)

            bt.logging.info("miner-api container restarted successfully.")
        except subprocess.CalledProcessError as e:
            bt.logging.warning(f"Error restarting miner-api: {e}")

    def initialize_miner_api(self, model_url: str) -> bool:
        # determines if container is already running. If it is then restarts it otherwise starts it
        self.restart_service(model_url)
        
        max_wait_time = 500
        return self._poll_service_for_readiness(max_wait_time)

    def stop_service(self):
        """Stop and clean up the miner-api service without affecting the validator service."""
        bt.logging.info(f"Stopping {self.service_name} service...")
        
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
            bt.logging.info(f"Removing Docker image: {image_name}...")
            subprocess.run(["docker", "rmi", image_name], check=True)
        except subprocess.CalledProcessError as e:
            bt.logging.warning(f"Error removing image: {e}. It may not exist or is in use.")

    def query_eval(self, request: EvalRequest, timeout: int) -> EvalResponse:
        """Invoke the API running in the nested Docker container with queries."""
        bt.logging.info(f"Querying API on container {self.service_name}...")
        try:
            response = requests.post(
                f"{self.api_url}/eval_query",
                json=request.dict(),
                timeout=timeout
            )
            resp = response.json()
            return EvalResponse(
                score = resp.get("score"),
                mistakes = resp.get("mistakes"),
                response_time = resp.get("response_time"),
                status_message = APIStatus.SUCCESS
            )
        
        except Timeout as e:
            bt.logging.error(f"Timed out API request: {e}")
            return EvalResponse(
                score = -1, 
                mistakes = [],
                response_time = None,
                status_message = APIStatus.TIMEOUT
            )

        except Exception as e:
            bt.logging.error(f"Failed to query API: {e}")
            return EvalResponse(
                score = -1, 
                mistakes = [],
                response_time = None,
                status_message = APIStatus.ERROR
            )
    
    def get_model_hash(self)->str:
        try:
            response = requests.get(
                f"{self.api_url}/get_model_hash",
                timeout=60
            )
            resp = response.json()
            return resp.get("hash")

        except Exception as e:
            bt.logging.error(f"Failed to get hash: {e}")
            return None

    def get_model_coldkey(self)->str:
        try:
            response = requests.get(
                f"{self.api_url}/get_model_coldkey",
                timeout=60
            )
            resp = response.json()
            return resp.get("coldkey")

        except Exception as e:
            bt.logging.error(f"Failed to get Coldkey: {e}")
            return None

    def get_container_size(self):
        try:
            result = subprocess.run(
                ["docker", "inspect", self.service_name, "--size"],
                capture_output=True,
                text=True,
                check=True
            )
            
            container_info = json.loads(result.stdout)
            size_rw = container_info[0].get("SizeRw", 0)
            
            return size_rw / (1024 ** 3)  # to GB
        
        except subprocess.CalledProcessError as e:
            print(f"Error running docker inspect: {e}")
            return None
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error parsing docker inspect output: {e}")
            return None

        


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


