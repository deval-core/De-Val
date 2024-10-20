import requests
from deval.protocol import init_request_from_task
from deval.api.models import EvalRequest, EvalResponse
import time
import subprocess
import bittensor as bt
import docker

class MinerDockerClient:

    def __init__(self):
        self.service_name = "miner-api"
        self.host = f"http://0.0.0.0" 
        self.port = 8000
        self.api_url = f"{self.host}:{self.port}"

    def _poll_service_for_readiness(self, max_wait_time: int) -> bool:
        num_checks = 10
        sleep_interval = int(max_wait_time/num_checks)

        for _ in range(num_checks):
            time.sleep(sleep_interval)
            try:
                response = requests.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    bt.logging.info("Successful connection to miner-api...")
                    return True

            except requests.ConnectionError as e:
                bt.logging.warning(f"Unable to connect to miner api... {e}")
                
            
        
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

    def restart_service(self):
        try:
            # Restart the miner-api container
            subprocess.run(["docker", "restart",  self.service_name], check=True)
            bt.logging.info("miner-api container restarted successfully.")
        except subprocess.CalledProcessError as e:
            bt.logging.warning(f"Error restarting miner-api: {e}")

    def initialize_miner_api(self) -> bool:
        # determines if container is already running. If it is then restarts it otherwise starts it
        self.restart_service()
        
        max_wait_time = 300
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


    