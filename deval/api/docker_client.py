import time
from deval.protocol import EvalRequest, EvalResponse

class DockerClient:

    def __init__(self):
        self.host = "localhost" # TODO: add actual host 

    def init_miner_docker(self, model_dir: str) -> None:
        pass 

    def invoke(self, request) -> dict:
        #TODO: implement
        """
        To keep in mind:
        - error handling a response
        - forcefully timing out if the response takes too long 
        """
        return {
                "completion": -1
            } 

    def query_miner(self, uid: int, request: EvalRequest) -> EvalResponse:
        start_time = time.time()
        response_raw = self.invoke(request)
        process_time = time.time() - start_time

        response = EvalResponse(
            uid = uid,
            completion = response_raw.get("completion", -1),
            process_time = process_time
        )
        return response

        

