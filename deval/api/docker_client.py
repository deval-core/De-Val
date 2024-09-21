import requests
from deval.protocol import EvalRequest, EvalResponse

class DockerClient:

    def __init__(self):
        self.host = "http://localhost" # TODO: add actual host 
        self.port = 8000

        self.api_url = f"{self.host}:{self.port}"

    def init_miner_docker(self, model_dir: str) -> None:
        pass 

    def invoke(self, request: EvalRequest) -> EvalResponse:
        #TODO: implement
        """
        To keep in mind:
        - error handling a response
        - forcefully timing out if the response takes too long 
        """
        eval_query_url = f"{self.api_url}/eval_query"
        response = requests.post(eval_query_url, request)
        print(response.json())

        return EvalResponse(
            completion = -1, 
            response_time = 10
        )

    def query_miner(self, uid: int, request: EvalRequest) -> EvalResponse:
        
        response = self.invoke(request)
        response.uid = uid
        
        return response

        

