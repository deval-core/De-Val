
import time
import bittensor as bt
from deval.llms.base_llm import BaseLLM
from deval.llms.config import LLMAPIs, LLMArgs
import os
import json
import boto3
from botocore.exceptions import ClientError



class AWSBedrockLLM(BaseLLM):
    def __init__(
        self,
        model_id: str,
        model_kwargs: LLMArgs,
    ):
        api = LLMAPIs.BEDROCK
        super().__init__(api, model_id, model_kwargs)
        self.llm = self.load()

    def query(
        self,
        prompt: str,
        system_prompt: str,
        tool_schema: dict | None = None
    ):
        self.messages = [{"content": [{"text":prompt}], "role": "user"}]
        self.system_prompt = system_prompt

        t0 = time.time()
        response = self.forward(messages=self.messages, tool_schema=tool_schema)

        self.messages = self.messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - t0]

        return response

    def forward(
        self, 
        messages: list[dict[str, str]],
        tool_schema: dict | None = None
    ) -> str:
        # Compose sampling params
        model_kwargs = self.model_kwargs # type of Dict of LLMArgs
        temperature = model_kwargs.get("temperature", 0.2)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 500)

        inference_config = {
            'maxTokens': max_tokens,
            'temperature': temperature,
            'topP': top_p,
        }

        if tool_schema:
            output = self.llm.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": self.system_prompt}],
                toolConfig=tool_schema,
                inferenceConfig=inference_config
            )
        else: 
            output = self.llm.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": self.system_prompt}],
                inferenceConfig=inference_config
            )
        
        content = self.parse_response(output)

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{content}"
        )

        return content
    
    def parse_response(self, output) -> str:
        if "mistral" in self.model_id:
            return self.parse_mistral(output)
        elif "anthropic" in self.model_id:
            return self.parse_anthropic_cohere(output)
        elif "cohere" in self.model_id:
            return self.parse_anthropic_cohere(output)
        else:
            return response
            #raise ValueError("Output parsing method for selected model ID not implemented")
    
    def parse_anthropic_cohere(self, output) -> str:
        # default to tools if provided, otherwise return message
        response = output.get('output').get('message').get('content')

        tool_calls = [r for r in response if "toolUse" in r]
        if len(tool_calls) > 0:
            tool_call = tool_calls[0].get("toolUse").get("input")
            content = json.dumps(tool_call)
        else:
            bt.logging.info("No tool response found, returning content")
            print("No tool response found, returning content")
            print(response)
            content =[r for r in response if "text" in r][0]['text']

        return content 



    def parse_mistral(self, output) -> str:
        # default to tools if provided, otherwise return message
        response = output.get("output").get("message").get("content")
        
        tool_calls = [r for r in response if "arguments" in r.get("text")]
        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            text = tool_call.get("text")
            content = json.dumps(json.loads(text).get("arguments"))
        else:
            bt.logging.info("No tool response found, returning content")
            print("No tool response found, returning content")
            content = [r for r in response if "text" in r][0]
            content =  content.get("text")
        
        return content
    
    def load(self):
        """Loads the Bedrock pipeline for the LLM"""
        access_key = os.getenv("AWS_ACCESS_KEY_ID", None)
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
        if access_key is None or secret_key is None:
            raise ValueError("Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to your environment to use this api")

        return boto3.client(
            service_name="bedrock-runtime",
        )

    def check_model_id_access(self) -> bool:
        # returns true if able to run a query against the selected model ID otherwise false
        # actual feature not yet implemented so this is Janky
        # https://github.com/aws/aws-sdk/issues/810

        prompt = "Do I have access?"
        system_prompt = "return True always"
        try:
            response = self.query(prompt, system_prompt)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                return False

        except Exception as e:
            bt.logging.error(f"Unhandled error at {e}")





if __name__ == "__main__":
    from deval.llms.base_llm import LLMFormatType
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())

    model_kwargs = LLMArgs(format = LLMFormatType.TEXT)
    llm = AWSBedrockLLM(model_id="anthropic.claude-3-haiku-20240307-v1:0",  model_kwargs=model_kwargs)

    message = "What is the capital of Texas?"
    response = llm.query(message, system_prompt="You are a helpful AI assistant")
    print(response)