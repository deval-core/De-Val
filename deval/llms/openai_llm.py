
import time
import bittensor as bt
from typing import List, Dict
from deval.llms.base_llm import BaseLLM, LLMArgs, LLMAPIs
from deval.mock import MockPipeline
from openai import OpenAI 
import os



class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model_id: str,
        model_kwargs: LLMArgs,
    ):
        api = LLMAPIs.OPENAI
        super().__init__(api, model_id, model_kwargs)
        self.llm = self.load(model_id)

    def query(
        self,
        prompt: str,
        system_prompt: str,
    ):
        self.messages = [{"content": system_prompt, "role": "system"}]
        messages = self.messages + [{"content": prompt, "role": "user"}]

        t0 = time.time()
        response = self.forward(messages=messages)

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - t0]

        return response

    def forward(self, messages: list[dict[str, str]]) -> str:
        # Compose sampling params
        model_kwargs = self.model_kwargs # type of Dict of LLMArgs
        temperature = model_kwargs.get("temperature", 0.2)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 500)
        format = model_kwargs.get("format").value # type: str


        output = self.llm.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                response_format={ "type": format }
            )
        response = output.choices[0].message
        content =  response.content

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{content}"
        )

        return content
    
    def load(self, api_key: str):
        """Loads the OpenAI GPT pipeline for the LLM"""
        api_key = os.getenv("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("Please add OPENAI_API_KEY to your environment")


        return OpenAI(api_key=api_key)





if __name__ == "__main__":
    from deval.llms.base_llm import LLMFormatType
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())

    model_kwargs = LLMArgs(format = LLMFormatType.TEXT)
    llm = OpenAILLM(model_id="gpt-4o-mini",  model_kwargs=model_kwargs)

    message = "What is the capital of Texas?"
    response = llm.query(message, system_prompt="You are a helpful AI assistant")
    print(response)