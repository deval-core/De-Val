# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import gc
import time
import bittensor as bt
from typing import List, Dict
#from deval.cleaners.cleaner import CleanerPipeline
from deval.llms.base_llm import BasePipeline, BaseLLM
from deval.mock import MockPipeline
from openai import OpenAI 
import os

API_KEY = os.getenv("OPENAI_API_KEY", None)
if API_KEY is None:
    raise ValueError("Please add OPENAI_API_KEY to your environment")


class OpenAIPipeline(BasePipeline):
    def __init__(self, model_id: str, mock=False):
        super().__init__()
        self.model_id = model_id
        self.llm = self.load_pipeline(model_id, mock)
        self.mock = mock

    def __call__(self, messages: List[Dict[str, str]], **model_kwargs: Dict) -> str:
        if self.mock:
            return self.llm(messages, **model_kwargs)

        # Compose sampling params
        temperature = model_kwargs.get("temperature", 0.2)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 500)
        format = model_kwargs.get("format", "json_object")


        output = self.llm.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                response_format={ "type": format }
            )
        response = output.choices[0].message
        return response.content

    def load_pipeline(self,model_id: str, mock=False):
        """Loads the OpenAI GPT pipeline for the LLM, or a mock pipeline if mock=True"""
        if mock or model_id == "mock":
            return MockPipeline(model_id)

        return OpenAI(api_key=API_KEY)




class OpenAILLM(BaseLLM):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.95,
        format="json_object"
    ):
        model_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "format": format
        }
        super().__init__(llm_pipeline, system_prompt, model_kwargs)

        # Keep track of generation data using messages and times
        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times = [0]

    def query(
        self,
        prompt: str,
        #cleaner: CleanerPipeline = None,
    ):
        # Adds the message to the list of messages for tracking purposes, even though it's not used downstream
        messages = self.messages + [{"content": prompt, "role": "user"}]

        t0 = time.time()
        response = self.forward(messages=messages)
        #response = self.clean_response(cleaner, response)

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - t0]

        return response

    def forward(self, messages: List[Dict[str, str]]):
        # make composed prompt from messages
        response = self.llm_pipeline(messages, **self.model_kwargs)

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{response}"
        )

        return response


if __name__ == "__main__":
    # Example usage
    llm_pipeline = OpenAIPipeline(
        model_id="gpt-3.5-turbo-0125",  mock=False
    )
    llm = OpenAILLM(llm_pipeline, system_prompt="You are a helpful AI assistant")

    message = "What is the capital of Texas?"
    response = llm.query(message)
    print(response)