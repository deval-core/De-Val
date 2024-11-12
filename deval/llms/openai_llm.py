
import time
import bittensor as bt
from deval.llms.base_llm import BaseLLM
from deval.llms.config import LLMAPIs, LLMArgs
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
        self.llm = self.load()

    def query(
        self,
        prompt: str,
        system_prompt: str,
        tool_schema: dict | None = None
    ):
        self.messages = [{"content": system_prompt, "role": "system"}]
        messages = self.messages + [{"content": prompt, "role": "user"}]

        t0 = time.time()
        response = self.forward(messages=messages, tool_schema=tool_schema)

        self.messages = messages + [{"content": response, "role": "assistant"}]
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
        format = model_kwargs.get("format").value # type: str

        output = self.llm.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                tools = [tool_schema],
                tool_choice="auto",
                response_format={ "type": format }
            )
        
        content = self.parse_response(output)

        bt.logging.debug(
            f"{self.__class__.__name__} generated the following output:\n{content}"
        )

        return content
    
    def parse_response(self, output) -> str:
        # default to tools if provided, otherwise return message
        response = output.choices[0].message
        
        tool_calls = response.tool_calls
        if tool_calls is not None:
            tool_call = tool_calls[0]
            content = tool_call.function.arguments
        else:
            bt.logging.info("No tool response found, returning content")
            content =  response.content
        
        return content
    
    def load(self):
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