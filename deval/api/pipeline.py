from transformers import Pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
from contextlib import nullcontext
from .prompts import (
    RELEVANCY_PROMPT, 
    HALLUCINATION_PROMPT, 
    ATTRIBUTION_PROMPT, 
    SUMMARY_COMPLETENESS_PROMPT,
)
import time
import json

class DeValPipeline(Pipeline):

    def __init__(self, model=None, tokenizer=None, model_dir = None, **kwargs):
        self.max_tokens = 250
        self.temperature = 0.5
        self.top_p = 0.95
        self.top_k = 0

        self.system_prompt = "You are an evaluation LLM. Your job is generate a score demonstrating how well the LLM you are evaluating responded and to identify its mistakes."

        # init tokenizer and model then attach
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        print(f"putting model to {self.device}")
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        for k, v in kwargs.items():
            preprocess_kwargs[k] = kwargs[k]
        return preprocess_kwargs, {}, {}

    def _gen_input_ids(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        return input_ids

       
    def _get_prompt(
        self, 
        task: str, 
    ) -> str:
        if task == "attribution":
            return  ATTRIBUTION_PROMPT
               
        elif task == 'summary_completeness':
            return SUMMARY_COMPLETENESS_PROMPT
        elif task == "hallucination":
            return HALLUCINATION_PROMPT
        elif task == "relevancy":
            return RELEVANCY_PROMPT
        else:
            raise ValueError(f"Unable to find the correct task: {task}")

       
    def preprocess(
        self, 
        inputs,
        tasks: list[str],
        rag_context: str,
        query: str | None,
        llm_response: str,
    ):
        # generate our prompts
        prompt = self._get_prompt(
            task=tasks[0],
        )

        # prep score evaluation 
        prompt = prompt.format(rag_context = rag_context, query = query, llm_response = llm_response)
        input_ids =self._gen_input_ids(prompt)

       
        return {
            "input_ids": input_ids
        }

    def _forward(self, model_inputs):
        input_ids = model_inputs['input_ids']

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext():

            start_score_time = time.time()
            # run eval score
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            response = outputs[0][input_ids.shape[-1]:]
            print(f"Score generation time: {time.time()-start_score_time}")


        return {
            "response": response
        }

    def postprocess(self, response):
        response = response.get('response')

        # decode and parse score
        decoded = self.tokenizer.decode(response, skip_special_tokens=True)
        completion = self._parse_score_response(decoded)

        output = json.loads(completion)

        return {
            'score_completion': output.get("score", -1.0),
            'mistakes_completion': output.get("mistakes", [])
        }



if __name__ == "__main__":
    from transformers.pipelines import PIPELINE_REGISTRY

    PIPELINE_REGISTRY.register_pipeline("de_val", pipeline_class=DeValPipeline)

    model_dir = "../model"

    tasks = ['relevancy']
    rag_context = "The earth is round. The sky is Blue."
    llm_response = "The earth is flat."
    query = "What color is the sky"

    pipe = DeValPipeline("de_val", model_dir = model_dir)
    print(pipe("", tasks=tasks, rag_context=rag_context, query=query, llm_response=llm_response))