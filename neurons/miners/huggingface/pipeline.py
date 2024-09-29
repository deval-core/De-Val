from transformers import Pipeline, AutoTokenizer, AutoModelForCausalLM
import re
from neurons.miners.huggingface.prompts import (
    RELEVANCY_PROMPT, 
    HALLUCINATION_PROMPT, 
    HALLUCINATION_MISTAKES_PROMPT,
    ATTRIBUTION_PROMPT, 
    ATTRIBUTION_MISTAKES_PROMPT,
    SUMMARY_COMPLETENESS_PROMPT,
    SUMMARY_MISTAKES_PROMPT
)

class DeValPipeline(Pipeline):

    def __init__(self, model=None, tokenizer=None, model_dir = None, **kwargs):
        self.max_tokens = 250
        self.temperature = 0.5
        self.top_p = 0.95
        self.top_k = 0

        self.system_prompt = "You are an evaluation LLM. Your job is generate a score demonstrating how well the LLM you are evaluating responded and to identify its mistakes."

        # init tokenizer and model then attach
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu")
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
        ).to(self.model.device)

        return input_ids

       
    def _get_prompt(
        self, 
        task: str, 
    ) -> str:
        if task == "attribution":
            return {
                "score": ATTRIBUTION_PROMPT,
                "mistakes": ATTRIBUTION_MISTAKES_PROMPT,
            }
        elif task == 'summary_completeness':
            return {
                "score": SUMMARY_COMPLETENESS_PROMPT,
                "mistakes": SUMMARY_MISTAKES_PROMPT,
            }
        elif task == "hallucination":
            return {
                "score": HALLUCINATION_PROMPT,
                "mistakes": HALLUCINATION_MISTAKES_PROMPT,
            }
        elif task == "relevancy":
            return {"score": RELEVANCY_PROMPT}
        else:
            raise ValueError(f"Unable to find the correct task: {task}")

    def _parse_score_response(self, response: str) -> float:
        float_regex = "((0\.\d+?|1\.0+?|0|1|\.\d+))"
        match = re.search(f"response: {float_regex}", response.lower())
        if match:
            score = match.group(1)
            print("score ", score)
            return float(score.strip()) if score != "" else -1.0
        else:
            print("Unable to parse response")
            return -1.0

    def _parse_mistakes_response(self, response: str) -> list[str]:
        response = response.split("\n")   
        response = [r.strip() for r in response]
        return [r for r in response if r != '']

       
    def preprocess(
        self, 
        inputs,
        tasks: list[str],
        rag_context: str,
        query: str | None,
        llm_response: str,
    ):
        # generate our prompts
        prompts = self._get_prompt(
            task=tasks[0],
        )

        # prep score evaluation 
        score_prompt = prompts.get("score")
        score_prompt = score_prompt.format(rag_context = rag_context, query = query, llm_response = llm_response)
        score_input_ids =self._gen_input_ids(score_prompt)

        # prep mistake identification 
        mistakes_prompt = prompts.get("mistakes", None)
                
        # we do not evaluate for all tasks
        if mistakes_prompt:
            mistakes_prompt = mistakes_prompt.format(rag_context = rag_context, llm_response = llm_response)
            mistakes_input_ids =self._gen_input_ids(mistakes_prompt)
        else:
            mistakes_input_ids = None

       
        return {
            "score_input_ids": score_input_ids,
            "mistakes_input_ids": mistakes_input_ids,
        }

    def _forward(self, model_inputs):
        score_input_ids = model_inputs['score_input_ids']
        mistake_input_ids = model_inputs.get('mistakes_input_ids', None)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # run eval score
        score_outputs = self.model.generate(
            input_ids=score_input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        score_response = score_outputs[0][score_input_ids.shape[-1]:]

        # run mistakes eval score
        mistakes_response = None
        if mistake_input_ids is not None:
            mistakes_outputs = self.model.generate(
                input_ids=mistake_input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            mistakes_response = mistakes_outputs[0][score_input_ids.shape[-1]:]


        return {
            "score_response": score_response,
            "mistakes_response": mistakes_response
        }

    def postprocess(self, response):
        score_response = response.get('score_response')
        mistakes_response = response.get('mistakes_response', None)

        # decode and parse score
        score_decoded = self.tokenizer.decode(score_response, skip_special_tokens=True)
        print(score_decoded)
        score_completion = self._parse_score_response(score_decoded)

        # decode and parse mistakes
        mistakes_completion = None
        if mistakes_response is not None:
            mistakes_decoded = self.tokenizer.decode(mistakes_response, skip_special_tokens = True)
            print(mistakes_decoded)
            mistakes_completion = self._parse_mistakes_response(mistakes_decoded)

        return {
            'score_completion': score_completion,
            'mistakes_completion': mistakes_completion
        }



if __name__ == "__main__":
    from transformers.pipelines import PIPELINE_REGISTRY

    PIPELINE_REGISTRY.register_pipeline("de_val", pipeline_class=DeValPipeline)

    model_dir = "../model"

    tasks = ['hallucination']
    rag_context = "The earth is round. The sky is Blue."
    llm_response = "The earth is flat."
    query = ""

    pipe = DeValPipeline("de_val", model_dir = model_dir)
    print(pipe("", tasks=tasks, rag_context=rag_context, query=query, llm_response=llm_response))