from transformers import Pipeline


"""
Notes
- how to pass in the model location and tokenizer?


"""


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, model_dir, maybe_arg=2):
        # parse input 

        # get tokenizer 

        # encode

        # pad and convert to tensor 

        # return 
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # take value from preprocess and pass to model 

        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        # get results 


        # post-process 
        best_class = model_outputs["logits"].softmax(-1)
        return best_class