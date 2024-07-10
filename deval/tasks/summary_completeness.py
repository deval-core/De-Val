import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
import random
from pydantic import BaseModel


# Used to obtain the set of contexts and claims 
COMPLETENESS_SYSTEM_PROMPT = """\
You are an expert at creating real-world business scenarios.  Your goal is to generate the input context that a business would use in a RAG setting. 
The context that you create should be greater than 5 sentences and the following summary should be no more than 3 sentences.
"""

COMPLETENESS_PROMPT_TEMPLATE = """\
Your goal is to generate a context consisting of at least 5-10 sentences and a corresponding summary.\

I will give you three parameters: a topic, a sub-topic, and a context type.\
In response you will return the context in JSON format following the structure defined below.\
The context should be based on a the topic and sub-topic. The summary should be derived directly from the generated context.

The context type parameter should guide the style that you write the context in.  For example, if provided a context type of 'book' \
then the context should be formatted and written like a book, whereas, if provided the context type 'screenplay' then the context should be formatted like a screenplay.

I will provide previous portions of the story, when generating the new context, keep in mind the past context.  \
The new context should follow the past context to generate a consistent story.

#Parameters:
- Topic: {topic}
- Sub-topic: {subtopic}
- Context type: {context_type}

#Past context:
{past_context}

#JSON structure
{{
    "context": string,
    "summary": string
}}

You should return only the JSON as a string and no other text or markers.
"""

class Config(BaseModel):
    context: str
    summary: str


@dataclass
class CompletenessTask(Task):
    name = TasksEnum.COMPLETENESS.value
    desc = "Estimates the comprehensiveness of a summary"
    goal = "to identify how complete a provided summary is"


    max_paragraphs = 2

    reward_definition = [
        dict(name="float_diff", weight=1.0),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.5),
    ]

    def __init__(self, llm_pipeline, context):
        self.context = context
        responses = []


        num_pagraphs = random.randint(1, self.max_paragraphs)
        print(f"Number of paragraphs: {num_pagraphs}")
        num_summaries = random.randint(1, num_pagraphs)
        print(f"Number of summaries {num_summaries}")
        system_prompt = COMPLETENESS_SYSTEM_PROMPT

        resp_tmp = None
        for _ in range(num_pagraphs):

            if resp_tmp is not None:
                past_context = resp_tmp.context
            else:
                past_context = ""

            query_prompt = COMPLETENESS_PROMPT_TEMPLATE.format(
                topic=context.topic, 
                subtopic=context.subtopic, 
                context_type=context.context_type,
                past_context=past_context)
            response = self.generate_input(llm_pipeline, query_prompt, system_prompt)

            # format 
            json_response = self.parse_llm_query(response)
            print(json_response)
            resp_tmp = Config(**json_response)
            responses.append(resp_tmp)
            

        self.generate_reference(responses, num_summaries)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def generate_reference(self, responses: list[Config], num_summaries: int):
        # context input 
        contexts = [r.context for r in responses]
        self.rag_context = "\n".join([c for c in contexts])

        # reference and responses  
        subset_summaries = random.sample(responses, num_summaries)
        self.reference = round(num_summaries / (len(responses)+ 1e-10), 2) 

        summaries = [r.summary for r in subset_summaries]
        self.llm_response = " ".join([c for c in summaries])
