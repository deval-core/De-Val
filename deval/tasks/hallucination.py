import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
import random
from pydantic import BaseModel



# Used to obtain the set of contexts and claims 
HALLUCINATION_SYSTEM_PROMPT = """\
You are an expert at creating real-world business scenarios.  Your goal is to generate the input context that a business would use in a RAG setting. 
The context that you create should be greater than 5 sentences and the following claim should be no more than 3 sentences.
"""

HALLUCINATION_PROMPT_TEMPLATE = """\
Your goal is to generate a context consisting of at least 5-10 sentences, and claim that is a statement to be derived from the context.\
However, I will request that this claim is either true or false based on the preceding context. 

I will give you five parameters: a topic, a sub-topic, whether the claim should be true or false, context type, and a difficulty rating.\
In response you will return the context in JSON format following the structure defined below.\
The context should be based on a the topic and sub-topic.

If the claim should be true, then a reader should be able to determine if that is the case by reading the preceding context.\
The same applies if it should be false, where the reader can identify that it is false given just the information from the preceding context.

The context type parameter should guide the style that you write the context in.  For example, if provided a context type of 'book' \
then the context should be formatted and written like a book, whereas, if provided the context type 'screenplay' then the context should be formatted like a screenplay.

I will give a difficulty rating - this rating should decide how difficult it should be for the reader to identify \
if the claim is a hallucination or not.  If the difficulty is hard then it should be very difficult for the reader to catch hallucinations, \
but if the difficulty is easy then it should be easy for the reader. 

I will provide previous portions of the story, when generating the new context, keep in mind the past context.  \
The new context should follow the past context to generate a consistent story.

#Parameters:
- Topic: {topic}
- Sub-topic: {subtopic}
- Context type: {context_type}
- Validity of the claim: {hallucination_or_not}
- Difficulty rating: {difficulty_rating}

#Past context:
{past_context}

#JSON structure
{{
    "context": string,
    "claim": string
}}

You should return only the JSON as a string and no other text or markers.
"""

class Config(BaseModel):
    context: str
    claim: str
    true_or_false: bool


@dataclass
class HallucinationTask(Task):
    name = TasksEnum.HALLUCINATION.value
    desc = "Estimates the number of hallucination in a response given a RAG context"
    goal = "to identify the correct number of hallucinations"

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
        num_claims = random.randint(1, num_pagraphs)
        probability_true = random.random()
        system_prompt = HALLUCINATION_SYSTEM_PROMPT

        resp_tmp = None
        for _ in range(num_pagraphs):
            true_or_false = True if random.random() <= probability_true else False

            if resp_tmp is not None:
                past_context = resp_tmp.context
            else:
                past_context = ""

            query_prompt = HALLUCINATION_PROMPT_TEMPLATE.format(
                topic=context.topic, 
                subtopic=context.subtopic, 
                context_type=context.context_type,
                hallucination_or_not=true_or_false, 
                difficulty_rating=context.difficulty, 
                past_context=past_context)

            response = self.generate_input(llm_pipeline, query_prompt, system_prompt)

            # format 
            json_response = self.parse_llm_query(response)
            json_response['true_or_false'] = true_or_false
            resp_tmp = Config(**json_response)
            responses.append(resp_tmp)
            

        self.generate_reference(responses, num_claims)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def generate_reference(self, responses: list[Config], num_claims: int):
        # context input 
        contexts = [r.context for r in responses]
        self.rag_context = "\n".join([c for c in contexts])

        # reference and responses  
        subset_claims = random.sample(responses, num_claims)
        num_true = len([claim for claim in subset_claims if claim.true_or_false == True])
        self.reference = round(num_true / (len(subset_claims) + 1e-10), 2) 

        claims = [r.claim for r in subset_claims]
        random.shuffle(claims)
        self.llm_response = "\n".join([c for c in claims])
