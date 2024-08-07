import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
import random
from pydantic import BaseModel



# Used to obtain the set of contexts and claims 
ATTRIBUTION_SYSTEM_PROMPT = """\
You are an expert at creating real-world business scenarios.  Your goal is to generate the input context that a business would use in a RAG setting and corresponding action items derived from the input context. 
The context that you create should be greater than 5 sentences and the following list of action items should be no more than 3 items.
"""

ATTRIBUTION_PROMPT_TEMPLATE = """\
Your goal is to generate a context consisting of at least 5-10 sentences that mimics a business setting. Each context should be a conversation between two or more people. \
Within this context, there should be 1 to 2 defined takeaways that would require action at a later date. For example, requesting that a team member schedule a follow-up meeting. \
These take aways should be defined as action items for follow up formatted as "person 1: schedule a follow-up meeting", where the action item is attributed to 'person 1' and stored in the json response. \
The actions must be derived from the context.  

I will give you four parameters: a topic, whether the action items should be attributed correctly or not, a context type, and the number of participants in the conversation.\
In response you will return the context in JSON format following the structure defined below.\
The context should be centered around the topic and the number of participants should define how many people are a part of the conversation.

If the attribution should be correct, then the person the action item is attributed to must be correct. \
However, if the attribution should be incorrect, then the person the action item is attributed to must be incorrect. \
But, you must only use the name of a participant found in the context when giving the incorrect attribution.

The context type parameter should guide the style that you write the context in and should be formatted as a dialogue between groups of people.\
For example, if it is a sales call with three participants then conversation may include 1 salesman and 2 prospective customers.

I will provide previous portions of the conversations, when generating the new context, keep in mind the past context.  \
The new context should follow the past context to generate a consistent conversation.

#Parameters:
- Topic: {topic}
- Context type: {context_type}
- Number of participants: {num_participants}
- Correct attribution or not: {attributed_correctly}

#Past context:
{past_context}

#JSON structure
{{
    "context": string,
    "action_items": list[string]
}}

You should return only the JSON as a string and no other text or markers.
"""

class Config(BaseModel):
    context: str
    action_items: list[str]
    true_or_false: bool


@dataclass
class AttributionTask(Task):
    name = TasksEnum.ATTRIBUTION.value
    desc = "Estimates the number of correctly attributed action items in a response given a RAG context"
    goal = "to identify the correct number of attributions"

    max_particpants = 5
    max_paragraphs = 10

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
        num_action_groups = random.randint(1, num_pagraphs)
        num_participants = random.randint(1, self.max_particpants)
        probability_true = random.random()
        system_prompt = ATTRIBUTION_SYSTEM_PROMPT

        resp_tmp = None
        for _ in range(num_pagraphs):
            true_or_false = True if random.random() <= probability_true else False

            if resp_tmp is not None:
                past_context = resp_tmp.context
            else:
                past_context = ""

            query_prompt = ATTRIBUTION_PROMPT_TEMPLATE.format(
                topic=context.topic,  
                context_type=context.context_type,
                num_participants = num_participants,
                attributed_correctly=true_or_false, 
                past_context=past_context)
            response = self.generate_input(llm_pipeline, query_prompt, system_prompt)

            # format 
            json_response = self.parse_llm_query(response)
            json_response['true_or_false'] = true_or_false
            resp_tmp = Config(**json_response)
            responses.append(resp_tmp)
            

        self.generate_reference(responses, num_action_groups)
        
        # TODO: I dont think these are right for any of them 
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def generate_reference(self, responses: list[Config], num_action_groups: int):
        contexts = [r.context for r in responses]
        self.rag_context = random.choice(self.joiners).join([c for c in contexts])

        # reference and responses  
        subset_action_items = random.sample(responses, num_action_groups)
        num_true = sum([len(a_item.action_items) for a_item in subset_action_items if a_item.true_or_false == True])
        total = sum([len(a_item.action_items) for a_item in subset_action_items])
        self.reference = round(num_true / (total + 1e-10), 2) 

        action_items = [random.choice(self.joiners).join(a for a in r.action_items) for r in subset_action_items]
        random.shuffle(action_items)
        self.llm_response = random.choice(self.joiners).join([a_item for a_item in action_items])
