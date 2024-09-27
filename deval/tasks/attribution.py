import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
import random
from pydantic import BaseModel, ValidationError
from json.decoder import JSONDecodeError
from deval.tasks.tool_schema import ToolSchemaGenerator



# Used to obtain the set of contexts and claims 
ATTRIBUTION_SYSTEM_PROMPT = """\
You are an expert at creating real-world business scenarios.  Your goal is to generate the input context that a business would use in a RAG setting and corresponding action items derived from the input context. 
The context that you create should be greater than 5 sentences and the following list of action items should be no more than 3 items.
"""

ATTRIBUTION_PROMPT_TEMPLATE = """\
Your goal is to generate a context consisting of at least 3-5 sentences that mimics a business setting. Each context should be a conversation between two or more people. \
Within this context, there should be 1 to 2 defined takeaways that would require action at a later date. For example, requesting that a team member schedule a follow-up meeting. \
These take aways should be defined as action items for follow up formatted as "person 1: schedule a follow-up meeting", where the action item is attributed to 'person 1' and stored in the json response. \
The actions must be related to the generated context.  You should avoid very general action items and instead be as specific as possible.  

I will give you four parameters: a topic, whether the action items should be attributed correctly or not, a context type, and the number of participants in the conversation.\
In response you will return the context in JSON format following the structure defined below.\
The context should be centered around the topic and the number of participants should define how many people are a part of the conversation.

If the attribution should be correct, then the person the action item is attributed to and the action item must be correct. \
However, if the attribution should be incorrect, then the person the action item is attributed to or the actual action item must be incorrect. \
But, you must only use the name of a participant found in the context when giving the incorrect attribution.

The context type parameter should guide the style that you write the context in and should be formatted as a dialogue between groups of people.\
For example, if it is a sales call with three participants then conversation may include 1 salesman and 2 prospective customers.

I will provide previous portions of the conversations, when generating the new context, keep in mind the past context.  \
The new context should follow the past context to generate a consistent conversation or story.

I will also provide past action items that you created.  The new action item that you create must be unique and cannot appear in this list. \
Do not ever create a new action item that already appears in the past action items list below. 

#Parameters:
- Topic: {topic}
- Context type: {context_type}
- Number of participants: {num_participants}
- Correct attribution or not: {attributed_correctly}

#Past context:
{past_context}

#Past action items
{past_action_items}

#JSON structure
{{
    "context": string,
    "action_item": string
}}

Return the requested informat as dictated by the provided tool schema. Do not return any other text besides the JSON response.
"""

class Config(BaseModel):
    context: str
    action_item: str
    true_or_false: bool


@dataclass
class AttributionTask(Task):
    name = TasksEnum.ATTRIBUTION.value
    desc = "Generate a context and associated action items for a misattribution evaluation task"
    goal = "Estimates the number of correctly attributed action items in a response given a RAG context"

    max_particpants = 5
    max_paragraphs = 5

    properties = {
        "context": {
            "type": "string",
            "description": "The generated context used as input into an LLM RAG pipeline",
        },
        "action_item": {
            "type": "string",
            "description": "The generated action items this derived from the context",
        },
    }
    required_values = ["context", "action_item"]

    tool_schema_generator = ToolSchemaGenerator(name, desc, properties, required_values)

    reward_definition = [
        dict(name="float_diff", weight=1.0),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.5),
    ]

    def __init__(self, llm_pipeline, context):
        self.context = context
        responses = []


        num_pagraphs = random.randint(5, self.max_paragraphs)
        num_action_groups = random.randint(3, num_pagraphs)
        num_participants = random.randint(2, self.max_particpants)
        probability_true = random.random()
        system_prompt = ATTRIBUTION_SYSTEM_PROMPT
        tool_schema = self.tool_schema_generator.get_schema(llm_pipeline)

        resp_tmp = None
        for _ in range(num_pagraphs):
            true_or_false = True if random.random() <= probability_true else False

            if resp_tmp is not None:
                past_context = resp_tmp.context
                past_action_items = past_action_items + "\n" + resp_tmp.action_item
            else:
                past_context = ""
                past_action_items = ""

            query_prompt = ATTRIBUTION_PROMPT_TEMPLATE.format(
                topic=context.topic,  
                context_type=context.context_type,
                num_participants = num_participants,
                attributed_correctly=true_or_false, 
                past_context=past_context,
                past_action_items=past_action_items)
            response = self.generate_input(llm_pipeline, query_prompt, system_prompt, tool_schema)

            # format 
            try:
                json_response = self.parse_llm_query(response)
                json_response['true_or_false'] = true_or_false
                resp_tmp = Config(**json_response)
                responses.append(resp_tmp)
            except (JSONDecodeError, ValidationError) as e:
                num_action_groups -= 1 # we decrease number of claims for each unparseable response
                bt.logging.debug(f"Experienced {e} in Attribution task")
                continue
            

        self.generate_reference(responses, num_action_groups)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
        self.api = llm_pipeline.api.value
        self.model_id = llm_pipeline.model_id

    def generate_reference(self, responses: list[Config], num_action_groups: int):
        contexts = [r.context for r in responses]
        self.rag_context = "".join([c + random.choice(self.joiners) for c in contexts])

        # reference and responses  
        subset_action_items = random.sample(responses, max(num_action_groups, 3)) # we must always have at least 3 action groups
        num_true = len([a_item for a_item in subset_action_items if a_item.true_or_false == True])
        self.reference = round(num_true / (len(subset_action_items) + 1e-10), 2) 

        action_items = [r.action_item for r in subset_action_items]
        random.shuffle(action_items)
        self.llm_response = "".join([a_item + random.choice(self.joiners) for a_item in action_items])