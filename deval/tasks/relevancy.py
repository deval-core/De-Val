import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
from deval.tasks.tool_schema import ToolSchemaGenerator
import random
from pydantic import BaseModel


# Used to obtain the set of contexts and QA pairs 
RELEVANCY_SYSTEM_PROMPT = """\
You are an expert at creating real-world business scenarios.  Your goal is to generate a QA pair from the provided context.
The QA pair should consist of a simple user query that a user may ask a question answer system followed by an answer. 
"""

RELEVANCY_PROMPT_TEMPLATE = """\
Your goal is to generate a QA pair corresponding to the provided context. \
The purpose of the response is to generate an answer to the generated query that is either relevant or irrelevant.  If the answer \
should be relevant then a human reader should be able to determine that the answer does respond to the query.  Whereas, if the answer \
should not be relevant then a human reader should be able to detect that the answer is not answering the generated query.

I will give you three inputs: whether the answer should be relevant to the query, a difficulty rating, and the context to rely upon.\
In response you will return the QA pairs in JSON format following the structure defined below.\

I will give a difficulty rating - this rating should decide how difficult it should be for the reader to identify \
if the answer to the query is relevant or not.  If the difficulty is hard then it should be very difficult for the reader to detect an irrelevant answer, \
but if the difficulty is easy then it should be easy for the reader. 


#Parameters:
- Relevant or not: {relevant_or_not}
- Difficulty rating: {difficulty_rating}

#Context:
{context}

#JSON structure
{{
    "query": string,
    "answer": string
}}

Return the requested informat as dictated by the provided tool schema. Do not return any other text besides the JSON response.
"""

class Config(BaseModel):
    context: str
    query: str
    answer: str
    relevant_or_not: bool


@dataclass
class RelevancyTask(Task):
    name = TasksEnum.RELEVANCY.value
    desc = "Estimates the relevancy of an answer to a user query based on a provided context"
    goal = "to identify whether an answer to a user query is relevant or not"

    properties = {
        "query": {
            "type": "string",
            "description": "The generated user query from the provided context",
        },
        "answer": {
            "type": "string",
            "description": "The generated answer based on the query and context that is either relevant or irrelevant",
        },
    }
    required_values = ["query", "answer"]

    tool_schema_generator = ToolSchemaGenerator(name, desc, properties, required_values)


    reward_definition = [
        dict(name="ordinal", weight=1.0),
    ]
    penalty_definition = []

    def __init__(self, llm_pipeline, context):
        self.context = context.content
        
        system_prompt = RELEVANCY_SYSTEM_PROMPT
        probability_relevant = random.random()
        relevant_or_not = True if random.random() <= probability_relevant else False
        tool_schema = self.tool_schema_generator.get_schema(llm_pipeline)

        query_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
            relevant_or_not = relevant_or_not,
            difficulty_rating = context.difficulty,
            context=self.context
        )
        response = self.generate_input(llm_pipeline, query_prompt, system_prompt, tool_schema)

        # format 
        json_response = self.parse_llm_query(response)
        json_response['context'] = self.context
        json_response['relevant_or_not'] = relevant_or_not
        response = Config(**json_response)
        

        self.generate_reference(response)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
        self.api = llm_pipeline.api.value
        self.model_id = llm_pipeline.model_id

    def generate_reference(self, response: Config):
        self.rag_context = response.context
        self.query = response.query
        self.llm_response = response.answer
        self.reference = 1.0 if response.relevant_or_not else 0.0
