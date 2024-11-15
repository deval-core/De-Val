import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import TasksEnum
from deval.tasks.tool_schema import ToolSchemaGenerator
import random
from pydantic import BaseModel, ValidationError
from json.decoder import JSONDecodeError
from deval.rewards.reward import RewardReferenceType
from deval.tasks.hallucination.hallucination_base import HallucinationBaseTask

TOPIC_GEN_SYSTEM_PROMPT = """\
You are an expert critical reader, adept at pulling out the most salient and key points of any text data
"""

TOPIC_GEN_PROMPT="""\
Your goal is to extract between 3 and 20 key topics from the provided context.  You must always provide at least 3 topics, \
but should provide no more than 20.  These key topics should be should single line summaries of all of the key points for the provided article. \
We will be generating actual summaries from these topics, so they should be unique from one another and can later be summarized with between 50 to 100 words. \
It is important for these key topics to be comprehensive of all key topics in the article. 

I will give you the context that you should derive the key topics from. You must return the key topics in JSON format following the structure defined below. 

# Context
{context}

#JSON structure
{{
    "key_topics": list[string]
}}

Return the requested information as dictated by the provided tool schema. Do not return any other text besides the JSON response.
"""


# Used to obtain the set of contexts and claims 
TOPIC_HALLUCINATION_SYSTEM_PROMPT = """\
You are an expert summarizer and write in a clear and coherent voice. 
"""

TOPIC_HALLUCINATION_PROMPT_TEMPLATE = """\
Your goal is to generate a summary that is either True or False using the provided context based on the provided topic. Your summary should be no more \
than 100 words and should only summarize the defined topic.  You must return the summary in text format without any other text. 

I will provide four pieces of information: a context, a topic, whether the summary should be true or false, and a difficulty rating.  

The context represents the entire article and the topic is the key point that we are looking to summarize.  \
The summary should only summarize information based on the provided context and should not contain extra fluff. 

I will define the validity of the summary as either True or False. \
If the summary should be true, then a reader should be able to determine if that is the case by reading the provided context. \
The same applies if it should be false, where the reader can identify that it is false given just the information from the provided context. \
Do not give a false summary to the query that cannot be determined to be untrue from the context. The entire summary should be false when requested with no true facts.

I will give a difficulty rating - this rating should decide how difficult it should be for the reader to identify \
if the summary is True or not.  If the difficulty is hard then it should be very difficult for the reader to catch hallucinations, \
but if the difficulty is easy then it should be easy for the reader. 


# Context:
{context}

# Topic:
{topic}

# Validity of the response: 
{hallucination_or_not}

# Difficulty rating: 
{difficulty_rating}

Do not return any other text besides the requested summary. 
"""

class Config(BaseModel):
    topic: str
    claim: str
    true_or_false: bool



@dataclass
class HallucinationWikipediaTopicTask(HallucinationBaseTask):
    #name = TasksEnum.HALLUCINATION.value
    #desc = "Generates summaries for all identified key topics and filters out randomly to generate a missing information task."
    #goal = "Estimates the comprehensiveness of a summary"

    properties = {
        "key_topics": {
            "type": "array",
                "items": {
                    "type": "string",
                },
            "description": "The generated set of topics derived from the context. Returned as a list of string where each item is a topic.",
        },
    }
    required_values = ["key_topics"]

    #tool_schema_generator = ToolSchemaGenerator(name, desc, properties, required_values)


    def __init__(self, llm_pipeline, context):
        full_content = context.content
        self.context = context
        topics = []
        summaries = []
        probability_true = random.random()
        topic_system_prompt = TOPIC_GEN_SYSTEM_PROMPT

        tool_schema_generator = ToolSchemaGenerator(self.name, self.desc, self.properties, self.required_values)
        tool_schema = tool_schema_generator.get_schema(llm_pipeline)

        query_prompt = TOPIC_GEN_PROMPT.format(
            context = full_content
        )
        response = self.generate_input(llm_pipeline, query_prompt, topic_system_prompt, tool_schema)
        try:
            json_response = self.parse_llm_query(response)
            topics = json_response['key_topics']
        except (JSONDecodeError, ValidationError) as e:
            num_summaries -= 1 # we decrease number of claims for each unparseable response
            bt.logging.debug(f"Experienced {e} in Summary Completeness Wikipedia task")

        for topic in topics:
            true_or_false = True if random.random() <= probability_true else False

            query_prompt = TOPIC_HALLUCINATION_PROMPT_TEMPLATE.format(
                topic=topic, 
                context=full_content,
                hallucination_or_not=true_or_false, 
                difficulty_rating=context.difficulty,
            )
            summary = self.generate_input(llm_pipeline, query_prompt, TOPIC_HALLUCINATION_SYSTEM_PROMPT, None)

            # format 
            summaries.append(
                Config(
                    topic=topic,
                    claim=summary,
                    true_or_false=true_or_false
                )
            )
               
        self.generate_reference(summaries, len(summaries), full_content)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
        self.api = llm_pipeline.api.value
        self.model_id = llm_pipeline.model_id

    def generate_reference(self, responses: list[Config], num_claims: int, context: str):
        # context input 
        self.rag_context = context

        # reference and responses  
        subset_claims = random.sample(responses, max(num_claims, 1)) # we must always have at least 1 claim
        num_true = len([claim for claim in subset_claims if claim.true_or_false in [True, 'Neither']])
        self.reference = round(num_true / (len(subset_claims) + 1e-10), 2) 

        claims = [r.claim for r in subset_claims]
        random.shuffle(claims)
        self.llm_response = "".join([c + random.choice(self.joiners) for c in claims])

        # store mistakes for comparisons
        self.reference_mistakes = [r.claim for r in subset_claims if r.true_or_false == False]
        self.reference_true_values = [r.claim for r in subset_claims if r.true_or_false == True]