import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
from deval.tasks.tool_schema import ToolSchemaGenerator
import random
from pydantic import BaseModel, ValidationError
from json.decoder import JSONDecodeError
from deval.rewards.reward import RewardReferenceType

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
TOPIC_SUMMARY_SYSTEM_PROMPT = """\
You are an expert summarizer and write in a clear and coherent voice. 
"""

TOPIC_SUMMARY_PROMPT_TEMPLATE = """\
Your goal is to generate a summary using the provided context based on the provided topic. Your summary should be no more \
than 100 words and should only summarize the defined topic.  You must return the summary in text format without any other text.   

I will provide two pieces of information: a context and a topic.  The context represents the entire article and the topic is the key point \
that we are looking to summarize.  The summary should only summarize information based on the provided context and should not contain extra fluff. 

# Context:
{context}

# Topic:
{topic}

Do not return any other text besides the requested summary. 
"""

class Config(BaseModel):
    topic: str
    summary: str


@dataclass
class CompletenessWikipediaTask(Task):
    name = TasksEnum.COMPLETENESS.value
    desc = "Generates summaries for all identified key topics and filters out randomly to generate a missing information task."
    goal = "Estimates the comprehensiveness of a summary"

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

    tool_schema_generator = ToolSchemaGenerator(name, desc, properties, required_values)

    reward_definition = [
        dict(name="float_diff", weight=0.5, reference_type = RewardReferenceType.SCORE),
        dict(name="rouge", weight=0.25, reference_type = RewardReferenceType.MISTAKES),
        dict(name="relevance", weight=0.25, reference_type = RewardReferenceType.MISTAKES),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.25, reference_type = RewardReferenceType.SCORE),
        dict(name="rouge", weight=0.125, reference_type = RewardReferenceType.MISTAKES),
        dict(name="relevance", weight=0.125, reference_type = RewardReferenceType.MISTAKES),
    ]

    def __init__(self, llm_pipeline, context):
        full_content = context.content
        self.context = context
        topics = []
        summaries = []


        topic_system_prompt = TOPIC_GEN_SYSTEM_PROMPT
        tool_schema = self.tool_schema_generator.get_schema(llm_pipeline)

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
            
            query_prompt = TOPIC_SUMMARY_PROMPT_TEMPLATE.format(
                topic=topic, 
                context=full_content
            )
            summary = self.generate_input(llm_pipeline, query_prompt, TOPIC_GEN_SYSTEM_PROMPT, None)

            # format 
            summaries.append(
                Config(
                    topic=topic,
                    summary=summary
                )
            )
               
        self.generate_reference(summaries, len(summaries), full_content)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
        self.api = llm_pipeline.api.value
        self.model_id = llm_pipeline.model_id

    def generate_reference(self, responses: list[Config], num_summaries: int, context: str):
        # context input 
        self.rag_context = context

        # reference and responses  
        subset_summaries = random.sample(responses, max(num_summaries, 3)) # we must always have at least 3 summary points
        self.reference = round(num_summaries / (len(responses)+ 1e-10), 2) 

        summaries = [r.summary for r in subset_summaries]
        self.llm_response = "".join([c + random.choice(self.joiners) for c in summaries])

        # store mistakes for comparisons
        self.reference_mistakes = [s.summary for s in responses if s not in subset_summaries]
        self.reference_true_values = [s.summary for s in subset_summaries]