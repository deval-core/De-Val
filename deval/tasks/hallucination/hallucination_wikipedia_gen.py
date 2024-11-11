import bittensor as bt
from dataclasses import dataclass
from deval.tasks.task import Task, TasksEnum
from deval.tasks.tool_schema import ToolSchemaGenerator
import random
from pydantic import BaseModel, ValidationError
from json.decoder import JSONDecodeError
from deval.rewards.reward import RewardReferenceType


# Used to obtain the set of contexts and claims 
HALLUCINATION_SYSTEM_PROMPT = """\
You are an expert at generating responses to a query that can be either true or false given the provided context.  
"""

HALLUCINATION_PROMPT_TEMPLATE = """\
Below, I provide you a context extracted from Wikipedia. Your goal is to generate a response to the provided query that is True or False based on the context. \
You must return the response in a JSON format according to the provided tool schema.

I will give you four inputs: a wikipedia context, whether the response should be true or false, a difficulty rating, and past responses that you generated.\

I will define the validity of the response as either True or False. \
If the response should be true, then a reader should be able to determine if that is the case by reading the provided context. The response should be directly derived from the context. \
The same applies if it should be false, where the reader can identify that it is false given just the information from the provided context. Do not give a false response to the query that cannot be determined to be untrue from the context. 

The generated response should be between 2 to 3 sentences long and between 30 to 60 words long, but only contain a single fact.  

I will give a difficulty rating - this rating should decide how difficult it should be for the reader to identify \
if the response is a hallucination or not.  If the difficulty is hard then it should be very difficult for the reader to catch hallucinations, \
but if the difficulty is easy then it should be easy for the reader. 

Lastly, I will give you past responses that you generated for this same context. Your next response should be unique, but also flow nicely with previously generated claims to form a single coherent response to the query. \
Do not repeat yourself in text. These responses will be combined and should form a consistent summary to the original query. \
However, it is more important to generate true or false responses based on the context

#Parameters:
- Context: {context}
- Validity of the response: {hallucination_or_not}
- Difficulty rating: {difficulty_rating}
- Past responses: {past_responses}


#JSON structure and tool schema
{{
    "response": string
}}

Return the requested informat as dictated by the provided tool schema. Do not return any other text besides the JSON response.
"""

class Config(BaseModel):
    context: str
    claim: str
    true_or_false: bool


@dataclass
class HallucinationWikipediaGenTask(Task):
    name = TasksEnum.HALLUCINATION.value
    desc = "Utilizes wikipedia as a base and generates fake and true claims for a hallucination evaluation task"
    goal = "Estimates the number of hallucination in a response given a RAG context"
    properties = {
        "response": {
            "type": "string",
            "description": "The generated response that is either true or false generated from the context",
        },
    }
    required_values = ["response"]

    tool_schema_generator = ToolSchemaGenerator(name, desc, properties, required_values)


    reward_definition = [
        dict(name="float_diff", weight=0.5, reference_type = RewardReferenceType.SCORE),
        dict(name="exact_match", weight=0.5, reference_type = RewardReferenceType.MISTAKES),
    ]
    penalty_definition = [
        dict(name="dist_penalty", weight=0.25, reference_type = RewardReferenceType.SCORE),
        dict(name="exact_match", weight=0.5, reference_type = RewardReferenceType.MISTAKES),
    ]

    def __init__(self, llm_pipeline, context):
        sections = context.sections
        full_content = context.content
        self.context = context
        responses = []
        probability_true = random.random()

        
        system_prompt = HALLUCINATION_SYSTEM_PROMPT
        tool_schema = self.tool_schema_generator.get_schema(llm_pipeline)

        resp_tmp = None
        for header, section in sections.items():
            section = "\n".join([s for s in section])
            print("SECTION: ", section)

            num_claims_per_section = random.randint(1, 3)
            past_responses = []
            for _ in range(num_claims_per_section):
                values = [True, False]
                true_or_false = random.choices(values, weights = [probability_true, 1-probability_true])[0]
                print("TRUE or False prob: ", true_or_false)


                query_prompt = HALLUCINATION_PROMPT_TEMPLATE.format(
                    context=section,
                    hallucination_or_not=true_or_false, 
                    difficulty_rating=context.difficulty,
                    past_responses=". ".join([r.claim for r in responses])
                )

                response = self.generate_input(llm_pipeline, query_prompt, system_prompt, tool_schema)
                print(f"CLAIM: {response}")

                # format 
                try:
                    json_response = self.parse_llm_query(response)
                    resp_tmp = Config(
                        context = section,
                        claim = json_response['response'],
                        true_or_false = true_or_false
                    )
                    responses.append(resp_tmp)
                    past_responses.append(json_response['response'])
                except (JSONDecodeError, ValidationError) as e:
                    print(f"Experienced {e} in Hallucination task")
                    continue
                

        num_claims = len(responses)
        self.generate_reference(responses, num_claims, full_content)
        
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
        self.api = llm_pipeline.api.value
        self.model_id = llm_pipeline.model_id

    def generate_reference(self, responses: list[Config], num_claims: int, content: str):
        # context input 
        self.rag_context = content

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