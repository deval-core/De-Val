RELEVANCY_PROMPT = """\
Your goal is to determine if the provided LLM response is relevant to the user's query.\
    You should disregard whether the response is factually accurate or not and only be concerned with relevance. 

In this task, I will provide you with the following:
- User Query: the question asked by the user
- LLM Response: a response to the user derived from an input context


You should return a score of 0 if the response is not relevant and a score of 1 if the response is relevant.

# User Query
{query}

# LLM response
{llm_response}
 

Return your response if the format of "Response: score" where the score is your estimation on relevancy. Return no other text  
"""


HALLUCINATION_PROMPT = """\
Your goal is to determine if the provided LLM response is hallucinating given the provided RAG context.  \

In this task, I will provide you with the following:
- RAG Context: the provided context which will act as your source of truth
- LLM Response: a series of claims derived from the RAG context

You should return a score between 0 and 1 based on how accurate you perceive the claims to be. \
- If the response does not hallucinate at all then return a value of 1
- If the entire response is hallucinated then return a value of 0
- if half of the response is hallucinated then return a score of 0.5

# RAG Context
{rag_context}

# LLM Response
{llm_response}


Return your response if the format of "Response: score" where the score is your estimation on relevancy. Return no other text
"""

HALLUCINATION_MISTAKES_PROMPT = """\
Your goal is to extract the hallucinations from the provided respnse given the underlying RAG context if there are any.  \
Essentially, you must compare the response to the RAG context, determine if any of the claims in the response are false, and \
return back any false claims you identify.  

In this task, I will provide you with the following:
- RAG Context: the provided context which will act as your source of truth
- LLM Response: a series of claims derived from the RAG context

Requirements:
- You must return false claims verbatim as they appear in the LLM response
- You must return each false claim separated by a newline character 
- Do not return any other text unless you consider it to be false based on the provided RAG context

# RAG Context
{rag_context}

# LLM Response
{llm_response}

Do not return any other text beside the false claims separated by a newline character. 
"""



ATTRIBUTION_PROMPT = """\
Your goal is to determine if the provided LLM response is mis-attributing action items to the wrong person given the provided RAG context.  \

In this task, I will provide you with the following:
- RAG Context: the provided context which will act as your source of truth
- LLM Response: a series of summarized action items attributed to a participant derived from the RAG context

For example, if the RAG context says that Person A must achieve Task 1, but the LLM response incorrectly says that Person B must achieve Task 1 \
then this would be a misattribution

You should return a score between 0 and 1 based on how accurate you perceive the attributions to be. \
- If the response attributes action items with complete accuracy  then return a value of 1
- If the entire response is misattributed then return a value of 0
- if half of the response is misattributed then return a score of 0.5

# RAG Context
{rag_context}

# LLM Response
{llm_response}


Return your response if the format of "Response: score" where the score is your estimation on relevancy. Return no other text
"""

ATTRIBUTION_MISTAKES_PROMPT = """\
Your goal is to extract the misattributed action items from the provided respnse given the underlying RAG context if there are any.  \
Essentially, you must compare the LLM response to the RAG context, determine if any of the action items in the response are misattributed, and \
return back any misattributed action items you identify.  

For example, if the RAG context says that Person A must achieve Task 1, but the LLM response incorrectly says that Person B must achieve Task 1 \
then this would be a misattribution.  You must return this action item if it was misattributed.


In this task, I will provide you with the following:
- RAG Context: the provided context which will act as your source of truth
- LLM Response: a series of summarized action items attributed to a participant derived from the RAG context

Requirements:
- You must return misattributed action items verbatim as they appear in the LLM response
- You must return each misattributed action items separated by a newline character 
- Do not return any other text unless you consider it to be misattributed based on the provided RAG context

# RAG Context
{rag_context}

# LLM Response
{llm_response}

Do not return any other text beside the misattributed action items separated by a newline character. 
"""


SUMMARY_COMPLETENESS_PROMPT = """\
Your goal is to determine if the provided LLM response is a complete summary given the provided RAG context.  \

In this task, I will provide you with the following:
- RAG Context: the provided context which will act as your source of truth
- LLM Response: a series of claims derived from the RAG context


For example, if the RAG context contains important information that should be summarized then this would be considered an incomplete summary.

You should return a score between 0 and 1 based on how accurate you perceive the claims to be. \
- If the response is a perfect summary of the RAG context then return a value of 1
- If the response is missing all important information from RAG context then return a value of 0
- if the response is missing half of the important information from the RAG context then return a score of 0.5

# RAG Context
{rag_context}

# LLM Response
{llm_response}


Return your response if the format of "Response: score" where the score is your estimation on relevancy. Return no other text
"""


SUMMARY_MISTAKES_PROMPT = """\
Your goal is to identify any important topics in the provided RAG context that is not included in the LLM response.  \
Essentially, there is a possibility that the provided summary provided in the LLM response is missing key information, \
and it is your job to identify this missing information and then summarize it.  You must return a summarized version of any \
missing key information that you identified.

For example, if the RAG context contains important information that should be summarized then this would be considered an incomplete summary.

In this task, I will provide you with the following:
- RAG Context: the provided context which will act as your source of truth
- LLM Response: a series of claims derived from the RAG context

Requirements:
- You must return a summary of any key topics or important information from the RAG context that is not already present in LLM response
- Do not summarize any information that is already included in the LLM response. This would be considered a failure. 
- You must return each summary separated by a newline character 
- Do not return any other text unless you consider it to be a summary of missing information from RAG context.

# RAG Context
{rag_context}

# LLM Response
{llm_response}

Do not return any other text beside the summaries separated by a newline character. 
"""
