from deval.task_repository import TaskRepository
from deval.tasks.task import TasksEnum
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import numpy as np

_ = load_dotenv(find_dotenv())

# INIT Variables
task_name = TasksEnum.HALLUCINATION.value
num_to_generate = 15
data_path = "./exports"

def extractTaskToRow(task, llm_pipeline):
    name = task.name
    rag_context = task.rag_context
    query = task.query
    llm_response = task.llm_response
    reference_score = task.reference
    api = llm_pipeline.api
    model_id = llm_pipeline.model_id
    reference_mistakes = task.reference_mistakes
    reference_true_values = task.reference_true_values

    return [name, api, model_id, rag_context, query, llm_response, reference_score, reference_mistakes, reference_true_values]


allowed_models = ["gpt-4o-mini"]
task_generator = TaskRepository(allowed_models=allowed_models)
rows =[]
for i in range(num_to_generate):
    print(i)

    llm_pipeline = task_generator.get_random_llm()
    print(f"Model ID: {llm_pipeline.model_id}")
 
    try:
        task = task_generator.create_task(llm_pipeline, task_name)
        print(task)
    except Exception as e:
        print(e) 
        continue

    row = extractTaskToRow(task, llm_pipeline)
    rows.append(row)

df = pd.DataFrame(rows, columns= ['task', "api", "model_id", 'rag_context', 'query', 'llm_response', 'reference', "reference_mistakes", "reference_true_values"])
df.to_csv(f"./scripts/exports/task_{task_name}.csv", index = False)