from deval.llms import OpenAIPipeline
from deval.task_generator import create_task
from deval.tasks import TasksEnum
import pandas as pd

# INIT Variables
task_name = TasksEnum.COMPLETENESS.value
num_to_generate = 10
data_path = "./exports"

def extractTaskToRow(task):
    name = task.name
    rag_context = task.rag_context
    query = task.query
    llm_response = task.llm_response
    reference = task.reference

    return [name, rag_context, query, llm_response, reference]


llm_pipeline = OpenAIPipeline(
    model_id="gpt-4o-mini",
    mock=False,
)  
rows = []

for i in range(num_to_generate):
    print(i)
    task = create_task(llm_pipeline, task_name)
    row = extractTaskToRow(task)
    rows.append(row)

df = pd.DataFrame(rows, columns= ['task', 'rag_context', 'query', 'llm_response', 'reference'])
df.to_csv(f"./scripts/exports/task_{task_name}.csv", index = False)