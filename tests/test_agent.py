import pytest
from deval.tasks.task import Task
from deval.agent import HumanAgent, create_persona

from .fixtures.llm import mock_llm_pipeline
from .fixtures.task import CONTEXTS, TASKS


@pytest.mark.parametrize("task", TASKS)
def test_agent_creation_with_dataset_context(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(), task=task
    )
    assert agent is not None



@pytest.mark.parametrize("task", TASKS)
def test_agent_contains_task(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(), task=task
    )
    assert agent.task is not None



@pytest.mark.parametrize("task", TASKS)
def test_agent_can_make_challenges(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(),
        task=task,
    )
    assert agent.challenge is not None
   


@pytest.mark.parametrize("task", TASKS)
def test_agent_progress_is_zero_on_init(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(), task=task
    )
    assert agent.progress == 0


@pytest.mark.parametrize("task", TASKS)
def test_agent_progress_is_one_when_task_is_complete(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    task.complete = True
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(), task=task
    )
    assert agent.progress == 1


@pytest.mark.parametrize("task", TASKS)
def test_agent_finished_is_true_when_task_is_complete(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    task.complete = True
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(), task=task
    )
    assert agent.finished == True


@pytest.mark.parametrize("task", TASKS)
def test_agent_finished_is_false_when_task_is_not_complete(task: Task):
    context = CONTEXTS[task]
    task = task(llm_pipeline=mock_llm_pipeline(), context=context)
    task.complete = False
    agent = HumanAgent(
        llm_pipeline=mock_llm_pipeline(), task=task
    )
    assert agent.finished == False
