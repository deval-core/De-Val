import pytest
import time
import asyncio
import sys
from functools import partial
import bittensor as bt
from deval.forward import run_step
from deval.base.validator import Validator
from deval.tasks import Task, QuestionAnsweringTask
from .fixtures.task import WIKI_CONTEXT
from deval.agent import HumanAgent
from unittest.mock import patch, Mock
from deval.protocol import EvalSynapse
from deval.tasks import TasksEnum

sys.argv = [__file__, "--mock", "--wandb.off", "--neuron.tasks", "qa"]
mock_neuron = Validator()

task = QuestionAnsweringTask(
    llm_pipeline=mock_neuron.llm_pipeline, context=WIKI_CONTEXT, create_reference=False
)


def generate_reference(x, delay=1):
    time.sleep(delay)
    return "Fake reference"


async def mock_dendrite_call(delay=1, **kwargs):
    time.sleep(delay)

    mock_synapse = EvalSynapse(
                Task = TasksEnum.HALLUCINATION,
                context_input = "",
                response = ""
            )
    mock_synapse.completion =1.0

    return [mock_synapse]


@pytest.mark.parametrize(
    "generate_reference_time, dendrite_time, expected_forward_time",
    [(0.5, 0.5, 0.5), (0.5, 0.4, 0.5), (0.4, 0.5, 0.5)],
)
def test_generate_reference_parallel_to_dendrite(
    generate_reference_time, dendrite_time, expected_forward_time
):
    task.generate_reference = partial(generate_reference, delay=generate_reference_time)
    mock_agent = HumanAgent(task, mock_neuron.llm_pipeline)

    mock_neuron.dendrite = partial(mock_dendrite_call, delay=dendrite_time)

    event = asyncio.run(run_step(mock_neuron, mock_agent, k=4, timeout=0.1))

    step_time = event["step_time"]
    reward_pipeline_time = sum(
        event[key] for key in event if key.endswith("batch_time")
    )
    network_and_reference_gen_time = step_time - reward_pipeline_time

    assert network_and_reference_gen_time == pytest.approx(
        expected_forward_time, abs=0.1
    )