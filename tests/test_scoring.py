import pytest
from datetime import datetime
from deval.rewards import (
    FloatDiffModel,
    RewardPipeline,
)



completion = ["0.5", "1/2", "1-0.5", "2*0.25"]
expected_result = [1.0, 1.0, 1.0, 1.0]
reference = ["0.5"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing(reference, completion, expected_result):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["1e3", "-1e3", "1e-3", "-1e-3"]
expected_result = [1.0, 0.0, 0.0, 0.0]
reference = ["1000"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_exponents(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["1.0.", "1.0", "1.0.0", "1,", "0 1"]
expected_result = [1.0, 1.0, 0.0, 1.0, 1.0]
reference = ["1.0"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_punctuation(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["-20", "-23", "23", "20", "1000", "2*10+3"]
expected_result = [0.0, 0.0, 1.0, 0.8695652173918714, 0.0, 1.0]
reference = ["23"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_negative_numbers(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result


completion = ["0", "0.001", "-0.0", "-0.001", "0.0001"]
expected_result = [1.0, 0.0, 1.0, 0.0, 0.0]
reference = ["0"] * len(completion)


@pytest.mark.parametrize("reference", reference)
@pytest.mark.parametrize(
    "completion, expected_result", zip(completion, expected_result)
)
def test_math_score_expression_parsing_with_zeros(
    reference, completion, expected_result
):
    score = FloatDiffModel().math_score(reference, completion)
    assert score == expected_result
