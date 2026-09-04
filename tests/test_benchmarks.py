"""
Tests for the benchmark suite and the self-improvement loop.

The model is faked throughout. What is under test is the scoring and the
accept/reject decision, and both must be deterministic — a benchmark
whose own arithmetic is unreliable measures nothing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from benchmarks.improve import (
    ACCEPTED,
    REJECTED,
    Attempt,
    evaluate_change,
    propose_prompt_change,
)
from benchmarks.runner import SuiteScore, score_bug_detection, score_trends
from benchmarks.tasks import BUG_TASKS, CODING_TASKS, TREND_TASKS
from tests.conftest import FakeLLM


# ---------- task definitions ----------

def test_bug_and_trend_suites_are_half_clean():
    """
    A critic that flags everything catches every bug. Unless clean cases
    are scored too, that critic looks perfect.
    """
    assert sum(t.has_bug for t in BUG_TASKS) == sum(not t.has_bug for t in BUG_TASKS)
    assert sum(t.is_inconsistent for t in TREND_TASKS) == sum(
        not t.is_inconsistent for t in TREND_TASKS
    )


def test_an_exact_coding_task_rejects_a_near_miss():
    # Tolerance defaults to 0. An exact answer with a tolerance would
    # pass on a wrong one.
    task = next(t for t in CODING_TASKS if t.task_id == "sum_1_to_100")
    assert task.passed({"total": 5050.0})
    assert not task.passed({"total": 5051.0})


def test_a_missing_metric_fails_the_task():
    task = CODING_TASKS[0]
    assert not task.passed({"something_else": 5050.0})


# ---------- scoring ----------

def test_the_trend_suite_scores_itself_without_a_model():
    # Arithmetic, so it must be perfect and identical every time.
    first = score_trends()
    second = score_trends()
    assert first.mean == second.mean == 1.0


def test_a_critic_that_flags_everything_does_not_score_well():
    # The whole point of including clean cases.
    always_flag = FakeLLM([
        {"verdict": "invalid", "reasoning": "r", "suspected_bug": "b"}
    ] * len(BUG_TASKS))

    score = score_bug_detection(always_flag, repeats=1)

    assert score.mean == 0.5
    assert score.detail["false_alarms"] == "3/3"


def test_a_critic_that_flags_nothing_scores_the_same_as_one_that_flags_everything():
    never_flag = FakeLLM([
        {"verdict": "trustworthy", "reasoning": "r", "suspected_bug": ""}
    ] * len(BUG_TASKS))

    score = score_bug_detection(never_flag, repeats=1)

    assert score.mean == 0.5
    assert score.detail["caught"] == "0/3"


def test_spread_is_reported_across_repeats():
    # A single pass measures luck as much as capability, so the spread
    # has to be visible.
    score = SuiteScore("s", runs=[1.0, 0.5])
    assert score.spread > 0

    steady = SuiteScore("s", runs=[0.833, 0.833, 0.833])
    assert steady.spread == 0


# ---------- self-improvement ----------

def _fake_score(mean):
    return SuiteScore("bug_detection", runs=[mean])


def test_a_change_that_does_not_help_is_rejected():
    """
    The real first run: the model proposed restating an instruction the
    prompt already contained. 83.3% -> 83.3%. It reads as sensible and
    changes nothing, which is exactly what this loop exists to catch.
    """
    llm = FakeLLM([])

    with patch("benchmarks.improve.score_bug_detection", return_value=_fake_score(0.833)):
        attempt = evaluate_change(llm, _fake_score(0.833), "restated the rule", "new prompt")

    assert attempt.verdict == REJECTED
    assert attempt.delta == 0


def test_a_small_gain_is_rejected_as_noise():
    # Without a margin, noise in one direction gets recorded as progress.
    llm = FakeLLM([])

    with patch("benchmarks.improve.score_bug_detection", return_value=_fake_score(0.85)):
        attempt = evaluate_change(llm, _fake_score(0.833), "tiny tweak", "new prompt")

    assert attempt.verdict == REJECTED
    assert "noise" in attempt.reason


def test_a_real_gain_is_accepted():
    llm = FakeLLM([])

    with patch("benchmarks.improve.score_bug_detection", return_value=_fake_score(1.0)):
        attempt = evaluate_change(llm, _fake_score(0.833), "real fix", "new prompt")

    assert attempt.verdict == ACCEPTED


def test_a_regression_is_rejected():
    llm = FakeLLM([])

    with patch("benchmarks.improve.score_bug_detection", return_value=_fake_score(0.5)):
        attempt = evaluate_change(llm, _fake_score(0.833), "made it worse", "new prompt")

    assert attempt.verdict == REJECTED
    assert attempt.delta < 0


def test_the_original_prompt_is_restored_even_when_the_benchmark_crashes():
    """
    A rejected change that survives a crash silently degrades the system
    the loop is meant to improve.
    """
    import agents.result_critic as critic_module

    original = critic_module.SYSTEM_PROMPT
    llm = FakeLLM([])

    with patch("benchmarks.improve.score_bug_detection", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            evaluate_change(llm, _fake_score(0.833), "d", "REPLACEMENT PROMPT")

    assert critic_module.SYSTEM_PROMPT == original


def test_a_proposal_with_no_prompt_raises():
    # "The model proposed nothing" must not look like a successful
    # proposal of an empty prompt.
    llm = FakeLLM([{"description": "I have no ideas"}])

    with pytest.raises(ValueError, match="no new prompt"):
        propose_prompt_change(llm, "current", _fake_score(0.833), ["a failure"])


def test_the_proposer_is_shown_the_failures():
    llm = FakeLLM([{"description": "d", "new_prompt": "p"}])

    propose_prompt_change(llm, "current", _fake_score(0.833), ["UNIQUE_FAILURE_MARKER"])

    assert "UNIQUE_FAILURE_MARKER" in llm.calls[0]["prompt"]


def test_the_proposer_is_told_not_to_just_flag_more():
    # The easy way to raise "caught" is to flag everything, which the
    # false-alarm count would then punish.
    llm = FakeLLM([{"description": "d", "new_prompt": "p"}])

    propose_prompt_change(llm, "current", _fake_score(0.833), [])

    assert "flags everything" in llm.calls[0]["system"]
