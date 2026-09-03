"""
Tests for agents/result_critic.py.

This critic exists because of one real failure: an experiment reported
8463 seconds for a 13-second loop and every automated check passed. So
the tests focus on the critic refusing to wave things through.
"""

from __future__ import annotations

import pytest

from agents.result_critic import (
    CHECKLIST,
    VERDICT_INVALID,
    VERDICT_TRUSTWORTHY,
    review_result,
)
from tests.conftest import FakeLLM


def _reply(verdict="trustworthy", **overrides):
    reply = {key: True for key in CHECKLIST}
    reply.update({"verdict": verdict, "reasoning": "looks fine", "suspected_bug": ""})
    reply.update(overrides)
    return reply


def test_a_clean_result_is_trusted():
    llm = FakeLLM([_reply()])
    c = review_result(llm, "m", "code", "out", {"acc": 0.9})
    assert c.is_trustworthy
    assert c.failed_checks == []


def test_a_measurement_bug_is_reported_with_its_cause():
    llm = FakeLLM([
        _reply(
            verdict="invalid",
            magnitude_is_plausible=False,
            no_obvious_measurement_bug=False,
            suspected_bug="timer started outside the loop",
        )
    ])
    c = review_result(llm, "m", "code", "out", {"seconds": 8463.0})

    assert not c.is_trustworthy
    assert c.verdict == VERDICT_INVALID
    assert "timer" in c.suspected_bug
    assert set(c.failed_checks) == {"magnitude_is_plausible", "no_obvious_measurement_bug"}


def test_an_unrecognised_verdict_raises_rather_than_trusting_the_result():
    # A broken reviewer that defaults to "trustworthy" is worse than none.
    llm = FakeLLM([_reply(verdict="seems ok to me")])
    with pytest.raises(ValueError, match="unrecognised verdict"):
        review_result(llm, "m", "code", "out", {"acc": 0.9})


def test_the_critic_is_shown_the_code_not_just_the_numbers():
    # A misplaced timer is invisible unless you read the loop it sits in.
    llm = FakeLLM([_reply()])
    review_result(llm, "m", "UNIQUE_CODE_MARKER", "out", {"acc": 0.9})
    assert "UNIQUE_CODE_MARKER" in llm.calls[0]["prompt"]


def test_the_system_prompt_treats_impressive_results_as_suspect():
    llm = FakeLLM([_reply()])
    review_result(llm, "m", "code", "out", {"acc": 0.9})
    assert "suspect" in llm.calls[0]["system"].lower()


def test_verdict_casing_is_tolerated():
    llm = FakeLLM([_reply(verdict="  TRUSTWORTHY ")])
    assert review_result(llm, "m", "c", "o", {}).verdict == VERDICT_TRUSTWORTHY
