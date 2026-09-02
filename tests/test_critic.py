"""
Tests for agents/critic.py.

The most important test here is `test_unrecognised_verdict_raises`: a broken
critic must fail loudly, never default to approving the result.
"""

from __future__ import annotations

import pytest

from agents.critic import (
    LITERATURE_STAGE_CHECKLIST,
    VERDICT_REJECTED,
    review_hypothesis,
)
from tests.conftest import FakeLLM


def _reply(verdict="accepted", **overrides):
    reply = {key: True for key in LITERATURE_STAGE_CHECKLIST}
    reply.update(
        {
            "verdict": verdict,
            "reasoning": "seems fine",
            "limitations": "small scale only",
        }
    )
    reply.update(overrides)
    return reply


def test_accepted_verdict_is_parsed():
    llm = FakeLLM([_reply()])

    critique = review_hypothesis(llm, "q", "summary", "hypothesis", "papers")

    assert critique.is_accepted
    assert critique.limitations == "small scale only"
    assert critique.failed_checks == []


def test_rejected_verdict_reports_failed_checks():
    llm = FakeLLM([_reply(verdict="rejected", is_hypothesis_testable=False)])

    critique = review_hypothesis(llm, "q", "summary", "hypothesis", "papers")

    assert not critique.is_accepted
    assert critique.verdict == VERDICT_REJECTED
    assert critique.failed_checks == ["is_hypothesis_testable"]


def test_unrecognised_verdict_raises_instead_of_defaulting_to_accepted():
    llm = FakeLLM([_reply(verdict="looks good to me!")])

    with pytest.raises(ValueError, match="unrecognised verdict"):
        review_hypothesis(llm, "q", "summary", "hypothesis", "papers")


def test_verdict_casing_and_whitespace_are_tolerated():
    llm = FakeLLM([_reply(verdict="  ACCEPTED ")])

    critique = review_hypothesis(llm, "q", "summary", "hypothesis", "papers")

    assert critique.is_accepted


def test_critic_is_shown_the_retrieved_papers():
    # If the critic can't see the evidence, it cannot tell a grounded claim
    # from an invented one — so this is a correctness requirement, not a
    # cosmetic detail about prompt contents.
    llm = FakeLLM([_reply()])

    review_hypothesis(llm, "q", "summary", "hypothesis", "UNIQUE_PAPER_MARKER")

    assert "UNIQUE_PAPER_MARKER" in llm.calls[0]["prompt"]


def test_critic_system_prompt_instructs_rejection():
    llm = FakeLLM([_reply()])

    review_hypothesis(llm, "q", "summary", "hypothesis", "papers")

    assert "reject" in llm.calls[0]["system"].lower()


def test_missing_checklist_keys_are_simply_absent():
    # A partially-answered checklist should not crash; the missing keys just
    # don't appear, which is visible in the report rather than being faked.
    llm = FakeLLM([{"verdict": "revise", "reasoning": "needs work"}])

    critique = review_hypothesis(llm, "q", "summary", "hypothesis", "papers")

    assert critique.checklist == {}
    assert critique.verdict == "revise"
