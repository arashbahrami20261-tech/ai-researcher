"""
Tests for agents/hypothesis.py — the module that closes the research loop.
"""

from __future__ import annotations

import pytest

from agents.hypothesis import Proposal, generate_followups
from tests.conftest import FakeLLM

FULL_SCORES = {
    "expected_information_gain": 4,
    "novelty": 3,
    "feasibility_at_small_scale": 5,
}


def _entry(name, cost=1, gain=4):
    return {
        "hypothesis": f"hypothesis {name}",
        "methodology": f"method {name}",
        "rationale": "because",
        "expected_information_gain": gain,
        "novelty": 3,
        "feasibility_at_small_scale": 5,
        "cost": cost,
    }


def test_cost_counts_against_a_proposal():
    # Summing every criterion made an expensive experiment outrank a cheap
    # one of identical merit. Cost is a drawback, not a merit.
    cheap = Proposal("h", "m", "r", {**FULL_SCORES, "cost": 1})
    pricey = Proposal("h", "m", "r", {**FULL_SCORES, "cost": 5})
    assert cheap.total_score > pricey.total_score


def test_proposals_come_back_ranked_best_first():
    llm = FakeLLM([{"proposals": [_entry("weak", gain=1), _entry("strong", gain=5)]}])
    props = generate_followups(llm, "q", "h", "m", {"acc": 0.9})
    assert props[0].hypothesis == "hypothesis strong"


def test_previous_failures_are_passed_to_the_model():
    # The spec calls failed experiments valuable knowledge; the cheapest
    # form of that value is not repeating them.
    llm = FakeLLM([{"proposals": [_entry("a")]}])
    generate_followups(
        llm, "q", "h", "m", {"acc": 0.9},
        previous_failures=["UNIQUE_FAILURE_MARKER"],
    )
    assert "UNIQUE_FAILURE_MARKER" in llm.calls[0]["prompt"]


def test_no_failures_means_no_failure_section():
    llm = FakeLLM([{"proposals": [_entry("a")]}])
    generate_followups(llm, "q", "h", "m", {"acc": 0.9})
    assert "already tried and failed" not in llm.calls[0]["prompt"]


def test_the_measured_numbers_reach_the_model():
    # Proposals must be grounded in the actual result, not invented.
    llm = FakeLLM([{"proposals": [_entry("a")]}])
    generate_followups(llm, "q", "h", "m", {"accuracy": 0.8734})
    assert "0.8734" in llm.calls[0]["prompt"]


def test_proposals_without_a_methodology_are_dropped():
    # A hypothesis with no method is not actionable.
    llm = FakeLLM([{"proposals": [
        {"hypothesis": "vague idea", "rationale": "why not"},
        _entry("usable"),
    ]}])
    props = generate_followups(llm, "q", "h", "m", {"acc": 0.9})
    assert len(props) == 1
    assert props[0].hypothesis == "hypothesis usable"


def test_count_is_respected():
    llm = FakeLLM([{"proposals": [_entry("a"), _entry("b"), _entry("c")]}])
    assert len(generate_followups(llm, "q", "h", "m", {"acc": 0.9}, count=2)) == 2


def test_a_reply_with_no_usable_proposals_raises():
    # "The model proposed nothing" and "the reply was broken" are
    # different situations; neither should look like an empty success.
    llm = FakeLLM([{"proposals": [{"hypothesis": "", "methodology": ""}]}])
    with pytest.raises(ValueError, match="No usable proposals"):
        generate_followups(llm, "q", "h", "m", {"acc": 0.9})


def test_a_missing_proposals_key_raises():
    llm = FakeLLM([{"something_else": []}])
    with pytest.raises(ValueError):
        generate_followups(llm, "q", "h", "m", {"acc": 0.9})


def test_the_prompt_demands_distinct_scores():
    # The 7B model gave all three proposals identical scores, which made
    # the ranking meaningless until the prompt said otherwise.
    llm = FakeLLM([{"proposals": [_entry("a")]}])
    generate_followups(llm, "q", "h", "m", {"acc": 0.9})
    assert "NOT be identical" in llm.calls[0]["prompt"]


def test_the_next_cycle_is_told_to_keep_the_metric_names():
    """
    Metric names must survive into the follow-up.

    On a live run the model renamed linear_seconds to linear_search_time
    between cycles. Each series then held a single point, and the trend
    check went silent — a series of one cannot contradict anything. The
    check was not broken; it was starved.
    """
    llm = FakeLLM([{"proposals": [_entry("a")]}])

    generate_followups(llm, "q", "h", "m", {"linear_seconds": 1.8, "binary_seconds": 0.001})

    prompt = llm.calls[0]["prompt"]
    assert "linear_seconds" in prompt
    assert "Do not rename" in prompt
