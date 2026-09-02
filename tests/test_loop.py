"""
Tests for research/loop.py — the end-to-end loop, with arXiv mocked and a
fake LLM injected. This is the closest thing to an integration test the
suite has without spending money or needing network access.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.critic import LITERATURE_STAGE_CHECKLIST
from database.models import Critique as CritiqueRow
from database.models import Hypothesis, Paper as PaperRow, Project, ResearchNote
from literature.arxiv_search import Paper
from research.loop import format_report, run_research
from tests.conftest import FakeLLM

FAKE_PAPERS = [
    Paper(
        title="A Paper About Attention",
        authors=["Ada Lovelace"],
        abstract="We study attention.",
        url="http://arxiv.org/abs/1234.5678",
        published="2026-01-01T00:00:00Z",
    )
]

GENERATOR_REPLY = {"summary": "The field is active.", "hypothesis": "X improves Y."}


def _critic_reply(verdict="accepted"):
    reply = {key: True for key in LITERATURE_STAGE_CHECKLIST}
    reply.update({"verdict": verdict, "reasoning": "ok", "limitations": "none"})
    return reply


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_run_research_persists_everything(_mock_search, session):
    llm = FakeLLM([GENERATOR_REPLY, _critic_reply()])

    result = run_research(llm, session, "how does attention work?")

    assert result.summary == "The field is active."
    assert result.hypothesis == "X improves Y."
    assert result.critique.is_accepted

    # Every part of the pass should be traceable in the database afterwards.
    assert session.query(Project).count() == 1
    assert session.query(ResearchNote).count() == 1
    assert session.query(Hypothesis).count() == 1
    assert session.query(PaperRow).count() == 1
    assert session.query(CritiqueRow).count() == 1


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_rejected_hypothesis_is_stored_with_rejected_status(_mock_search, session):
    llm = FakeLLM([GENERATOR_REPLY, _critic_reply(verdict="rejected")])

    run_research(llm, session, "a question")

    hypothesis = session.query(Hypothesis).one()
    assert hypothesis.status == "rejected"


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_accepted_hypothesis_keeps_proposed_status(_mock_search, session):
    llm = FakeLLM([GENERATOR_REPLY, _critic_reply()])

    run_research(llm, session, "a question")

    assert session.query(Hypothesis).one().status == "proposed"


@patch("research.loop.search_papers", return_value=[])
def test_empty_search_raises_instead_of_asking_the_model_anyway(_mock_search, session):
    llm = FakeLLM([GENERATOR_REPLY])

    with pytest.raises(LookupError):
        run_research(llm, session, "a question with no results")

    # Nothing should have been written for a failed pass.
    assert session.query(Project).count() == 0


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_critic_failure_is_surfaced_not_swallowed(_mock_search, session):
    # Critic returns an unusable verdict. The run should still complete, but
    # the report must say the result was NOT reviewed.
    llm = FakeLLM([GENERATOR_REPLY, {"verdict": "maybe?"}])

    result = run_research(llm, session, "a question")

    assert result.critique is None
    assert result.critique_error is not None
    assert "NOT REVIEWED" in format_report(result)
    # No critique row should be written when there was no valid verdict.
    assert session.query(CritiqueRow).count() == 0


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_missing_hypothesis_in_model_reply_raises(_mock_search, session):
    llm = FakeLLM([{"summary": "only a summary"}])

    with pytest.raises(ValueError, match="no hypothesis"):
        run_research(llm, session, "a question")


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_report_includes_sources_and_verdict(_mock_search, session):
    llm = FakeLLM([GENERATOR_REPLY, _critic_reply()])

    report = format_report(run_research(llm, session, "a question"))

    assert "A Paper About Attention" in report
    assert "ACCEPTED" in report
    assert "http://arxiv.org/abs/1234.5678" in report


@patch("research.loop.search_papers", return_value=FAKE_PAPERS)
def test_critic_can_be_skipped(_mock_search, session):
    llm = FakeLLM([GENERATOR_REPLY])  # only one reply — critic must not be called

    result = run_research(llm, session, "a question", run_critic=False)

    assert result.critique is None
    assert result.critique_error is None
