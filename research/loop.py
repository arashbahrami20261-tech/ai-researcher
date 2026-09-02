"""
The research loop — orchestration.

Moved out of `scripts/run_research.py` so the loop can be tested, imported,
and later called from a FastAPI endpoint without dragging along CLI argument
parsing and printing. `scripts/run_research.py` is now a thin wrapper whose
only job is turning command-line arguments into a call to `run_research`.

Current flow (Milestones 1-5):
    question -> arXiv search -> structured summary + hypothesis
             -> critic review -> persist everything -> return a report
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.critic import Critique, critique_to_json, review_hypothesis
from database.models import Critique as CritiqueRow
from database.models import Hypothesis, Paper as PaperRow, Project, ResearchNote
from literature.arxiv_search import Paper, search_papers
from tools.llm_provider import LLMProvider

GENERATOR_SYSTEM_PROMPT = (
    "You are a careful AI research assistant. Base every claim only on the "
    "papers provided. If the papers do not support a claim, do not make it. "
    "Never invent citations, results, or numbers."
)


@dataclass
class ResearchResult:
    """Everything one pass of the loop produced. Returned, not printed."""

    question: str
    papers: list[Paper]
    summary: str
    hypothesis: str
    critique: Critique | None
    critique_error: str | None = None
    project_id: int | None = None


def render_papers(papers: list[Paper]) -> str:
    """Format retrieved papers for inclusion in a prompt."""
    return "\n\n".join(
        f"Title: {p.title}\n"
        f"Authors: {', '.join(p.authors)}\n"
        f"Published: {p.published}\n"
        f"URL: {p.url}\n"
        f"Abstract: {p.abstract}"
        for p in papers
    )


def generate_summary_and_hypothesis(
    llm: LLMProvider, question: str, papers_block: str, paper_count: int
) -> tuple[str, str]:
    """
    Ask the model for a literature summary and one testable hypothesis.

    Uses `generate_structured` rather than parsing labels out of prose. The
    previous version split the reply on the literal string "HYPOTHESIS",
    which silently produced an empty hypothesis whenever the model formatted
    the heading differently.
    """
    prompt = (
        f"Research question: {question}\n\n"
        f"Here are {paper_count} papers found via arXiv search:\n\n"
        f"{papers_block}\n\n"
        "Based only on these papers, return JSON with exactly this shape:\n"
        "{\n"
        '  "summary": "3-5 sentences on the current state of this area",\n'
        '  "hypothesis": "one concrete, testable follow-up hypothesis"\n'
        "}"
    )

    reply = llm.generate_structured(prompt, system=GENERATOR_SYSTEM_PROMPT, max_tokens=1000)

    summary = str(reply.get("summary", "")).strip()
    hypothesis = str(reply.get("hypothesis", "")).strip()

    if not summary:
        raise ValueError("Model returned no summary.")
    if not hypothesis:
        raise ValueError("Model returned no hypothesis.")

    return summary, hypothesis


def run_research(
    llm: LLMProvider,
    session,
    question: str,
    max_papers: int = 5,
    run_critic: bool = True,
) -> ResearchResult:
    """
    Run one full pass of the research loop and persist the results.

    `llm` and `session` are passed in rather than constructed here, so tests
    can inject a fake provider and an in-memory database. This is the whole
    reason the LLM abstraction layer exists.

    Raises `LookupError` if the literature search found nothing — an empty
    result is a real outcome the caller must handle, not something to paper
    over by letting the model answer from memory instead.
    """
    papers = search_papers(question, max_results=max_papers)
    if not papers:
        raise LookupError(f"arXiv returned no papers for: {question!r}")

    papers_block = render_papers(papers)
    summary, hypothesis = generate_summary_and_hypothesis(
        llm, question, papers_block, len(papers)
    )

    critique: Critique | None = None
    critique_error: str | None = None
    if run_critic:
        try:
            critique = review_hypothesis(llm, question, summary, hypothesis, papers_block)
        except ValueError as exc:
            # A critic that fails to produce a parseable verdict must not be
            # treated as an approval. Record the failure and carry it through
            # to the report so the result is visibly unreviewed.
            critique_error = str(exc)

    project = _persist(session, question, papers, summary, hypothesis, critique)

    return ResearchResult(
        question=question,
        papers=papers,
        summary=summary,
        hypothesis=hypothesis,
        critique=critique,
        critique_error=critique_error,
        project_id=project.id,
    )


def _persist(
    session,
    question: str,
    papers: list[Paper],
    summary: str,
    hypothesis: str,
    critique: Critique | None,
) -> Project:
    """
    Write one research pass to the database in a single transaction.

    Papers are stored too (they weren't before), so a result can be traced
    back to the exact evidence it was built on months later — the spec's
    provenance requirement.
    """
    project = Project(name=question[:250], description="Created by research.loop.run_research")
    session.add(project)
    session.flush()  # assigns project.id without committing

    for p in papers:
        session.add(
            PaperRow(
                title=p.title[:500],
                authors=", ".join(p.authors),
                abstract=p.abstract,
                url=p.url[:500],
                published=p.published[:50],
                source=p.source,
            )
        )

    note = ResearchNote(project_id=project.id, text=summary)
    session.add(note)

    hypothesis_row = Hypothesis(project_id=project.id, text=hypothesis)
    # A rejected hypothesis is recorded with that status rather than stored
    # as if it had passed review. Failed directions are useful memory.
    if critique is not None and not critique.is_accepted:
        hypothesis_row.status = critique.verdict
    session.add(hypothesis_row)
    session.flush()

    if critique is not None:
        session.add(
            CritiqueRow(
                target_type="hypothesis",
                target_id=hypothesis_row.id,
                verdict=critique.verdict,
                checklist_json=critique_to_json(critique),
                reasoning=critique.reasoning,
            )
        )

    session.commit()
    return project


def format_report(result: ResearchResult) -> str:
    """Render a result as a markdown report. Pure string building, no I/O."""
    lines = [
        f"# Research Report: {result.question}",
        "",
        "## Summary",
        "",
        result.summary,
        "",
        "## Proposed Hypothesis",
        "",
        result.hypothesis,
        "",
    ]

    if result.critique is not None:
        c = result.critique
        lines += ["## Critic Review", "", f"**Verdict:** {c.verdict.upper()}", ""]
        for key, value in c.checklist.items():
            lines.append(f"- {key}: {'pass' if value else 'FAIL'}")
        if c.reasoning:
            lines += ["", f"**Reasoning:** {c.reasoning}"]
        if c.limitations:
            lines += ["", f"**Limitations:** {c.limitations}"]
        lines.append("")
    elif result.critique_error:
        lines += [
            "## Critic Review",
            "",
            f"**NOT REVIEWED** — the critic failed: {result.critique_error}",
            "",
            "Treat the hypothesis above as unreviewed.",
            "",
        ]

    lines += ["## Sources", ""]
    lines += [f"- {p.title} ({p.url})" for p in result.papers]

    return "\n".join(lines)
