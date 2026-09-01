"""
End-to-end MVP research loop, runnable from the command line:

    python scripts/run_research.py "your research question here"

Flow: question -> arXiv search -> Claude summary/hypothesis -> saved to
SQLite -> printed as a markdown report. This is deliberately the entire
MVP loop — no Docker, no multi-agent orchestration, no knowledge graph.
Those are later milestones, built on top of this once it works.
"""

from __future__ import annotations

import sys

# Add the project root to the path so `database.*`, `literature.*`, and
# `tools.*` resolve when this script is run directly (rather than via
# `python -m`).
sys.path.insert(0, ".")

from database.db import SessionLocal, init_db
from database.models import Project, ResearchNote, Hypothesis
from literature.arxiv_search import search_papers
from tools.llm_provider import ClaudeProvider


def run(question: str) -> None:
    init_db()
    session = SessionLocal()
    llm = ClaudeProvider()

    print(f"\nResearch question: {question}")

    print("Searching arXiv...")
    papers = search_papers(question, max_results=5)
    if not papers:
        print("No papers found — try rephrasing the question.")
        return

    # Build the LLM prompt from real retrieved abstracts only, so the
    # summary is grounded in what was actually found, not the model's
    # general knowledge.
    papers_block = "\n\n".join(
        f"Title: {p.title}\nAuthors: {', '.join(p.authors)}\nAbstract: {p.abstract}"
        for p in papers
    )
    prompt = (
        f"Research question: {question}\n\n"
        f"Here are {len(papers)} papers found via arXiv search:\n\n"
        f"{papers_block}\n\n"
        "Based only on these papers, write:\n"
        "1. A short summary (3-5 sentences) of the current state of this area.\n"
        "2. One concrete, testable hypothesis for a follow-up research direction.\n"
        "Label the two sections clearly as SUMMARY and HYPOTHESIS."
    )

    print("Asking Claude to summarize and propose a hypothesis...")
    reply = llm.generate(prompt, max_tokens=800)

    # Naive split on the labels the prompt asked for. Good enough for the
    # MVP; a later milestone can ask for structured JSON output instead.
    summary_text = reply
    hypothesis_text = ""
    if "HYPOTHESIS" in reply:
        parts = reply.split("HYPOTHESIS", 1)
        summary_text = parts[0].replace("SUMMARY", "").strip()
        hypothesis_text = parts[1].lstrip(":").strip()

    project = Project(name=question[:250], description="Auto-created by run_research.py")
    session.add(project)
    session.flush()  # assigns project.id without committing yet

    session.add(ResearchNote(project_id=project.id, text=summary_text))
    if hypothesis_text:
        session.add(Hypothesis(project_id=project.id, text=hypothesis_text))

    session.commit()
    session.close()

    print("\nSaved to ai_researcher.db\n")
    print("=" * 60)
    print(f"# Research Report: {question}\n")
    print("## Summary\n")
    print(summary_text)
    if hypothesis_text:
        print("\n## Proposed Hypothesis\n")
        print(hypothesis_text)
    print("\n## Sources\n")
    for p in papers:
        print(f"- {p.title} ({p.url})")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_research.py "your research question"')
        sys.exit(1)
    run(" ".join(sys.argv[1:]))
