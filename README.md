# AI Researcher — MVP (Milestone 1)

An autonomous AI research agent, built incrementally. This is the MVP slice:
question → arXiv search → Claude summary + hypothesis → saved to SQLite →
printed report.

See `docs/phase0-architecture.md` for the full architecture, roadmap, and
what comes after this milestone.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and put your real ANTHROPIC_API_KEY in it
```

## Run

```bash
python scripts/run_research.py "How do transformers handle long context?"
```

This will:
1. Search arXiv for relevant papers
2. Ask Claude to summarize them and propose a follow-up hypothesis
3. Save the project, note, and hypothesis to `ai_researcher.db` (SQLite, gitignored)
4. Print a markdown-style report to the terminal

## What's not here yet

Code execution sandbox, experiment tracking, evaluation engine, critic
agent, hypothesis ranking, knowledge graph, multi-agent orchestration,
self-improvement loop. These are later milestones — see the architecture
doc for the full sequence. Nothing here silently does more than described
above.

## Project layout

```
agents/          # (empty — added when the Critic agent lands)
research/        # (empty — orchestration logic for later milestones)
literature/       arxiv_search.py — MVP literature backend
memory/          # (empty — reserved for the future knowledge graph)
experiments/     # (empty — Milestone 6+)
evaluation/      # (empty — Milestone 7+)
models/          # (empty — Pydantic schemas, added as needed)
tools/            llm_provider.py — LLM abstraction (Claude backend)
database/         models.py, db.py — SQLAlchemy schema + engine
security/        # (empty — sandbox config, added with Milestone 6)
tests/           # (empty — add tests alongside each new module)
configs/         # (empty)
scripts/          run_research.py — the CLI entrypoint
docs/             phase0-architecture.md
```
