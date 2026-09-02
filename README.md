# AI Researcher

An autonomous AI research agent, built incrementally in milestones.

Current state: **question → arXiv search → summary + hypothesis → critic
review → saved to database → markdown report.** Verified end-to-end against
a local model.

See `docs/phase0-architecture.md` for the full architecture and roadmap.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and put your real ANTHROPIC_API_KEY in it

alembic upgrade head
```

The `alembic upgrade head` step creates the database schema. It is not
optional — the app refuses to run against an unmigrated database rather
than silently creating tables that later can't be altered.

## Run

```bash
# Against Anthropic's API (needs ANTHROPIC_API_KEY in .env):
python scripts/run_research.py "How do transformers handle long context?"

# Against a local Ollama model (no API key, no cost):
#   ollama serve          # in a separate terminal
#   ollama pull qwen2.5:7b
python scripts/run_research.py --provider ollama "How do transformers handle long context?"
```

## Tests

```bash
pytest              # 33 tests, no network, no API key needed
pytest -m live      # additionally hits the real arXiv API
```

Every test injects a fake LLM provider, so the suite never spends money
and never depends on a network connection.

## Milestone status

| # | Milestone | Status |
|---|---|---|
| 0 | Architecture proposal | done |
| 1 | Repo scaffold + LLM abstraction | done |
| 2 | Literature retrieval (arXiv) | done |
| 3 | Persistent memory (SQLite + migrations) | done |
| 4 | Basic research loop | done |
| 5 | Critic agent | done |
| 6 | Coding agent + Docker sandbox | not started |
| 7 | Experiment tracking + evaluation engine | not started |
| 8 | Hypothesis generation from results | not started |
| 9 | Knowledge graph | not started |
| 10 | Multi-agent orchestration, benchmarking, self-improvement | not started |

## Known limitations

- The Anthropic path has still not been run with a real API key. The loop
  has been verified end-to-end using a local Ollama model instead
  (`qwen2.5:7b`), which is what the `LLMProvider` abstraction was built to
  allow. `ClaudeProvider`'s default model string may also be stale.
- A local 7B model is much weaker than a frontier model. It produces usable
  summaries and hypotheses, but expect vaguer reasoning.
- The critic reviews hypotheses only. The experiment-stage checklist (data
  leakage, sample size, statistical significance, baseline choice) is
  defined in `agents/critic.py` but deliberately unused until Milestone 7,
  when there are experiments to check it against.
- `experiments` and `metrics` tables exist but nothing writes to them yet.
- SQLite, single user, no API layer. Postgres is a one-line change to
  `DATABASE_URL` in `database/db.py`.

## What the first live runs found

Running the loop for real surfaced two retrieval bugs that 31 passing tests
had not, because every test mocked arXiv's *reply* and none checked the
*request* being sent:

1. `search_papers` defaulted to `sort_by_newest=True`. Every caller
   inherited that without choosing it. arXiv receives hundreds of papers a
   day, so date-sorting returned whatever was posted that morning — a
   question about transformers came back with superconductor thermodynamics
   and vascular image segmentation. Fixed by defaulting to arXiv's
   relevance ranking; recency is still available on request.

2. The query used an unrestricted `all:` prefix across all of arXiv. A
   question about *long* context returned papers on long paths in hypercube
   subgraphs: the word matched, the field did not. Fixed by restricting to
   `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.NE`.

Relevant sources went from 0/5 to 5/5 across three runs. Both fixes have
regression tests that assert on the outgoing request.

The critic behaved as designed throughout: it rejected the first two runs
for hypotheses not grounded in the retrieved papers, and passed that check
on the third while still returning REVISE — the hypothesis named two
mechanisms to combine without specifying how, which is not yet testable.

## Project layout

```
agents/           critic.py — the Critic agent (Milestone 5)
research/         loop.py — the research loop orchestration
literature/       arxiv_search.py — literature backend
tools/            llm_provider.py — LLM abstraction (Claude + Ollama)
database/         models.py, db.py — SQLAlchemy schema + engine
migrations/       Alembic migration history
tests/            conftest.py (FakeLLM) + 33 tests
scripts/          run_research.py — thin CLI entrypoint
docs/             phase0-architecture.md
memory/           (empty — reserved for the knowledge graph, Milestone 9)
experiments/      (empty — Milestone 6+)
evaluation/       (empty — Milestone 7+)
security/         (empty — sandbox config, Milestone 6)
models/           (empty — Pydantic schemas, added as needed)
configs/          (empty)
```
