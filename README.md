# AI Researcher

An autonomous AI research agent, built incrementally in milestones.

Current state: **question → arXiv search → summary + hypothesis → critic
review → saved to database → markdown report.**

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
python scripts/run_research.py "How do transformers handle long context?"
```

## Tests

```bash
pytest              # 31 tests, no network, no API key needed
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

- **The live path has not been run end-to-end with a real API key.** All
  tests use a fake provider, so they prove the logic is wired correctly,
  not that the Anthropic call itself succeeds.
- The critic reviews hypotheses only. The experiment-stage checklist
  (data leakage, sample size, statistical significance, baseline choice)
  is defined in `agents/critic.py` but deliberately not used yet — there
  are no experiments to check it against until Milestone 7.
- `experiments` and `metrics` tables exist but nothing writes to them yet.
- SQLite, single user, no API layer. Postgres is a one-line change to
  `DATABASE_URL` in `database/db.py` when it's needed.

## Project layout

```
agents/           critic.py — the Critic agent (Milestone 5)
research/         loop.py — the research loop orchestration
literature/       arxiv_search.py — literature backend
tools/            llm_provider.py — LLM abstraction + structured output
database/         models.py, db.py — SQLAlchemy schema + engine
migrations/       Alembic migration history
tests/            conftest.py (FakeLLM) + 31 tests
scripts/          run_research.py — thin CLI entrypoint
docs/             phase0-architecture.md
memory/           (empty — reserved for the knowledge graph, Milestone 9)
experiments/      (empty — Milestone 6+)
evaluation/       (empty — Milestone 7+)
security/         (empty — sandbox config, Milestone 6)
models/           (empty — Pydantic schemas, added as needed)
configs/          (empty)
```
