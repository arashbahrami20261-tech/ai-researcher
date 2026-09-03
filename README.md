# AI Researcher

An autonomous AI research agent, built incrementally in milestones.

Current state: **question → arXiv search → summary + hypothesis → critic
review → saved to database → markdown report.** Separately, the coding agent
writes Python and executes it inside a locked-down Docker container. Both
paths verified end-to-end against a local model.

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
pytest              # 68 tests, no network, no API key needed
pytest -m live      # 7 more: real arXiv calls + real Docker containers
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
| 6 | Coding agent + Docker sandbox | done |
| 7 | Experiment tracking + evaluation engine | done |
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
- The research loop and the experiment runner are still separate entry
  points. Nothing yet turns a generated hypothesis into an experiment
  automatically — that is Milestone 8.
- Nothing checks whether a result is *meaningful*. A live run produced a
  timing bug that reported 8463 seconds for a 13-second loop: the code
  ran, the metric parsed, the number was nonsense. Catching that needs a
  critic on results, not just on hypotheses.
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

## Sandbox

Generated code is untrusted, so it runs in a container with: no network,
512m memory (swap pinned to match), 0.5 CPU, a 64-process limit, read-only
root filesystem, non-root user, all Linux capabilities dropped, and
no-new-privileges. The code directory is mounted read-only.

These are not assumed to work. `tests/test_sandbox.py` launches real
containers and asserts that network calls fail, the host filesystem is
invisible, writes outside /tmp are refused, and infinite loops get killed.
A security control nobody has watched fail is not a control.

Requires Docker, runnable without sudo.

## Evaluation

Every result is measured against an explicit baseline, and the engine
refuses to call something an improvement when the evidence is thin. It
returns INCONCLUSIVE for a single run, NO_DIFFERENCE for a change under
1%, and INCONCLUSIVE again when the effect is smaller than the spread
across runs.

That refusal is the point. A live run had binary search beating its
baseline by 94% and the engine still declined to call it an improvement,
because one run is not evidence. Experiments store their seed, the exact
code that ran, stdout, and any error — failed runs included, since a
failure that leaves no trace teaches nothing.

## Project layout

```
agents/           critic.py (Milestone 5), coder.py (Milestone 6)
research/         loop.py — the research loop orchestration
literature/       arxiv_search.py — literature backend
tools/            llm_provider.py — LLM abstraction (Claude + Ollama)
database/         models.py, db.py — SQLAlchemy schema + engine
migrations/       Alembic migration history
tests/            conftest.py (FakeLLM) + 75 tests
scripts/          run_research.py — thin CLI entrypoint
docs/             phase0-architecture.md
memory/           (empty — reserved for the knowledge graph, Milestone 9)
experiments/      runner.py — runs and records one experiment
evaluation/       compare.py — baseline comparison, refuses weak claims
security/         sandbox.py — Docker isolation for generated code
models/           (empty — Pydantic schemas, added as needed)
configs/          (empty)
```
