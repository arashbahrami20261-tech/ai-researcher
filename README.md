# AI Researcher

An autonomous AI research agent, built incrementally in milestones.

Current state: the system runs a **closed research loop** — a hypothesis
becomes generated Python, executed in a locked-down Docker container,
measured against a baseline, checked for consistency against earlier
results in the same chain, and turned into the next hypothesis. A
separate literature path handles arXiv search and paper summaries.

Verified end-to-end against a local model: a 3-cycle run scaled from
50,000 to 200,000 elements, each step chosen by the system.

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
pytest              # 127 tests, no network, no API key needed
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
| 8 | Hypothesis generation from results (closed loop) | done |
| 9 | Knowledge graph + trend checks | done |
| 10 | Benchmarking + controlled self-improvement | done |

## Known limitations

- The Anthropic path has still not been run with a real API key. The loop
  has been verified end-to-end using a local Ollama model instead
  (`qwen2.5:7b`), which is what the `LLMProvider` abstraction was built to
  allow. `ClaudeProvider`'s default model string may also be stale.
- A local 7B model is much weaker than a frontier model. It produces usable
  summaries and hypotheses, but expect vaguer reasoning.
- The trend check uses cycle number as a proxy for input size, because
  input size is not recorded per experiment. It can tell that a value
  fell when it should have risen, but not whether it rose by the right
  amount when the input doubled. That is the weaker half of the check.
- The model critic's judgement of consistency is not reliable at 7B.
  Asked twice about the same three numbers it gave opposite answers, so
  the numeric check carries that job and the model critic is kept for
  reading code.
- The literature loop (research/loop.py) and the experiment loop
  (research/cycle.py) are still separate entry points. A hypothesis from
  a paper search does not yet flow into an experiment.
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

## The research cycle

`research/cycle.py` runs a chain where each hypothesis comes from the
previous result rather than from a person. One cycle is: hypothesis ->
generated code -> sandboxed run -> metrics -> baseline comparison ->
result review -> ranked follow-up proposals. The best-ranked proposal
becomes the next cycle.

A live 3-cycle run went from 50,000 to 200,000 to 400,000 elements,
each step chosen by the system.

Two gates can end a chain early: an experiment that measures nothing,
and a result the critic marks `invalid`. A `suspicious` verdict is
recorded as a warning but does not stop the run — an earlier version
treated every doubt as fatal and the loop never reached a second cycle.

`max_cycles` is required. Nothing here decides on its own to keep going.

## Knowledge graph and trend checks

Each cycle records two edges: what it `tests`, and what it `follows`.
Walking the `follows` edges backwards gives the result critic the earlier
results in the same chain — something the isolated review never had.

Alongside it, `evaluation/trends.py` compares the numbers directly. That
part is deliberately not a prompt. Asked twice whether the same three
timings were consistent, a 7B local model answered "not consistent" once
and "generally consistent" the next time. Comparing numbers is
arithmetic, not judgement, and arithmetic should give the same answer
every time.

Two constants here were wrong on the first attempt and are documented
where they live: metric names drifted between cycles until they were
enforced in code, and the tolerance was set to 0.5 without looking at
data, which made it miss a real 46.8% drop by three percentage points.

## Benchmarks and self-improvement

`benchmarks/` scores the system against fixed tasks with known answers,
so "it got better" is a number rather than an opinion. Half the bug and
trend cases are clean on purpose: a critic that flags everything catches
every bug, and looks perfect unless false alarms are counted too.

Current baseline, bug detection: **83.3% over 3 runs, sd 0.0, zero false
alarms.** It catches both magnitude anomalies. It misses the one bug
visible only in the code — accuracy scored against training labels
instead of the test set — because the number itself looks plausible.

`improve.py` proposes a prompt change, re-runs the benchmark, and accepts
only on a gain above 5%. The first real run was **rejected**: the model
proposed adding "ensure the correct metric is being measured", which
restates what the prompt already said. The benchmark went 83.3% to
83.3%. Without the loop that change would have been applied, because it
reads as sensible.

The finding is that some weaknesses are not prompt-shaped. Detecting
data leakage needs conceptual understanding, not a stronger reminder.

Self-improvement can only alter a prompt string. It cannot edit logic,
add dependencies, or touch the sandbox.

## Multi-agent orchestration: deliberately not built

The roadmap listed a Research Director coordinating specialised agents.
`research/cycle.py` already does that coordination, in deterministic
code that can be tested. Adding a model to decide which agent runs next
would make the system less predictable without making it more capable —
and the spec says to add specialised agents only where they improve
reliability.

Recorded as a decision, not as unfinished work.

## Project layout

```
agents/           critic.py, coder.py, result_critic.py, hypothesis.py
research/         loop.py (literature), cycle.py (the closed loop)
literature/       arxiv_search.py — literature backend
tools/            llm_provider.py — LLM abstraction (Claude + Ollama)
database/         models.py, db.py — SQLAlchemy schema + engine
migrations/       Alembic migration history
tests/            conftest.py (FakeLLM) + 134 tests
scripts/          run_research.py — thin CLI entrypoint
docs/             phase0-architecture.md
knowledge_graph/  store.py — edges between experiments and hypotheses
experiments/      runner.py — runs and records one experiment
evaluation/       compare.py (baselines), trends.py (chain consistency)
security/         sandbox.py — Docker isolation for generated code
models/           (empty — Pydantic schemas, added as needed)
benchmarks/       tasks.py, runner.py, improve.py
configs/          (empty)
```
