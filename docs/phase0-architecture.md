# AI Researcher — Phase 0: Architecture Proposal

## 1. Repository Inspection

I checked the current working environment: there is **no existing repository or code** for this project yet. This is a greenfield build — everything below assumes we start from an empty directory.

---

## 2. Proposed Architecture (high level)

Build this as a **modular monolith**, not microservices. One Python codebase with clean internal boundaries (interfaces/abstract base classes) so components can be swapped or split out later. Microservices, message queues, and distributed orchestration would add operational complexity with no payoff at solo-developer scale.

Core architectural layers:

```
┌─────────────────────────────────────────┐
│  CLI / API (FastAPI, optional in MVP)    │
├─────────────────────────────────────────┤
│  Research Director (orchestration)       │
├──────────────┬──────────────┬───────────┤
│ Literature    │ Hypothesis   │ Critic    │
│ Engine        │ Generator    │ Agent     │
├──────────────┴──────────────┴───────────┤
│  Coding Agent + Sandboxed Runner (Docker)│
├───────────────────────────────────────────┤
│  Experiment Engine + Evaluation Engine     │
├───────────────────────────────────────────┤
│  Long-Term Memory (Postgres + pgvector)    │
├───────────────────────────────────────────┤
│  LLM Abstraction Layer (Claude first)      │
└───────────────────────────────────────────┘
```

**LLM abstraction layer**: a single `LLMProvider` interface (`generate()`, `generate_structured()`) with a Claude implementation first. API keys read only from environment variables, never hard-coded.

**Data store for MVP**: SQLite (zero setup, file-based). Migrate to Postgres + pgvector only once semantic search over papers is actually needed (Milestone 5+).

---

## 3. Proposed Directory Structure

```
ai_researcher/
  agents/              # Research Director, Critic, Hypothesis Generator, etc.
  research/            # core research-loop orchestration logic
  literature/          # paper search, retrieval, provenance tracking
  memory/              # persistent research memory (DB access layer)
  experiments/         # experiment definition, execution, tracking
  evaluation/          # metrics, statistical comparison against baselines
  models/              # Pydantic schemas (Experiment, Hypothesis, Paper, ...)
  tools/               # LLM abstraction, sandbox runner, utilities
  database/            # SQLAlchemy models + Alembic migrations
  knowledge_graph/      # (post-MVP) relationships between papers/methods/experiments
  security/            # sandbox config, resource limits, approval gates
  benchmarks/          # internal self-evaluation suite (post-MVP)
  reports/             # generated research report templates/output
  tests/
  configs/
  scripts/
  docs/
```

---

## 4. Database Schema (core tables)

| Table | Key columns |
|---|---|
| `projects` | id, name, description, created_at |
| `papers` | id, title, authors, abstract, url, published_at, source, embedding |
| `authors` | id, name |
| `concepts` | id, name, description |
| `hypotheses` | id, project_id, text, motivation, status (proposed/tested/rejected), related_paper_ids |
| `experiments` | id, hypothesis_id, question, methodology, code_version, dataset_ref, config_json, random_seed, baseline_ref, status |
| `experiment_runs` | id, experiment_id, started_at, finished_at, env_info_json, logs_ref, artifacts_ref |
| `metrics` | id, run_id, name, value, confidence_interval |
| `datasets` | id, name, version, source, checksum |
| `models` | id, name, version, provider |
| `results` | id, run_id, summary, interpretation, limitations |
| `errors` | id, run_id, message, traceback |
| `research_notes` | id, project_id, text, created_at |
| `citations` | id, paper_id, cited_paper_id |
| `research_tasks` | id, project_id, description, status, priority |

Use **Alembic** for migrations from day one, even on SQLite, so the move to Postgres later is a config change, not a rewrite.

---

## 5. Agent Architecture

MVP ships with **three** agents, not the full ten:

1. **Research Director** — takes a research question, calls Literature Engine, calls Coder, writes results to memory. This *is* the orchestration loop for MVP.
2. **Literature Researcher** — searches arXiv (free, no API key needed) and/or Semantic Scholar, returns papers with provenance.
3. **Critic** — a single LLM call that checks a result against a fixed checklist (baseline appropriate? sample size? statistically meaningful?) before it's marked "accepted."

Coder/Experiment Runner, Hypothesis Generator, Data Analyst, and the rest get added in later milestones, only once the basic loop is proven to work end-to-end. Adding all ten roles up front, per the spec's own instruction, would be unnecessary complexity before reliability is established.

---

## 6. Security Model

- All generated code execution happens inside a **Docker container** with: CPU limit, memory limit, execution timeout, no network access by default, read-only filesystem except a scratch directory.
- API keys: environment variables only (`.env`, gitignored), never committed, never logged.
- **Human control modes**, default = **Mode 1 (approve everything)**. The system never silently escalates to a more autonomous mode — mode changes are an explicit user action.
- Every experiment run is logged with what code ran, what data it touched, and what it produced — no silent state changes.

---

## 7. Milestone Roadmap (realistic sequencing)

| # | Milestone | What "done" looks like |
|---|---|---|
| 1 | Repo scaffold + LLM abstraction | `research_query("...")` returns a Claude response via the abstraction layer |
| 2 | Literature retrieval (arXiv only) | Given a question, returns 5–10 relevant papers with metadata |
| 3 | Persistent memory (SQLite + schema) | A research question + its papers + a summary are saved and retrievable |
| 4 | Basic research loop (no code execution yet) | question → search → LLM summary/hypothesis → saved report, run end-to-end via CLI |
| 5 | Critic agent | Loop output is checked against the review checklist before being marked accepted |
| 6 | Coding agent + Docker sandbox | Agent can write and run a small experiment script safely |
| 7 | Experiment tracking + evaluation engine | Metrics, baseline comparison, reproducibility metadata stored per run |
| 8 | Hypothesis generation from results | After a run, system proposes 2–3 follow-up experiments, ranked |
| 9 | Knowledge graph | Relationships between papers/methods/experiments become queryable |
| 10 | Multi-agent orchestration, benchmarking, controlled self-improvement | Full system as specified |

Milestones 1–5 are the real MVP. 6 onward is where this becomes a multi-month project even working with AI assistance at every step — worth knowing going in, especially with your university start date approaching.

---

## 8. MVP Definition

**A CLI tool that**: takes a research question → searches arXiv → asks Claude to summarize findings and propose a hypothesis → saves everything (question, papers, summary, hypothesis) to a local SQLite database → prints a short markdown report.

No FastAPI, no Docker, no knowledge graph, no multi-agent orchestration, no self-improvement loop. Everything else in the spec builds on top of this once it works.

---

## 9. Complexity Estimate

- **MVP (Milestones 1–4)**: a few focused sessions of build time — realistic to have working within days, learning Python fundamentals along the way.
- **Full system as specified (all 16 milestones)**: this is the scope of a multi-person engineering effort over months, even with heavy AI assistance — Docker sandboxing, a knowledge graph, statistically rigorous evaluation, and a *controlled* self-improvement loop are each substantial subsystems on their own.

---

## 10. First 5 Concrete Implementation Tasks

1. Initialize the repo: folder structure above, git, `.env.example`, `.gitignore`, `requirements.txt` (start minimal: `anthropic`, `requests`, `pydantic`, `sqlalchemy`).
2. Build the LLM abstraction layer: one `LLMProvider` base class, one `ClaudeProvider` implementation, reading the API key from an environment variable.
3. Build the arXiv literature search tool: a function that takes a query string and returns a list of papers (title, authors, abstract, URL) — arXiv's API is free and needs no key.
4. Define the SQLite schema (`projects`, `research_notes`, `papers`, `hypotheses` tables only, for now) with SQLAlchemy models + one Alembic migration.
5. Wire it together into one CLI script: question in → arXiv search → Claude summary/hypothesis → save to DB → print markdown report.

---

Per the project rules: stopping here for approval before any of Milestones 1–5 get implemented.
