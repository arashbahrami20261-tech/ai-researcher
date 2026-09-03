"""
Database schema — MVP subset.

Only 4 tables for now (projects, research_notes, papers, hypotheses), per
the Phase 0 architecture: the full schema (experiments, experiment_runs,
metrics, datasets, models, results, errors, citations, research_tasks)
gets added in later milestones once there's an experiment loop that
actually needs them. Adding empty tables now would just be unused
complexity.
"""

from __future__ import annotations

import datetime

from sqlalchemy import ForeignKey, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime.datetime:
    # Small wrapper so every timestamp column below uses timezone-aware
    # UTC time via one place, instead of the now-deprecated
    # datetime.datetime.utcnow() spread across the file.
    return datetime.datetime.now(datetime.UTC)


class Project(Base):
    """A research project groups notes, papers, and hypotheses together."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    notes: Mapped[list["ResearchNote"]] = relationship(back_populates="project")
    hypotheses: Mapped[list["Hypothesis"]] = relationship(back_populates="project")


class Paper(Base):
    """
    A single paper found during literature search, with provenance
    (source + url) kept explicit — this is what lets the system later
    distinguish "verified in the literature" from "the model guessed this".
    """

    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    authors: Mapped[str] = mapped_column(Text)  # stored as a comma-joined string for the MVP
    abstract: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(500))
    published: Mapped[str] = mapped_column(String(50), default="")
    source: Mapped[str] = mapped_column(String(50), default="arxiv")


class ResearchNote(Base):
    """A free-text note tied to a project — e.g. a literature summary."""

    __tablename__ = "research_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="notes")


class Hypothesis(Base):
    """A candidate explanation or research direction generated from a note."""

    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="proposed")  # proposed/tested/rejected
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="hypotheses")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="hypothesis")


class Experiment(Base):
    """
    A single experiment run to test a hypothesis. The schema itself (not
    just its data) is deliberately laid out per the Phase 0 spec: every
    field needed to reproduce the run later, Execution happens in the Docker
    sandbox (security/sandbox.py); this table is the record of it.
    """

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(ForeignKey("hypotheses.id"))
    question: Mapped[str] = mapped_column(Text)
    methodology: Mapped[str] = mapped_column(Text, default="")
    code_version: Mapped[str] = mapped_column(String(100), default="")  # e.g. a git commit hash
    dataset_ref: Mapped[str] = mapped_column(String(255), default="")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    random_seed: Mapped[int] = mapped_column(default=42)
    baseline_ref: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(50), default="planned")  # planned/running/done/failed

    # What actually ran, and what came back. The fields above describe what
    # was *intended*; these three record what *happened*. Without them,
    # `code_version` is just a hash and a result six months from now cannot
    # be traced back to the script that produced it.
    code: Mapped[str] = mapped_column(Text, default="")
    stdout: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    hypothesis: Mapped["Hypothesis"] = relationship(back_populates="experiments")
    metrics: Mapped[list["Metric"]] = relationship(back_populates="experiment")


class Critique(Base):
    """
    The Critic agent's review of a research output (Milestone 5).

    Stored rather than printed-and-forgotten for two reasons the spec calls
    for: the critic must be *able to reject* a result, so the verdict has to
    outlive the run that produced it; and rejected results are themselves
    valuable memory ("we already tried this and it didn't hold up").

    `target_type` / `target_id` form a loose polymorphic link so one critic
    can review a note, a hypothesis, or later an experiment, without needing
    a separate table per reviewable thing.
    """

    __tablename__ = "critiques"

    id: Mapped[int] = mapped_column(primary_key=True)
    target_type: Mapped[str] = mapped_column(String(50))  # note/hypothesis/experiment
    target_id: Mapped[int] = mapped_column()
    verdict: Mapped[str] = mapped_column(String(50))  # accepted/revise/rejected
    # Free-text reasoning plus the structured checklist answers, kept as JSON
    # so new checklist questions can be added without a schema migration.
    checklist_json: Mapped[str] = mapped_column(Text, default="{}")
    reasoning: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=_utcnow
    )


class Metric(Base):
    """One measured value from an experiment (e.g. accuracy, loss, F1)."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    name: Mapped[str] = mapped_column(String(100))
    value: Mapped[float] = mapped_column()
    # Nullable: most metrics won't have a confidence interval until the
    # evaluation engine (Milestone 7) supports multi-run statistics.
    confidence_interval: Mapped[str] = mapped_column(String(100), default="")

    experiment: Mapped["Experiment"] = relationship(back_populates="metrics")
