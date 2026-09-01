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


class Project(Base):
    """A research project groups notes, papers, and hypotheses together."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
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
        DateTime, default=datetime.datetime.utcnow
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
        DateTime, default=datetime.datetime.utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="hypotheses")
