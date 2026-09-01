"""
Tests for database/models.py.

These use an in-memory SQLite database (`sqlite:///:memory:`) created
fresh for each test, instead of the real `ai_researcher.db` file. That
way running the test suite never touches — or accidentally wipes — your
actual research data.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Project, ResearchNote, Hypothesis, Experiment, Metric


@pytest.fixture
def session():
    """A fresh in-memory database, torn down automatically after each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    yield s
    s.close()


def test_create_project(session):
    project = Project(name="test project", description="a test")
    session.add(project)
    session.commit()

    assert project.id is not None
    fetched = session.query(Project).filter_by(name="test project").first()
    assert fetched.description == "a test"


def test_note_belongs_to_project(session):
    project = Project(name="p1")
    session.add(project)
    session.flush()

    note = ResearchNote(project_id=project.id, text="a note")
    session.add(note)
    session.commit()

    assert project.notes[0].text == "a note"


def test_hypothesis_default_status_is_proposed(session):
    project = Project(name="p1")
    session.add(project)
    session.flush()

    h = Hypothesis(project_id=project.id, text="a hypothesis")
    session.add(h)
    session.commit()

    assert h.status == "proposed"


def test_experiment_links_to_hypothesis_and_metrics(session):
    project = Project(name="p1")
    session.add(project)
    session.flush()

    hypothesis = Hypothesis(project_id=project.id, text="a hypothesis")
    session.add(hypothesis)
    session.flush()

    experiment = Experiment(
        hypothesis_id=hypothesis.id,
        question="does X improve Y?",
        random_seed=42,
    )
    session.add(experiment)
    session.flush()

    metric = Metric(experiment_id=experiment.id, name="accuracy", value=0.87)
    session.add(metric)
    session.commit()

    assert hypothesis.experiments[0].question == "does X improve Y?"
    assert experiment.metrics[0].value == pytest.approx(0.87)
    assert experiment.status == "planned"  # default value
