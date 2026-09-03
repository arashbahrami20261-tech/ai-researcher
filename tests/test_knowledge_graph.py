"""
Tests for knowledge_graph/store.py.

The graph exists to give the result critic something the isolated
review never had: the other results in the same chain. So the test that
matters most is that walking backwards along `follows` edges actually
returns earlier experiments, in the order they happened.
"""

from __future__ import annotations

import pytest

from database.models import Experiment, GraphEdge, Metric
from knowledge_graph.store import (
    NODE_EXPERIMENT,
    NODE_HYPOTHESIS,
    REL_FOLLOWS,
    add_edge,
    edges_for,
    link_experiment_chain,
    prior_results_in_chain,
)


def _experiment(session, methodology, metrics):
    """Create an experiment row with metrics attached."""
    row = Experiment(hypothesis_id=1, question="q", methodology=methodology, status="done")
    session.add(row)
    session.flush()
    for name, value in metrics.items():
        session.add(Metric(experiment_id=row.id, name=name, value=value))
    session.commit()
    return row.id


def test_an_edge_is_stored(session):
    add_edge(session, NODE_EXPERIMENT, 1, REL_FOLLOWS, NODE_EXPERIMENT, 2, note="why")

    edge = session.query(GraphEdge).one()
    assert edge.relation == REL_FOLLOWS
    assert edge.note == "why"


def test_an_unknown_relation_is_rejected(session):
    # These columns have no foreign keys by design, so a typo would
    # otherwise be stored and be invisible to every query that looks for
    # the correct spelling.
    with pytest.raises(ValueError, match="Unknown relation"):
        add_edge(session, NODE_EXPERIMENT, 1, "contradict", NODE_EXPERIMENT, 2)

    assert session.query(GraphEdge).count() == 0


def test_linking_a_cycle_records_what_it_tests_and_what_it_follows(session):
    link_experiment_chain(session, experiment_id=2, previous_experiment_id=1, hypothesis_id=7)

    relations = {e.relation for e in session.query(GraphEdge).all()}
    assert relations == {"tests", "follows"}


def test_the_first_cycle_has_nothing_to_follow(session):
    link_experiment_chain(session, experiment_id=1, previous_experiment_id=None, hypothesis_id=7)

    edges = session.query(GraphEdge).all()
    assert len(edges) == 1
    assert edges[0].relation == "tests"


def test_prior_results_come_back_oldest_first(session):
    # A critic reading a sequence needs it in the order it happened, or a
    # trend that reverses looks like a trend that never existed.
    first = _experiment(session, "50k elements", {"t": 1.0})
    second = _experiment(session, "100k elements", {"t": 2.0})
    third = _experiment(session, "200k elements", {"t": 0.5})

    link_experiment_chain(session, second, first, 1)
    link_experiment_chain(session, third, second, 1)

    prior = prior_results_in_chain(session, third)

    assert [p.experiment_id for p in prior] == [first, second]
    assert prior[0].metrics == {"t": 1.0}


def test_the_first_experiment_has_no_prior_results(session):
    only = _experiment(session, "50k elements", {"t": 1.0})
    link_experiment_chain(session, only, None, 1)

    assert prior_results_in_chain(session, only) == []


def test_the_walk_stops_at_the_limit(session):
    ids = [_experiment(session, f"run {i}", {"t": float(i)}) for i in range(6)]
    for previous, current in zip(ids, ids[1:]):
        link_experiment_chain(session, current, previous, 1)

    # The whole history does not fit in a prompt, and recent results are
    # the ones a new measurement is most likely to contradict.
    assert len(prior_results_in_chain(session, ids[-1], limit=2)) == 2


def test_a_dangling_edge_ends_the_walk_instead_of_crashing(session):
    real = _experiment(session, "50k elements", {"t": 1.0})
    # Points at an experiment that does not exist.
    add_edge(session, NODE_EXPERIMENT, real, REL_FOLLOWS, NODE_EXPERIMENT, 9999)

    assert prior_results_in_chain(session, real) == []


def test_the_summary_names_the_experiment_and_its_numbers(session):
    first = _experiment(session, "a sorted list of 50000 integers", {"t": 1.5})
    second = _experiment(session, "100k", {"t": 2.0})
    link_experiment_chain(session, second, first, 1)

    text = prior_results_in_chain(session, second)[0].summary()

    assert "#" in text
    assert "t=1.5" in text


def test_edges_for_finds_both_directions(session):
    add_edge(session, NODE_EXPERIMENT, 5, REL_FOLLOWS, NODE_EXPERIMENT, 4)
    add_edge(session, NODE_EXPERIMENT, 6, REL_FOLLOWS, NODE_EXPERIMENT, 5)

    assert len(edges_for(session, NODE_EXPERIMENT, 5)) == 2
