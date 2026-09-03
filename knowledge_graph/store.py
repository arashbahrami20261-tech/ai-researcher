"""
Research knowledge graph — Milestone 9.

Records how the pieces of a research project relate to each other:
which experiment follows which, which tests which hypothesis, which
contradicts which.

Why this exists now rather than as a nice-to-have: the result critic
reviews each experiment in isolation. On a live 3-cycle run it passed
all three, and never noticed that binary search timings across those
cycles contradict each other (0.00068s at 50k elements, 0.00136s at
200k, 0.00055s at 400k). Binary search does not get faster on a bigger
list. A critic with no access to the other results cannot catch that.

This module is what gives it that access.
"""

from __future__ import annotations

from dataclasses import dataclass

from database.models import Experiment, GraphEdge, Metric

# Node types. Kept as constants rather than free strings so a typo
# produces an ImportError rather than an edge nobody can ever query.
NODE_EXPERIMENT = "experiment"
NODE_HYPOTHESIS = "hypothesis"
NODE_PAPER = "paper"
NODE_PROJECT = "project"

# Relations. Direction matters: A follows B means A came after B.
REL_FOLLOWS = "follows"
REL_TESTS = "tests"
REL_CONTRADICTS = "contradicts"
REL_SUPPORTS = "supports"
REL_REPRODUCES = "reproduces"

KNOWN_RELATIONS = {
    REL_FOLLOWS,
    REL_TESTS,
    REL_CONTRADICTS,
    REL_SUPPORTS,
    REL_REPRODUCES,
}


@dataclass
class PriorResult:
    """One earlier experiment, flattened into what a critic needs to see."""

    experiment_id: int
    methodology: str
    metrics: dict[str, float]

    def summary(self) -> str:
        values = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
        return f"Experiment #{self.experiment_id}: {self.methodology[:150]} -> {values}"


def add_edge(
    session,
    source_type: str,
    source_id: int,
    relation: str,
    target_type: str,
    target_id: int,
    note: str = "",
) -> GraphEdge:
    """
    Record one relationship.

    Rejects unknown relations rather than storing them. The database has
    no foreign keys on these columns by design, so this is the only place
    a typo gets caught — an edge with relation "contradict" would be
    silently invisible to every query that looks for "contradicts".
    """
    if relation not in KNOWN_RELATIONS:
        raise ValueError(
            f"Unknown relation {relation!r}. Known: {sorted(KNOWN_RELATIONS)}"
        )

    edge = GraphEdge(
        source_type=source_type,
        source_id=source_id,
        relation=relation,
        target_type=target_type,
        target_id=target_id,
        note=note,
    )
    session.add(edge)
    session.commit()
    return edge


def link_experiment_chain(
    session, experiment_id: int, previous_experiment_id: int | None, hypothesis_id: int
) -> None:
    """
    Record the two edges every experiment in a cycle has: what it tests,
    and what it came after. Called by the cycle runner so the graph is
    built as research happens rather than reconstructed later.
    """
    add_edge(session, NODE_EXPERIMENT, experiment_id, REL_TESTS, NODE_HYPOTHESIS, hypothesis_id)

    if previous_experiment_id is not None:
        add_edge(
            session,
            NODE_EXPERIMENT,
            experiment_id,
            REL_FOLLOWS,
            NODE_EXPERIMENT,
            previous_experiment_id,
        )


def prior_results_in_chain(session, experiment_id: int, limit: int = 5) -> list[PriorResult]:
    """
    Walk backwards along `follows` edges and return earlier results.

    Returns oldest-first, because a critic reading a sequence needs it in
    the order it happened to spot a trend that reverses.

    `limit` exists because the whole history does not fit in a prompt and
    recent results are the ones a new measurement is most likely to
    contradict.
    """
    results: list[PriorResult] = []
    current = experiment_id

    for _ in range(limit):
        edge = (
            session.query(GraphEdge)
            .filter_by(
                source_type=NODE_EXPERIMENT,
                source_id=current,
                relation=REL_FOLLOWS,
                target_type=NODE_EXPERIMENT,
            )
            .first()
        )
        if edge is None:
            break

        current = edge.target_id
        experiment = session.query(Experiment).filter_by(id=current).first()
        if experiment is None:
            # The edge points at something that no longer exists. Stop
            # rather than pretend the chain continues.
            break

        metrics = {
            m.name: m.value
            for m in session.query(Metric).filter_by(experiment_id=current).all()
        }
        results.append(
            PriorResult(
                experiment_id=current,
                methodology=experiment.methodology,
                metrics=metrics,
            )
        )

    results.reverse()
    return results


def edges_for(session, node_type: str, node_id: int) -> list[GraphEdge]:
    """Every edge touching this node, in either direction."""
    outgoing = (
        session.query(GraphEdge)
        .filter_by(source_type=node_type, source_id=node_id)
        .all()
    )
    incoming = (
        session.query(GraphEdge)
        .filter_by(target_type=node_type, target_id=node_id)
        .all()
    )
    return outgoing + incoming
