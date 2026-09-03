"""
The closed research cycle — Milestone 8.

Runs the loop the spec describes:

    hypothesis -> experiment -> result critic -> follow-up hypotheses
    -> next experiment -> ...

Two gates decide whether a cycle continues, and both can stop it:

  1. The experiment must actually produce metrics. A script that exits 0
     without measuring anything is not a result.
  2. The result critic must not reject the measurement. A hypothesis
     built on a misplaced timer propagates that error into every
     experiment after it, so a rejected result ends the chain rather than
     seeding the next one.

`max_cycles` is not optional. An agent that keeps proposing follow-ups
runs until the budget is gone, and the spec asks for cost control and for
autonomy that stays bounded. Default mode is still human-in-the-loop:
this function runs a fixed number of cycles and returns; nothing here
decides on its own to keep going.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.hypothesis import Proposal, generate_followups
from agents.result_critic import VERDICT_INVALID, ResultCritique, review_result
from database.models import Experiment, Hypothesis
from evaluation.compare import format_comparison
from evaluation.trends import TrendIssue, check_monotonic
from knowledge_graph.store import link_experiment_chain, prior_results_in_chain
from experiments.runner import ExperimentOutcome, run_experiment
from tools.llm_provider import LLMProvider


@dataclass
class Cycle:
    """One pass: an experiment, its review, and what it suggests next."""

    cycle_number: int
    hypothesis: str
    methodology: str
    outcome: ExperimentOutcome
    critique: ResultCritique | None = None
    proposals: list[Proposal] = field(default_factory=list)
    trend_issues: list[TrendIssue] = field(default_factory=list)
    stopped_because: str = ""


def _previous_failures(session, project_hypothesis_ids: list[int]) -> list[str]:
    """
    Collect what has already gone wrong, to pass into the next proposal.

    Reads from the database rather than from memory in this process, so a
    cycle resumed in a later session still knows what failed before.
    """
    if not project_hypothesis_ids:
        return []

    rows = (
        session.query(Experiment)
        .filter(Experiment.hypothesis_id.in_(project_hypothesis_ids))
        .filter(Experiment.status == "failed")
        .all()
    )
    return [f"{row.methodology[:200]} -> {row.error[:200]}" for row in rows if row.error]


def run_cycles(
    llm: LLMProvider,
    session,
    project_id: int,
    question: str,
    hypothesis: str,
    methodology: str,
    max_cycles: int = 3,
    baseline_metric: str | None = None,
    baseline_value: float | None = None,
    higher_is_better: bool = True,
    timeout_seconds: int = 60,
) -> list[Cycle]:
    """
    Run up to `max_cycles` experiments, each one following from the last.

    Returns every cycle attempted, including the one that stopped the
    chain. A caller that only sees successful cycles cannot tell a run
    that finished from a run that was cut short.
    """
    cycles: list[Cycle] = []
    hypothesis_ids: list[int] = []
    previous_experiment_id: int | None = None
    # (cycle number, metrics) per successful cycle, used for the numeric
    # trend check. Cycle number stands in for input size: the loop scales
    # up as it goes, so position in the chain is a usable proxy.
    metric_history: list[tuple[int, dict[str, float]]] = []
    # Names are fixed by the first cycle that measures anything, and every
    # cycle after it must use the same ones. Asking the model nicely was
    # not enough: it kept the names in cycle 2 and renamed them in cycle 3,
    # which left each series holding a single point and silently disabled
    # the trend check.
    required_metrics: list[str] | None = None
    current_hypothesis = hypothesis
    current_methodology = methodology

    for n in range(1, max_cycles + 1):
        row = Hypothesis(project_id=project_id, text=current_hypothesis)
        session.add(row)
        session.commit()
        hypothesis_ids.append(row.id)

        outcome = run_experiment(
            llm=llm,
            session=session,
            hypothesis_id=row.id,
            question=question,
            methodology=current_methodology,
            baseline_metric=baseline_metric,
            baseline_value=baseline_value,
            higher_is_better=higher_is_better,
            timeout_seconds=timeout_seconds,
            required_metrics=required_metrics,
        )

        cycle = Cycle(n, current_hypothesis, current_methodology, outcome)

        # Record the edges as research happens rather than reconstructing
        # the chain later from timestamps.
        link_experiment_chain(
            session, outcome.experiment_id, previous_experiment_id, row.id
        )
        previous_experiment_id = outcome.experiment_id

        if not outcome.succeeded:
            cycle.stopped_because = f"Experiment produced no metrics: {outcome.error[:200]}"
            cycles.append(cycle)
            break

        if required_metrics is None:
            required_metrics = list(outcome.metrics)

        metric_history.append((n, outcome.metrics))

        # Deterministic check first. Comparing numbers across the chain is
        # arithmetic, and the model gave contradictory answers to the same
        # comparison on consecutive runs.
        for name in outcome.metrics:
            series = [
                (float(pos), values[name])
                for pos, values in metric_history
                if name in values
            ]
            issue = check_monotonic(name, series, should_increase=True)
            if issue is not None:
                cycle.trend_issues.append(issue)

        prior = [r.summary() for r in prior_results_in_chain(session, outcome.experiment_id)]

        try:
            cycle.critique = review_result(
                llm=llm,
                methodology=current_methodology,
                code=session.query(Experiment).get(outcome.experiment_id).code,
                stdout=outcome.stdout,
                metrics=outcome.metrics,
                prior_results=prior,
            )
        except ValueError as exc:
            # An unreviewable result is not a reviewed one. Stop rather
            # than build the next hypothesis on an unchecked measurement.
            cycle.stopped_because = f"Result critic failed: {exc}"
            cycles.append(cycle)
            break

        # Only a concrete measurement bug ends the chain. A "suspicious"
        # verdict is a flag, not a stop: on a live run the critic returned
        # suspicious for two plausible timings on nothing more than "the
        # targets might be out of range", and treating every doubt as
        # fatal meant the loop never reached a second cycle. Three verdict
        # levels should produce three behaviours, not two.
        if cycle.critique.verdict == VERDICT_INVALID:
            cycle.stopped_because = (
                f"Result critic rejected the measurement: "
                f"{cycle.critique.suspected_bug or cycle.critique.reasoning}"
            )
            cycles.append(cycle)
            break

        if n == max_cycles:
            cycle.stopped_because = f"Reached the {max_cycles}-cycle limit."
            cycles.append(cycle)
            break

        try:
            cycle.proposals = generate_followups(
                llm=llm,
                question=question,
                hypothesis=current_hypothesis,
                methodology=current_methodology,
                metrics=outcome.metrics,
                comparison_summary=(
                    format_comparison(outcome.comparison) if outcome.comparison else ""
                ),
                previous_failures=_previous_failures(session, hypothesis_ids),
            )
        except ValueError as exc:
            cycle.stopped_because = f"No usable follow-up proposals: {exc}"
            cycles.append(cycle)
            break

        cycles.append(cycle)

        # Highest-ranked proposal becomes the next cycle.
        best = cycle.proposals[0]
        current_hypothesis = best.hypothesis
        current_methodology = best.methodology

    return cycles


def format_cycles(cycles: list[Cycle]) -> str:
    """Render a run as a readable report."""
    lines = ["# Research Cycles", ""]

    for c in cycles:
        lines += [f"## Cycle {c.cycle_number}", "", f"**Hypothesis:** {c.hypothesis}", ""]
        lines.append(f"**Method:** {c.methodology}")
        lines.append("")

        if c.outcome.metrics:
            lines.append("**Metrics:**")
            lines += [f"- {k} = {v}" for k, v in c.outcome.metrics.items()]
            lines.append("")

        if c.outcome.comparison:
            lines += [f"**Evaluation:** {format_comparison(c.outcome.comparison)}", ""]

        if c.trend_issues:
            lines.append("**Trend check:** INCONSISTENT")
            lines += [f"- {issue}" for issue in c.trend_issues]
            lines.append("")

        if c.critique:
            lines.append(f"**Result review:** {c.critique.verdict.upper()}")
            if c.critique.suspected_bug:
                lines.append(f"- Suspected bug: {c.critique.suspected_bug}")
            if c.critique.verdict == "suspicious":
                lines.append(
                    "- Flagged but not fatal; the chain continued. Treat the "
                    "numbers above as unconfirmed."
                )
            lines.append("")

        if c.proposals:
            lines.append("**Proposed next:**")
            lines += [
                f"- ({p.total_score:g}) {p.hypothesis}" for p in c.proposals
            ]
            lines.append("")

        if c.stopped_because:
            lines += [f"**Chain ended:** {c.stopped_because}", ""]

    return "\n".join(lines)
