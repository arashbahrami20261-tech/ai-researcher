"""
Experiment runner — Milestone 7.

This is where the pieces from earlier milestones become an experiment
rather than a demo. The coding agent writes the script, the sandbox runs
it, the evaluation engine judges the numbers, and every part of that is
written to the database before anything is reported back.

The spec\'s rule for this module: "Never accept a result without recording
how it was produced." So the experiment row is created *before* execution
starts, not after. If the process dies mid-run, the record still exists
with status "running" — which is a useful thing to find later. An
experiment that leaves no trace when it crashes is worse than one that
fails loudly.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from agents.coder import write_and_run
from database.models import Experiment, Metric
from evaluation.compare import Comparison, compare_to_baseline
from tools.llm_provider import LLMProvider

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class ExperimentOutcome:
    """What one experiment produced, plus the row id it was recorded under."""

    experiment_id: int
    succeeded: bool
    metrics: dict[str, float]
    comparison: Comparison | None
    stdout: str
    error: str


def parse_metrics(stdout: str) -> dict[str, float]:
    """
    Pull `METRIC name=value` lines out of a script\'s output.

    A fixed output contract, rather than asking the model to return JSON
    alongside its printed results. Scripts print all sorts of things while
    they run; a distinctive prefix means the metrics can be found without
    the model having to keep its entire stdout parseable.

    Unparseable values are skipped rather than raising: a malformed metric
    line should not throw away a run that otherwise worked.
    """
    metrics: dict[str, float] = {}
    for match in re.finditer(r"^METRIC\s+(\w+)\s*=\s*([-+0-9.eE]+)\s*$", stdout, re.MULTILINE):
        name, raw = match.group(1), match.group(2)
        try:
            metrics[name] = float(raw)
        except ValueError:
            continue
    return metrics


def run_experiment(
    llm: LLMProvider,
    session,
    hypothesis_id: int,
    question: str,
    methodology: str,
    baseline_metric: str | None = None,
    baseline_value: float | None = None,
    higher_is_better: bool = True,
    seed: int | None = None,
    timeout_seconds: int = 60,
    required_metrics: list[str] | None = None,
) -> ExperimentOutcome:
    """
    Design, run, record, and evaluate one experiment.

    `seed` is generated if not supplied, and always stored. An experiment
    without a recorded seed is not reproducible, and the spec treats
    reproducibility as non-negotiable.

    `baseline_value` is optional but the comparison only happens when it is
    given. That is deliberate: a result with no baseline gets reported as a
    bare number, never as an improvement.
    """
    if seed is None:
        seed = random.randint(1, 2**31 - 1)

    # Recorded before the run, so a crash still leaves evidence.
    experiment = Experiment(
        hypothesis_id=hypothesis_id,
        question=question,
        methodology=methodology,
        random_seed=seed,
        baseline_ref=f"{baseline_metric}={baseline_value}" if baseline_metric else "",
        status=STATUS_RUNNING,
        config_json=json.dumps(
            {"timeout_seconds": timeout_seconds, "higher_is_better": higher_is_better}
        ),
    )
    session.add(experiment)
    session.commit()

    # The METRIC prefix is stated twice: as a rule, and as a worked
    # example. A 7B model reliably ignored the rule alone and printed
    # bare "name=value" lines, which parse as nothing. Small models
    # follow examples far better than they follow instructions.
    task = (
        f"{methodology}\n\n"
        f"Use random seed {seed} wherever randomness is involved, so the "
        f"run is reproducible.\n\n"
        "CRITICAL OUTPUT REQUIREMENT: every measured value must be printed "
        "on its own line, starting with the literal word METRIC.\n\n"
        "Correct   -> METRIC accuracy=0.87\n"
        "Incorrect -> accuracy=0.87        (this will not be counted)\n\n"
        + (
            f"Use exactly these metric names: {', '.join(required_metrics)}.\n"
            if required_metrics
            else ""
        )
        + "Use only the Python standard library."
    )

    def _needs_metrics(stdout: str) -> str:
        """
        Reject a run that measured nothing, so the coding agent retries.

        Passed to write_and_run rather than checked afterwards: by the
        time the runner sees the output, the attempt is spent. The model
        needs to be told while it still has retries left.
        """
        found = parse_metrics(stdout)
        if not found:
            return (
                "The script ran without error but printed no METRIC lines, "
                "so nothing was measured. Every value must be printed on "
                "its own line as: METRIC name=value"
            )

        # Names must match across a chain or each series holds one point
        # and the trend check has nothing to compare. Telling the model in
        # the prompt was not enough: it kept the names in one cycle and
        # renamed them in the next.
        if required_metrics:
            missing = [name for name in required_metrics if name not in found]
            if missing:
                return (
                    f"The script printed {sorted(found)} but these exact "
                    f"metric names are required: {missing}. Use these names "
                    f"exactly; results cannot be compared across experiments "
                    f"if the names change."
                )
        return ""

    outcome = write_and_run(
        llm, task, timeout_seconds=timeout_seconds, output_check=_needs_metrics
    )

    experiment.code = outcome.code
    experiment.stdout = outcome.result.stdout if outcome.result else ""
    experiment.error = outcome.result.stderr if outcome.result else "no result"

    if not outcome.succeeded:
        experiment.status = STATUS_FAILED
        session.commit()
        return ExperimentOutcome(
            experiment_id=experiment.id,
            succeeded=False,
            metrics={},
            comparison=None,
            stdout=experiment.stdout,
            error=experiment.error,
        )

    metrics = parse_metrics(experiment.stdout)
    for name, value in metrics.items():
        session.add(Metric(experiment_id=experiment.id, name=name, value=value))

    # A run that produced no parseable metric did not measure anything, so
    # it is not a successful experiment even though the script exited 0.
    experiment.status = STATUS_DONE if metrics else STATUS_FAILED
    if not metrics:
        experiment.error = "Script ran but printed no METRIC lines."

    comparison = None
    if baseline_metric and baseline_value is not None and baseline_metric in metrics:
        comparison = compare_to_baseline(
            metric_name=baseline_metric,
            baseline=baseline_value,
            observed_runs=[metrics[baseline_metric]],
            higher_is_better=higher_is_better,
        )

    session.commit()

    return ExperimentOutcome(
        experiment_id=experiment.id,
        succeeded=bool(metrics),
        metrics=metrics,
        comparison=comparison,
        stdout=experiment.stdout,
        error=experiment.error,
    )
