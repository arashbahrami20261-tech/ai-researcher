"""
Tests for experiments/runner.py.

The model and the sandbox are both faked here; what is under test is the
bookkeeping. The spec\'s rule for this module is that no result is
accepted without a record of how it was produced, so these check that the
record exists even when the run fails.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.coder import CodingOutcome
from database.models import Experiment, Metric
from experiments.runner import parse_metrics, run_experiment
from security.sandbox import ExecutionResult
from tests.conftest import FakeLLM


def _outcome(stdout, ok=True):
    result = ExecutionResult(
        stdout=stdout, stderr="" if ok else "boom", exit_code=0 if ok else 1, timed_out=False
    )
    return CodingOutcome(code="print('x')", result=result, attempts=[("print('x')", result)])


def test_parse_metrics_finds_prefixed_lines_only():
    stdout = "starting\nMETRIC accuracy=0.87\nloss=0.31\ndone"
    # The unprefixed line is ignored on purpose: without the prefix there
    # is no way to tell a metric from ordinary output.
    assert parse_metrics(stdout) == {"accuracy": 0.87}


def test_parse_metrics_handles_negative_and_scientific_notation():
    assert parse_metrics("METRIC delta=-1.5e-3") == {"delta": -0.0015}


def test_parse_metrics_skips_malformed_values_without_losing_good_ones():
    stdout = "METRIC good=1.0\nMETRIC bad=abc\nMETRIC also_good=2.0"
    assert parse_metrics(stdout) == {"good": 1.0, "also_good": 2.0}


def test_a_successful_run_is_recorded_with_its_metrics(session):
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC acc=0.9")):
        outcome = run_experiment(llm, session, 1, "q", "do a thing")

    assert outcome.succeeded
    assert outcome.metrics == {"acc": 0.9}
    assert session.query(Metric).count() == 1
    assert session.query(Experiment).one().status == "done"


def test_a_seed_is_always_generated_and_stored(session):
    # An experiment with no recorded seed cannot be reproduced.
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC acc=0.9")):
        run_experiment(llm, session, 1, "q", "m")

    assert session.query(Experiment).one().random_seed > 0


def test_an_explicit_seed_is_kept(session):
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC acc=0.9")):
        run_experiment(llm, session, 1, "q", "m", seed=12345)

    assert session.query(Experiment).one().random_seed == 12345


def test_a_failed_run_still_leaves_a_record(session):
    # A failure that vanishes is worse than one that is logged.
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("", ok=False)):
        outcome = run_experiment(llm, session, 1, "q", "m")

    assert not outcome.succeeded
    experiment = session.query(Experiment).one()
    assert experiment.status == "failed"
    assert experiment.code


def test_a_run_that_prints_no_metrics_counts_as_failed(session):
    # Exit code 0 is not success if nothing was measured.
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("all done!")):
        outcome = run_experiment(llm, session, 1, "q", "m")

    assert not outcome.succeeded
    assert session.query(Experiment).one().status == "failed"


def test_no_baseline_means_no_comparison(session):
    # A bare number must never be reported as an improvement.
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC acc=0.99")):
        outcome = run_experiment(llm, session, 1, "q", "m")

    assert outcome.comparison is None


def test_a_baseline_produces_a_comparison(session):
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC acc=0.99")):
        outcome = run_experiment(
            llm, session, 1, "q", "m", baseline_metric="acc", baseline_value=0.5
        )

    assert outcome.comparison is not None
    assert outcome.comparison.baseline == 0.5


def test_the_prompt_shows_a_worked_metric_example(session):
    # A 7B model ignored the format rule until it was given an example.
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC a=1")) as war:
        run_experiment(llm, session, 1, "q", "m")

    task = war.call_args[0][1]
    assert "METRIC accuracy=0.87" in task


def test_the_runner_asks_the_coder_to_retry_when_no_metrics_appear(session):
    """
    The runner supplies an output check rather than inspecting stdout
    afterwards: by the time the runner sees the output, the attempt is
    already spent and the model has no chance to fix it.
    """
    llm = FakeLLM(["irrelevant"])

    with patch("experiments.runner.write_and_run", return_value=_outcome("METRIC a=1")) as war:
        run_experiment(llm, session, 1, "q", "m")

    check = war.call_args.kwargs["output_check"]
    assert check("METRIC a=1") == ""
    assert "METRIC" in check("nothing measured here")
