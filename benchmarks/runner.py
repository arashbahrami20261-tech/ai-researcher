"""
Benchmark runner — Milestone 10.

Scores the system against benchmarks/tasks.py so "it got better" can be
a measurement instead of a claim.

Two things this runner does that a naive one would not.

It runs each task several times. A 7B model asked the same consistency
question twice gave opposite answers, so a single pass measures luck as
much as capability. `repeats` defaults to 3 and the score carries the
spread, not just the mean.

It reports false alarms separately from misses. A critic that flags
every result catches every bug and is useless. Precision and recall are
both reported because either one alone can be gamed by a component that
always says yes or always says no.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from agents.coder import write_and_run
from agents.result_critic import VERDICT_TRUSTWORTHY, review_result
from benchmarks.tasks import BUG_TASKS, CODING_TASKS, TREND_TASKS
from evaluation.trends import check_monotonic
from experiments.runner import parse_metrics
from tools.llm_provider import LLMProvider

DEFAULT_REPEATS = 3


@dataclass
class SuiteScore:
    """One suite's result across all repeats."""

    suite: str
    runs: list[float] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.runs) if self.runs else 0.0

    @property
    def spread(self) -> float:
        """Standard deviation across repeats. High spread means unreliable."""
        return statistics.stdev(self.runs) if len(self.runs) >= 2 else 0.0

    def __str__(self) -> str:
        spread = f" (sd {self.spread:.2f})" if self.spread else ""
        return f"{self.suite}: {self.mean:.1%}{spread} over {len(self.runs)} run(s)"


def score_coding(llm: LLMProvider, repeats: int = DEFAULT_REPEATS) -> SuiteScore:
    """
    How often does the model write a script producing the right number?

    Uses the same output_check the experiment runner uses, so the score
    reflects the pipeline as it actually runs rather than an easier
    version of it.
    """
    score = SuiteScore("coding")
    per_task: dict[str, int] = {task.task_id: 0 for task in CODING_TASKS}

    for _ in range(repeats):
        passed = 0
        for task in CODING_TASKS:
            prompt = (
                f"{task.instruction}\n\n"
                f"Print the answer on its own line as: "
                f"METRIC {task.metric_name}=value\n"
                f"Use only the Python standard library."
            )

            def check(stdout, name=task.metric_name):
                return "" if name in parse_metrics(stdout) else (
                    f"No METRIC line named {name} was printed."
                )

            print(f"  coding: {task.task_id}", flush=True)
            outcome = write_and_run(llm, prompt, timeout_seconds=30, output_check=check)
            if outcome.succeeded and task.passed(parse_metrics(outcome.result.stdout)):
                passed += 1
                per_task[task.task_id] += 1

        score.runs.append(passed / len(CODING_TASKS))

    score.detail = per_task
    return score


def score_bug_detection(llm: LLMProvider, repeats: int = DEFAULT_REPEATS) -> SuiteScore:
    """
    Does the result critic catch known bugs without flagging clean code?

    Reported as accuracy, with caught/false-alarm counts in the detail.
    Accuracy alone hides which way a component fails, and the two failure
    modes need different fixes.
    """
    score = SuiteScore("bug_detection")
    caught = 0
    false_alarms = 0
    total_buggy = sum(1 for t in BUG_TASKS if t.has_bug) * repeats
    total_clean = sum(1 for t in BUG_TASKS if not t.has_bug) * repeats

    for _ in range(repeats):
        correct = 0
        for task in BUG_TASKS:
            print(f"  bug: {task.task_id}", flush=True)
            try:
                critique = review_result(
                    llm, task.methodology, task.code, task.stdout, task.metrics
                )
                flagged = critique.verdict != VERDICT_TRUSTWORTHY
            except ValueError:
                # An unparseable verdict is a failure, not a pass. Scoring
                # it as "did not flag" would reward a broken critic on the
                # clean cases.
                flagged = not task.has_bug

            if flagged == task.has_bug:
                correct += 1
            if flagged and task.has_bug:
                caught += 1
            if flagged and not task.has_bug:
                false_alarms += 1

        score.runs.append(correct / len(BUG_TASKS))

    score.detail = {
        "caught": f"{caught}/{total_buggy}",
        "false_alarms": f"{false_alarms}/{total_clean}",
    }
    return score


def score_trends() -> SuiteScore:
    """
    Does the trend check catch known inconsistencies?

    No model, no repeats: this is arithmetic. It is scored anyway because
    a threshold change can silently break it, and a benchmark that only
    covers the model-driven parts would not notice.
    """
    score = SuiteScore("trends")
    correct = 0
    detail = {}

    for task in TREND_TASKS:
        issue = check_monotonic(
            task.metric_name, task.series, should_increase=task.should_increase
        )
        flagged = issue is not None
        ok = flagged == task.is_inconsistent
        correct += ok
        detail[task.task_id] = "correct" if ok else (
            "false alarm" if flagged else "missed"
        )

    score.runs.append(correct / len(TREND_TASKS))
    score.detail = detail
    return score


def run_all(llm: LLMProvider, repeats: int = DEFAULT_REPEATS) -> dict[str, SuiteScore]:
    """Run every suite. Returns scores keyed by suite name."""
    return {
        "coding": score_coding(llm, repeats),
        "bug_detection": score_bug_detection(llm, repeats),
        "trends": score_trends(),
    }


def format_scores(scores: dict[str, SuiteScore]) -> str:
    lines = ["# Benchmark", ""]
    for score in scores.values():
        lines.append(f"## {score}")
        for key, value in score.detail.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)
