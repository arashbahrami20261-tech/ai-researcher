"""
Numeric consistency checks across a chain of experiments — Milestone 9.

Written in Python rather than delegated to the model, on purpose.

The result critic was given the history of a chain and asked whether a
new measurement fit the trend. Asked the same question twice with the
same numbers, a 7B local model answered "not consistent" once and
"generally consistent" the next time. It does not know the answer; it
guesses, and sometimes guesses right.

Comparing three numbers is not a judgement call. It is arithmetic, and
arithmetic belongs in code: deterministic, testable, and free. The model
critic still runs for the things that genuinely need reading code —
misplaced timers, measuring the wrong thing — but the trend check does
not need a language model and should not depend on one.
"""

from __future__ import annotations

from dataclasses import dataclass

# How far a value may sit from the trend before it is flagged.
#
# Was 0.5, chosen without looking at real data. A live run then produced
# linear_seconds of 1.85 -> 13.91 -> 7.41 while the list doubled from
# 100k to 200k elements. Linear search does not get faster on a longer
# list, but the drop was 46.8% — just under the threshold, so nothing was
# reported. The number that was supposed to catch this class of bug was
# tuned to miss it by 3 percentage points.
#
# 0.25 catches that case while still ignoring ordinary timing jitter,
# which on this machine sits around 10%.
#
# This is still an arbitrary constant, and a threshold on direction alone
# is a weak test: when the input doubles, the right question is whether
# the time roughly doubled, not merely whether it fell. Answering that
# needs the actual input size per experiment, which is not recorded yet.
DEFAULT_TOLERANCE = 0.25


@dataclass
class TrendIssue:
    """One inconsistency found across a series of measurements."""

    metric_name: str
    description: str
    values: list[tuple[float, float]]  # (input_size, measured_value)

    def __str__(self) -> str:
        points = ", ".join(f"{size:g}->{value:g}" for size, value in self.values)
        return f"{self.metric_name}: {self.description} [{points}]"


def check_monotonic(
    metric_name: str,
    series: list[tuple[float, float]],
    should_increase: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TrendIssue | None:
    """
    Check that a metric moves in the expected direction as input size grows.

    `series` is a list of (input_size, measured_value), in any order; it
    gets sorted by size here so the caller does not have to.

    `should_increase` is required rather than inferred. Time grows with
    input size; accuracy might not. Guessing the direction would silently
    invert every verdict, so the caller states it.

    Returns None when the series is consistent, or a TrendIssue naming
    the point that breaks the trend. Fewer than three points returns None:
    two measurements cannot establish a trend to break.
    """
    if len(series) < 3:
        return None

    points = sorted(series)
    violations = []

    for (prev_size, prev_value), (size, value) in zip(points, points[1:]):
        if prev_value == 0:
            continue

        change = (value - prev_value) / abs(prev_value)
        # A violation is a move in the wrong direction, large enough that
        # measurement noise does not explain it.
        wrong_way = change < -tolerance if should_increase else change > tolerance
        if wrong_way:
            violations.append((prev_size, size, change))

    if not violations:
        return None

    direction = "grow" if should_increase else "shrink"
    worst = max(violations, key=lambda v: abs(v[2]))
    return TrendIssue(
        metric_name=metric_name,
        description=(
            f"expected to {direction} with input size, but fell "
            f"{abs(worst[2]):.0%} between size {worst[0]:g} and {worst[1]:g}"
            if should_increase
            else f"expected to {direction} with input size, but rose "
            f"{abs(worst[2]):.0%} between size {worst[0]:g} and {worst[1]:g}"
        ),
        values=points,
    )


def check_series(
    series_by_metric: dict[str, list[tuple[float, float]]],
    increasing_metrics: set[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[TrendIssue]:
    """
    Run the trend check over several metrics at once.

    `increasing_metrics` names the metrics expected to grow with input
    size. Anything not listed is skipped rather than guessed at: a metric
    whose expected direction is unknown cannot be checked, and inventing
    a direction produces confident nonsense.
    """
    if not increasing_metrics:
        return []

    issues = []
    for name, series in series_by_metric.items():
        if name not in increasing_metrics:
            continue
        issue = check_monotonic(name, series, should_increase=True, tolerance=tolerance)
        if issue is not None:
            issues.append(issue)
    return issues
