"""
Evaluation engine — Milestone 7.

The spec is blunt about what this exists to prevent: "The agent must never
claim that a method is better merely because it looks better. Require
measurable evidence."

So every comparison here is against an explicit baseline. There is no way
to call this module and get back "the result was good" — you get a
difference from a stated reference point, and a verdict that says whether
that difference is large enough to mean anything.

The statistics are deliberately simple. A t-test on three runs would give
a confident-looking p-value that means nothing; better to report the
spread honestly and say when there is not enough data to judge.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# A result is only called an improvement if it clears this fraction of the
# baseline. Without a threshold, noise in the third decimal place gets
# reported as progress.
DEFAULT_MIN_EFFECT = 0.01

# Below this many runs, variance cannot be estimated meaningfully. The
# engine says "inconclusive" rather than inventing confidence.
MIN_RUNS_FOR_VARIANCE = 3

BETTER = "better"
WORSE = "worse"
NO_DIFFERENCE = "no_difference"
INCONCLUSIVE = "inconclusive"


@dataclass
class Comparison:
    """One metric, measured against a baseline."""

    metric_name: str
    baseline: float
    observed: float
    verdict: str
    absolute_change: float
    relative_change: float
    runs: int
    stdev: float | None = None
    note: str = ""

    @property
    def is_improvement(self) -> bool:
        return self.verdict == BETTER


def compare_to_baseline(
    metric_name: str,
    baseline: float,
    observed_runs: list[float],
    higher_is_better: bool = True,
    min_effect: float = DEFAULT_MIN_EFFECT,
) -> Comparison:
    """
    Compare one or more measurements against a baseline value.

    `observed_runs` is a list, not a single number, on purpose. One run
    tells you almost nothing: rerun the same experiment with a different
    seed and the number moves. Passing a list forces the caller to be
    explicit about how much evidence they actually have.

    `higher_is_better` matters because accuracy and loss point in opposite
    directions. Getting this wrong silently inverts every verdict, so it
    has no safe default guess — the caller states it.

    Raises ValueError on an empty list or a zero baseline; both are caller
    errors that would otherwise produce a meaningless number.
    """
    if not observed_runs:
        raise ValueError("No observed runs to compare.")
    if baseline == 0:
        raise ValueError(
            "Baseline is zero; relative change is undefined. "
            "Use a non-zero reference or compare absolute values."
        )

    mean = statistics.mean(observed_runs)
    runs = len(observed_runs)
    stdev = statistics.stdev(observed_runs) if runs >= 2 else None

    absolute = mean - baseline
    relative = absolute / abs(baseline)

    # Normalise direction so the rest of the logic reads the same way for
    # accuracy (up is good) and loss (down is good).
    signed = relative if higher_is_better else -relative

    if abs(signed) < min_effect:
        verdict = NO_DIFFERENCE
        note = f"Change is smaller than the {min_effect:.1%} threshold."
    elif runs < MIN_RUNS_FOR_VARIANCE:
        verdict = INCONCLUSIVE
        note = (
            f"Only {runs} run(s). At least {MIN_RUNS_FOR_VARIANCE} are needed "
            "before a difference can be distinguished from noise."
        )
    elif stdev is not None and abs(absolute) < stdev:
        # The effect is smaller than the spread between the runs themselves.
        verdict = INCONCLUSIVE
        note = (
            f"The difference ({absolute:.4f}) is smaller than the spread "
            f"across runs ({stdev:.4f}), so it cannot be told apart from noise."
        )
    else:
        verdict = BETTER if signed > 0 else WORSE
        note = f"Consistent across {runs} runs."

    return Comparison(
        metric_name=metric_name,
        baseline=baseline,
        observed=mean,
        verdict=verdict,
        absolute_change=absolute,
        relative_change=relative,
        runs=runs,
        stdev=stdev,
        note=note,
    )


def format_comparison(c: Comparison) -> str:
    """One human-readable line. Always states the baseline it was measured against."""
    spread = f" (sd {c.stdev:.4f})" if c.stdev is not None else ""
    return (
        f"{c.metric_name}: {c.observed:.4f} vs baseline {c.baseline:.4f} "
        f"({c.relative_change:+.2%}, {c.runs} run(s){spread}) "
        f"-> {c.verdict.upper()}. {c.note}"
    )
