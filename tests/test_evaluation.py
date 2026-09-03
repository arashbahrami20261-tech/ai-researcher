"""
Tests for evaluation/compare.py.

The behaviour worth testing here is refusal: this module exists to stop
the system claiming an improvement it cannot support, so most of these
assert that a verdict is *withheld*, not that it is given.
"""

from __future__ import annotations

import pytest

from evaluation.compare import (
    BETTER,
    INCONCLUSIVE,
    NO_DIFFERENCE,
    WORSE,
    compare_to_baseline,
)


def test_clear_improvement_across_several_runs_is_called_better():
    c = compare_to_baseline("accuracy", 0.80, [0.92, 0.91, 0.93])
    assert c.verdict == BETTER
    assert c.is_improvement


def test_a_single_run_is_never_conclusive_however_good_it_looks():
    # The real experiment that motivated this: binary search beat its
    # baseline by 94% on one run. Impressive, and still not evidence.
    c = compare_to_baseline("seconds", 0.01, [0.0006], higher_is_better=False)
    assert c.verdict == INCONCLUSIVE
    assert not c.is_improvement


def test_change_below_the_threshold_is_not_an_improvement():
    c = compare_to_baseline("accuracy", 0.80, [0.804, 0.803, 0.805])
    assert c.verdict == NO_DIFFERENCE


def test_a_regression_is_reported_as_worse():
    c = compare_to_baseline("accuracy", 0.90, [0.70, 0.71, 0.69])
    assert c.verdict == WORSE


def test_direction_is_respected_for_metrics_where_lower_is_better():
    # Loss dropping is good. Reading it as "the number went down, so it
    # got worse" would invert every verdict in the system.
    c = compare_to_baseline("loss", 0.50, [0.30, 0.31, 0.29], higher_is_better=False)
    assert c.verdict == BETTER


def test_an_effect_smaller_than_the_noise_is_inconclusive():
    # Mean is 0.83 against a 0.80 baseline, but the runs range from 0.70
    # to 0.95. The spread swallows the effect.
    c = compare_to_baseline("accuracy", 0.80, [0.70, 0.95, 0.84])
    assert c.verdict == INCONCLUSIVE


def test_the_baseline_is_always_carried_in_the_result():
    # A number with no stated reference point is not a finding.
    c = compare_to_baseline("accuracy", 0.80, [0.92, 0.91, 0.93])
    assert c.baseline == 0.80
    assert c.runs == 3


def test_empty_runs_raise():
    with pytest.raises(ValueError):
        compare_to_baseline("accuracy", 0.80, [])


def test_zero_baseline_raises_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="zero"):
        compare_to_baseline("accuracy", 0.0, [0.5])
