"""
Tests for evaluation/trends.py.

This module exists because a 7B model, asked the same question twice
about the same three numbers, answered "not consistent" once and
"generally consistent" the next time. Comparing numbers is arithmetic,
not judgement, so it belongs in code — and these tests are the proof
that it now behaves the same way every time.
"""

from __future__ import annotations

from evaluation.trends import check_monotonic, check_series


def test_the_real_contradiction_is_caught():
    # The actual measurements from a live 3-cycle run. Binary search does
    # not get faster on a bigger list; the third point is noise.
    series = [(50000, 0.00068), (200000, 0.00136), (400000, 0.00055)]
    issue = check_monotonic("binary_search_time", series)

    assert issue is not None
    assert "fell" in issue.description
    assert issue.metric_name == "binary_search_time"


def test_a_consistent_series_is_not_flagged():
    series = [(50000, 0.0006), (200000, 0.0009), (400000, 0.0012)]
    assert check_monotonic("t", series) is None


def test_small_noise_is_tolerated():
    # A 10% dip is ordinary timing jitter. Flagging it would recreate the
    # false-alarm problem the model critic had.
    series = [(50000, 0.0010), (200000, 0.0009), (400000, 0.0013)]
    assert check_monotonic("t", series) is None


def test_two_points_cannot_break_a_trend():
    # Two measurements do not establish a trend, so there is nothing to
    # contradict.
    assert check_monotonic("t", [(100, 1.0), (200, 0.1)]) is None


def test_unsorted_input_is_handled():
    # The caller should not have to sort; the same series in a different
    # order must give the same answer.
    scrambled = [(400000, 0.00055), (50000, 0.00068), (200000, 0.00136)]
    assert check_monotonic("t", scrambled) is not None


def test_direction_can_be_inverted():
    # A metric expected to shrink (error rate) rising sharply is the same
    # kind of violation in the other direction.
    #
    # The second assertion used to expect None here. That was an artefact
    # of the old 0.5 threshold: the 0.5 -> 0.3 step is a 40% drop, which
    # slipped under it. At 0.25 the same series is a violation in both
    # directions, which is correct — it falls sharply and then rises
    # sharply, so neither expectation holds.
    series = [(100, 0.5), (200, 0.3), (400, 0.9)]
    assert check_monotonic("error", series, should_increase=False) is not None
    assert check_monotonic("error", series, should_increase=True) is not None


def test_a_purely_rising_series_is_clean_when_it_should_rise():
    # The half of the old test that was actually about direction: a
    # series that only goes up violates "should shrink" and satisfies
    # "should grow".
    series = [(100, 0.3), (200, 0.6), (400, 0.9)]
    assert check_monotonic("error", series, should_increase=False) is not None
    assert check_monotonic("error", series, should_increase=True) is None


def test_a_zero_value_does_not_divide_by_zero():
    series = [(100, 0.0), (200, 1.0), (400, 0.1)]
    check_monotonic("t", series)  # must not raise


def test_check_series_only_checks_metrics_with_a_stated_direction():
    # A metric whose expected direction is unknown cannot be checked;
    # guessing one produces confident nonsense.
    data = {
        "time": [(1, 1.0), (2, 2.0), (4, 0.1)],
        "mystery": [(1, 1.0), (2, 2.0), (4, 0.1)],
    }
    issues = check_series(data, increasing_metrics={"time"})

    assert len(issues) == 1
    assert issues[0].metric_name == "time"


def test_check_series_with_no_stated_metrics_checks_nothing():
    data = {"time": [(1, 1.0), (2, 2.0), (4, 0.1)]}
    assert check_series(data) == []


def test_the_issue_message_names_the_numbers():
    # An assertion with no evidence attached is not useful to a human
    # reading the report later.
    series = [(50000, 0.00068), (200000, 0.00136), (400000, 0.00055)]
    text = str(check_monotonic("binary_search_time", series))

    assert "binary_search_time" in text
    assert "400000" in text


def test_the_case_the_old_threshold_missed():
    """
    Real measurements from a live 3-cycle run, list doubling 100k -> 200k.

    Linear search does not get faster on a longer list. The drop was
    46.8%, and the threshold was 50%, so the check that existed to catch
    exactly this reported nothing. The constant had been picked without
    looking at any real data.
    """
    series = [(1.0, 1.8533), (2.0, 13.9113), (3.0, 7.4070)]
    issue = check_monotonic("linear_seconds", series)

    assert issue is not None
    assert "fell" in issue.description


def test_ordinary_jitter_is_still_ignored_at_the_tighter_threshold():
    # Lowering the threshold is only safe if it does not start flagging
    # noise. ~10% swings are normal timing variation on this machine.
    series = [(1.0, 1.00), (2.0, 1.10), (3.0, 1.02)]
    assert check_monotonic("t", series) is None
