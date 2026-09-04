"""
Benchmark task definitions — Milestone 10.

A fixed set of tasks with known answers, so "the system got better" can
be a measurement rather than an opinion. The spec is explicit that any
proposed improvement must be evaluated against a baseline, and this is
the baseline.

Three suites, one per component that can actually be scored objectively:

  CODING     — can the model write a script that produces the right
               number? Checked against an expected value, not against a
               human reading it.
  BUG_DETECTION — given code with a known measurement bug, does the
               result critic catch it? Half the cases are clean, because
               a critic that flags everything scores as well as a good
               one unless you test both directions.
  TREND      — given a series with a known inconsistency, does the trend
               check catch it? Same split: half are clean.

What is deliberately not here: hypothesis quality and result
interpretation. Both need a human to judge, and a benchmark scored by
the same model being benchmarked measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodingTask:
    """A coding task whose output can be checked against a known value."""

    task_id: str
    instruction: str
    metric_name: str
    expected: float
    # Some answers are exact (a sum), others approximate (a timing).
    # An exact task with a tolerance would pass on a wrong answer.
    tolerance: float = 0.0

    def passed(self, metrics: dict[str, float]) -> bool:
        if self.metric_name not in metrics:
            return False
        actual = metrics[self.metric_name]
        if self.tolerance == 0.0:
            return actual == self.expected
        return abs(actual - self.expected) <= abs(self.expected * self.tolerance)


@dataclass
class BugTask:
    """Code with or without a known measurement bug."""

    task_id: str
    methodology: str
    code: str
    stdout: str
    metrics: dict
    has_bug: bool
    bug_description: str = ""


@dataclass
class TrendTask:
    """A metric series with or without a known inconsistency."""

    task_id: str
    metric_name: str
    series: list
    should_increase: bool
    is_inconsistent: bool


CODING_TASKS = [
    CodingTask(
        "sum_1_to_100",
        "Compute the sum of the integers from 1 to 100 inclusive.",
        "total", 5050.0,
    ),
    CodingTask(
        "count_primes_under_100",
        "Count how many prime numbers are strictly less than 100.",
        "prime_count", 25.0,
    ),
    CodingTask(
        "fib_20",
        "Compute the 20th Fibonacci number, where fib(1)=1 and fib(2)=1.",
        "fib", 6765.0,
    ),
    CodingTask(
        "longest_word",
        "Given the sentence 'the quick brown fox jumps over the lazy dog', "
        "report the length of the longest word.",
        "max_length", 5.0,
    ),
    CodingTask(
        "sort_stability",
        "Sort the list [5, 3, 8, 1, 9, 2] ascending and report the third "
        "element (1-indexed).",
        "third", 3.0,
    ),
    CodingTask(
        "mean_of_squares",
        "Compute the mean of the squares of the integers 1 through 10.",
        "mean", 38.5,
    ),
    CodingTask(
        "binary_search_steps",
        "Using bisect on a sorted list of the integers 0 to 999, report the "
        "index returned when searching for 750.",
        "index", 750.0,
    ),
    CodingTask(
        "string_reversal",
        "Reverse the string 'benchmark' and report the length of the result.",
        "length", 9.0,
    ),
]

BUG_TASKS = [
    BugTask(
        "timer_outside_loop",
        "Time 1000 linear searches over a list of 100000 integers.",
        """start = time.perf_counter()
total = 0
for target in targets:
    for item in data:
        if item == target:
            break
    total += time.perf_counter() - start
print(f"METRIC seconds={total}")""",
        "METRIC seconds=8463.076",
        {"seconds": 8463.076},
        has_bug=True,
        bug_description="timer started outside the loop, accumulated from the same point",
    ),
    BugTask(
        "clean_timing",
        "Time 1000 linear searches over a list of 100000 integers.",
        """start = time.perf_counter()
for target in targets:
    for item in data:
        if item == target:
            break
elapsed = time.perf_counter() - start
print(f"METRIC seconds={elapsed}")""",
        "METRIC seconds=13.68",
        {"seconds": 13.68},
        has_bug=False,
    ),
    BugTask(
        "measures_wrong_thing",
        "Measure the accuracy of the classifier on the test set.",
        """correct = sum(1 for p, y in zip(preds, train_labels) if p == y)
print(f"METRIC accuracy={correct / len(train_labels)}")""",
        "METRIC accuracy=0.99",
        {"accuracy": 0.99},
        has_bug=True,
        bug_description="scored against training labels, not the test set",
    ),
    BugTask(
        "clean_accuracy",
        "Measure the accuracy of the classifier on the test set.",
        """correct = sum(1 for p, y in zip(preds, test_labels) if p == y)
print(f"METRIC accuracy={correct / len(test_labels)}")""",
        "METRIC accuracy=0.87",
        {"accuracy": 0.87},
        has_bug=False,
    ),
    BugTask(
        "impossible_magnitude",
        "Time 500 binary searches over a list of 400000 integers.",
        """start = time.perf_counter()
for target in targets:
    bisect.bisect_left(data, target)
print(f"METRIC seconds={(time.perf_counter() - start) / 1000}")""",
        "METRIC seconds=0.0000003",
        {"seconds": 3e-07},
        has_bug=True,
        bug_description="elapsed time divided by 1000, so the number is 1000x too small",
    ),
    BugTask(
        "clean_binary_timing",
        "Time 500 binary searches over a list of 400000 integers.",
        """start = time.perf_counter()
for target in targets:
    bisect.bisect_left(data, target)
print(f"METRIC seconds={time.perf_counter() - start}")""",
        "METRIC seconds=0.0009",
        {"seconds": 0.0009},
        has_bug=False,
    ),
]

TREND_TASKS = [
    TrendTask(
        "the_real_case",
        "linear_seconds",
        [(1.0, 1.8533), (2.0, 13.9113), (3.0, 7.4070)],
        should_increase=True,
        is_inconsistent=True,
    ),
    TrendTask(
        "binary_search_impossible_speedup",
        "binary_seconds",
        [(1.0, 0.00068), (2.0, 0.00136), (3.0, 0.00055)],
        should_increase=True,
        is_inconsistent=True,
    ),
    TrendTask(
        "clean_growth",
        "seconds",
        [(1.0, 1.0), (2.0, 2.1), (3.0, 4.3)],
        should_increase=True,
        is_inconsistent=False,
    ),
    TrendTask(
        "clean_with_jitter",
        "seconds",
        [(1.0, 1.00), (2.0, 1.08), (3.0, 1.03)],
        should_increase=True,
        is_inconsistent=False,
    ),
    TrendTask(
        "error_rate_rising",
        "error_rate",
        [(1.0, 0.30), (2.0, 0.28), (3.0, 0.55)],
        should_increase=False,
        is_inconsistent=True,
    ),
    TrendTask(
        "error_rate_falling_cleanly",
        "error_rate",
        [(1.0, 0.50), (2.0, 0.35), (3.0, 0.22)],
        should_increase=False,
        is_inconsistent=False,
    ),
]
