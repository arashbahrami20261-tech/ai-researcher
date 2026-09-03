"""
Result critic — Milestone 8.

The existing critic in agents/critic.py reviews *hypotheses*. This one
reviews *results*, and it exists because of a specific failure.

An early experiment reported 8463 seconds for a loop that takes 13. The
model had started the timer outside the loop and accumulated elapsed time
from the same point a thousand times. Every automated check passed: the
code ran, the exit code was 0, the METRIC line parsed. The number was
nonsense and nothing in the system noticed.

The evaluation engine answers "is this difference statistically
meaningful?" That is a different question from "is this number
believable at all?", and no amount of statistics answers the second one.
Only reading the code against the output does.

This critic runs before hypothesis generation on purpose. A hypothesis
built on a bad measurement propagates the error into every experiment
that follows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.llm_provider import LLMProvider

CHECKLIST = [
    # Does the code measure the thing the experiment claims to measure?
    "code_measures_what_was_asked",
    # Is the magnitude physically plausible for this kind of operation?
    "magnitude_is_plausible",
    # Timer placement, off-by-one, accumulating inside a loop, and so on.
    "no_obvious_measurement_bug",
    # Could something other than the stated cause produce this number?
    "no_alternative_explanation",
    # Does this number sit consistently with earlier results in the same
    # chain? Only meaningful when prior results are supplied; without
    # them the critic has nothing to compare against and this is skipped.
    "consistent_with_earlier_results",
]

VERDICT_TRUSTWORTHY = "trustworthy"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_INVALID = "invalid"
VALID_VERDICTS = {VERDICT_TRUSTWORTHY, VERDICT_SUSPICIOUS, VERDICT_INVALID}

SYSTEM_PROMPT = (
    "You review experimental results for measurement errors. You are given "
    "the code that ran and the numbers it produced. Your job is to find "
    "reasons the numbers might be wrong, not to confirm them. "
    "Pay particular attention to: timers started or stopped in the wrong "
    "place, values accumulated inside a loop that should be measured once, "
    "results whose magnitude is implausible for the operation described, "
    "and code that measures something adjacent to what was asked. "
    "A result that looks impressive is more suspect, not less. "
    "But say so only when you can point at the line responsible. "
    "Speculation such as 'the values might be out of range' or 'the "
    "library might not behave as expected' is not a finding. If the "
    "code reads correctly and the magnitude is plausible for the "
    "operation described, the result is trustworthy. Flagging every "
    "result you have not personally verified makes the review useless."
)


@dataclass
class ResultCritique:
    """The critic's verdict on one set of measurements."""

    verdict: str
    checklist: dict = field(default_factory=dict)
    reasoning: str = ""
    suspected_bug: str = ""

    @property
    def is_trustworthy(self) -> bool:
        return self.verdict == VERDICT_TRUSTWORTHY

    @property
    def failed_checks(self) -> list[str]:
        return [key for key, value in self.checklist.items() if value is False]


def review_result(
    llm: LLMProvider,
    methodology: str,
    code: str,
    stdout: str,
    metrics: dict[str, float],
    prior_results: list[str] | None = None,
) -> ResultCritique:
    """
    Review one experiment's output against the code that produced it.

    The critic sees the code, not just the numbers. Reviewing metrics
    alone cannot catch a misplaced timer — the bug is invisible unless you
    read the loop it sits in.

    Raises ValueError if the reply cannot be parsed into a usable verdict.
    As with the hypothesis critic, an unparseable answer must never
    default to "trustworthy": a broken reviewer that waves everything
    through is worse than no reviewer.
    """
    checklist_spec = "\n".join(f'  "{key}": true or false,' for key in CHECKLIST)
    metric_lines = "\n".join(f"  {name} = {value}" for name, value in metrics.items())

    # Prior results from the same chain, if the caller supplied them.
    # A live 3-cycle run passed all three reviews while the timings
    # contradicted each other, because each review saw only its own
    # numbers. A critic cannot catch an inconsistency it cannot see.
    history_block = ""
    if prior_results:
        history_block = (
            "Earlier results in this same chain of experiments, oldest "
            "first:\n" + "\n".join(f"  {r}" for r in prior_results) + "\n\n"
            "If the new number does not fit the trend of these, say so: "
            "a measurement that reverses an established trend is more likely to be noise or a bug than a discovery.\n\n"
        )

    prompt = (
        f"The experiment was asked to do this:\n{methodology}\n\n"
        f"The model wrote this code:\n{code}\n\n"
        f"It printed:\n{stdout}\n\n"
        f"Parsed metrics:\n{metric_lines}\n\n"
        f"{history_block}"
        "Review these measurements. Return JSON with exactly this shape:\n"
        "{\n"
        f"{checklist_spec}\n"
        '  "verdict": "trustworthy" or "suspicious" or "invalid",\n'
        '  "reasoning": "why you reached this verdict",\n'
        '  "suspected_bug": "the specific bug you found, or empty if none"\n'
        "}\n\n"
        "Use 'invalid' only if you can name the specific line or "
        "construct that produces a wrong number. Use 'suspicious' only "
        "if the magnitude is genuinely implausible for the operation "
        "described. Otherwise use 'trustworthy'. Uncertainty about code "
        "you did not write is not grounds for either of the first two."
    )

    raw = llm.generate_structured(prompt, system=SYSTEM_PROMPT, max_tokens=1200)
    return _to_critique(raw)


def _to_critique(raw: dict) -> ResultCritique:
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Result critic returned an unrecognised verdict: "
            f"{raw.get('verdict')!r}. Expected one of {sorted(VALID_VERDICTS)}."
        )

    checklist = {key: bool(raw[key]) for key in CHECKLIST if key in raw}

    return ResultCritique(
        verdict=verdict,
        checklist=checklist,
        reasoning=str(raw.get("reasoning", "")).strip(),
        suspected_bug=str(raw.get("suspected_bug", "")).strip(),
    )
