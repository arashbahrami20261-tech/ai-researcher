"""
Hypothesis generation from results — Milestone 8.

This is what closes the loop. Until now the system ran one experiment and
stopped; the next question always came from a human. Here a finished
experiment becomes the input to the next one.

The spec sets out the steps: analyse the result, identify unexpected
observations, generate several possible explanations, rank them, and
propose the highest-value follow-up.

Two design decisions worth stating.

First, a result that failed the result critic never reaches this module.
A hypothesis built on a bad measurement propagates the error into every
experiment after it, and the system would spend its budget chasing an
artefact of a misplaced timer.

Second, previous failures are passed in. The spec calls failed
experiments valuable knowledge, and the cheapest form of that value is
not repeating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.llm_provider import LLMProvider

# The spec's experiment-selection criteria. Kept as data rather than baked
# into the prompt string so the ranking can be tuned without a rewrite.
SELECTION_CRITERIA = [
    "expected_information_gain",
    "novelty",
    "feasibility_at_small_scale",
    "cost",
]

# Criteria where a high score is a drawback rather than a merit.
LOWER_IS_BETTER = {"cost"}

SYSTEM_PROMPT = (
    "You propose follow-up experiments based on results that have already "
    "been measured. Ground every proposal in the actual numbers you are "
    "given; do not invent results or cite work you were not shown. "
    "A good follow-up is one whose outcome you cannot already predict \u2014 "
    "an experiment that can only confirm what is already known is worth "
    "little. Prefer experiments that can run in under a minute using only "
    "the Python standard library."
)


@dataclass
class Proposal:
    """One proposed follow-up experiment."""

    hypothesis: str
    methodology: str
    rationale: str
    scores: dict = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        """
        Ranking score. Higher is better.

        Cost is subtracted, not added. Summing every criterion made an
        expensive experiment outrank a cheap one of identical merit,
        which is backwards: cost is a reason to prefer an experiment
        less, not more.
        """
        total = 0.0
        for name, value in self.scores.items():
            if name in LOWER_IS_BETTER:
                total -= float(value)
            else:
                total += float(value)
        return total


def generate_followups(
    llm: LLMProvider,
    question: str,
    hypothesis: str,
    methodology: str,
    metrics: dict[str, float],
    comparison_summary: str = "",
    previous_failures: list[str] | None = None,
    count: int = 3,
) -> list[Proposal]:
    """
    Propose follow-up experiments from a completed result.

    Returns proposals sorted best-first by their summed criteria scores.
    Ranking is done by the model against stated criteria rather than by a
    formula here, because the criteria (novelty, information gain) are
    judgements, not measurements \u2014 but the scores are stored so a human can
    see why an experiment was chosen.

    Raises ValueError if the reply cannot be parsed. An unparseable answer
    must not silently become an empty list: "the model proposed nothing"
    and "the model's reply was broken" are different situations.
    """
    metric_lines = "\n".join(f"  {name} = {value}" for name, value in metrics.items())

    # The metric names must survive into the next cycle. On a live run the
    # model renamed linear_seconds to linear_search_time between cycles,
    # which split each series into single points and silently disabled the
    # trend check: a series of one cannot contradict anything.
    names = ", ".join(metrics)
    naming_rule = (
        f"The follow-up MUST report the same metric names as this "
        f"experiment: {names}. Do not rename them. Results cannot be "
        f"compared across experiments if the names change.\n\n"
    )
    criteria_spec = "\n".join(
        f'      "{c}": a number from 1 to 5,' for c in SELECTION_CRITERIA
    )

    failures_block = ""
    if previous_failures:
        failures_block = (
            "These approaches were already tried and failed. Do not propose "
            "them again:\n" + "\n".join(f"  - {f}" for f in previous_failures) + "\n\n"
        )

    prompt = (
        f"Research question:\n{question}\n\n"
        f"Hypothesis tested:\n{hypothesis}\n\n"
        f"Method used:\n{methodology}\n\n"
        f"Measured results:\n{metric_lines}\n\n"
        f"{comparison_summary}\n\n"
        f"{failures_block}"
        f"{naming_rule}"
        f"Propose {count} follow-up experiments. Score them so they can "
        "be ranked against each other: the scores must NOT be identical "
        "across proposals. If two seem equally valuable, decide which one "
        "you would run first and score it higher.\n\n"
        "Return JSON with exactly "
        "this shape:\n"
        "{\n"
        '  "proposals": [\n'
        "    {\n"
        '      "hypothesis": "one concrete, testable statement",\n'
        '      "methodology": "what the experiment should do, concretely",\n'
        '      "rationale": "what this would tell us that we do not know",\n'
        f"{criteria_spec}\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    raw = llm.generate_structured(prompt, system=SYSTEM_PROMPT, max_tokens=2000)
    return _to_proposals(raw, count)


def _to_proposals(raw: dict, count: int) -> list[Proposal]:
    """Validate and rank. Malformed entries are dropped, not guessed at."""
    entries = raw.get("proposals")
    if not isinstance(entries, list):
        raise ValueError(
            f"Expected a 'proposals' list, got {type(entries).__name__}."
        )

    proposals: list[Proposal] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        hypothesis = str(entry.get("hypothesis", "")).strip()
        methodology = str(entry.get("methodology", "")).strip()
        # A proposal without both of these is not actionable, so it is
        # dropped rather than stored as a half-formed idea.
        if not hypothesis or not methodology:
            continue

        scores = {}
        for criterion in SELECTION_CRITERIA:
            try:
                scores[criterion] = float(entry[criterion])
            except (KeyError, TypeError, ValueError):
                continue

        proposals.append(
            Proposal(
                hypothesis=hypothesis,
                methodology=methodology,
                rationale=str(entry.get("rationale", "")).strip(),
                scores=scores,
            )
        )

    if not proposals:
        raise ValueError("No usable proposals in the model reply.")

    proposals.sort(key=lambda p: p.total_score, reverse=True)
    return proposals[:count]
