"""
Critic agent — Milestone 5.

Why a separate agent instead of one more instruction in the main prompt:
the model that produced a result is a poor judge of it. Asking the same
call to both generate a hypothesis and evaluate it reliably produces
self-congratulation. A separate call, with only the *output* in front of it
and an explicit mandate to reject, is the cheapest available approximation
of peer review.

The critic has real authority here: its verdict is stored, and a REJECTED
hypothesis is not silently kept as if it had passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tools.llm_provider import LLMProvider

# The checklist from the project spec. Only the questions that can actually
# be answered at this stage are included — the experiment-specific ones
# (data leakage, sample size, statistical significance, baseline choice)
# are held back until Milestone 7, when experiments actually run and there
# is something real to check them against. Asking the model those questions
# now would produce confident answers about data that does not exist.
LITERATURE_STAGE_CHECKLIST = [
    "is_hypothesis_clear",
    "is_hypothesis_testable",
    "is_grounded_in_provided_papers",
    "is_novel_versus_provided_papers",
    "is_feasible_at_small_scale",
]

# Held back deliberately; documented so it is obvious this is a staged
# rollout rather than an oversight.
EXPERIMENT_STAGE_CHECKLIST = [
    "is_baseline_appropriate",
    "is_free_of_data_leakage",
    "is_reproducible",
    "are_metrics_appropriate",
    "is_sample_size_sufficient",
    "is_improvement_statistically_meaningful",
    "are_alternative_explanations_ruled_out",
]

VERDICT_ACCEPTED = "accepted"
VERDICT_REVISE = "revise"
VERDICT_REJECTED = "rejected"
VALID_VERDICTS = {VERDICT_ACCEPTED, VERDICT_REVISE, VERDICT_REJECTED}

SYSTEM_PROMPT = (
    "You are a rigorous peer reviewer for AI research. Your job is to find "
    "problems, not to be encouraging. A hypothesis that is vague, already "
    "published, unfalsifiable, or not supported by the provided papers must "
    "be rejected. Reviewers who accept everything are useless. Judge only "
    "against the papers provided; do not rely on your own recollection of "
    "the literature, and say so explicitly if the papers are insufficient "
    "to judge novelty."
)


@dataclass
class Critique:
    """The critic's structured verdict on one research output."""

    verdict: str
    checklist: dict = field(default_factory=dict)
    reasoning: str = ""
    limitations: str = ""

    @property
    def is_accepted(self) -> bool:
        return self.verdict == VERDICT_ACCEPTED

    @property
    def failed_checks(self) -> list[str]:
        """Checklist keys the critic answered 'no' to."""
        return [key for key, value in self.checklist.items() if value is False]


def review_hypothesis(
    llm: LLMProvider,
    question: str,
    summary: str,
    hypothesis: str,
    papers_block: str,
) -> Critique:
    """
    Review a generated summary + hypothesis against the checklist.

    `papers_block` is the same rendered list of retrieved abstracts that was
    given to the generating call. The critic sees exactly the evidence the
    generator saw — otherwise it cannot tell a grounded claim from an
    invented one.

    Returns a `Critique`. Raises `ValueError` if the model's reply cannot be
    parsed into a usable verdict; the caller decides what to do about that
    rather than getting a silently fabricated "accepted".
    """
    checklist_spec = "\n".join(f'  "{key}": true or false,' for key in LITERATURE_STAGE_CHECKLIST)

    prompt = (
        f"Research question:\n{question}\n\n"
        f"Papers that were retrieved and given to the generating model:\n"
        f"{papers_block}\n\n"
        f"Generated summary:\n{summary}\n\n"
        f"Generated hypothesis:\n{hypothesis}\n\n"
        "Review the hypothesis. Return JSON with exactly this shape:\n"
        "{\n"
        f"{checklist_spec}\n"
        '  "verdict": "accepted" or "revise" or "rejected",\n'
        '  "reasoning": "why you reached this verdict",\n'
        '  "limitations": "what this hypothesis cannot show even if confirmed"\n'
        "}\n\n"
        "Use 'rejected' if any of the first three checks fail. Use 'revise' "
        "if the idea is salvageable but the hypothesis as written is not yet "
        "testable or not clearly distinct from the provided papers."
    )

    raw = llm.generate_structured(prompt, system=SYSTEM_PROMPT, max_tokens=1000)
    return _to_critique(raw)


def _to_critique(raw: dict) -> Critique:
    """
    Validate the model's JSON before trusting it.

    An unrecognised verdict is treated as a parse failure rather than being
    coerced to 'accepted'. Defaulting to acceptance on a malformed reply
    would let a broken critic wave everything through, which is worse than
    having no critic at all.
    """
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Critic returned an unrecognised verdict: {raw.get('verdict')!r}. "
            f"Expected one of {sorted(VALID_VERDICTS)}."
        )

    checklist = {
        key: bool(raw[key]) for key in LITERATURE_STAGE_CHECKLIST if key in raw
    }

    return Critique(
        verdict=verdict,
        checklist=checklist,
        reasoning=str(raw.get("reasoning", "")).strip(),
        limitations=str(raw.get("limitations", "")).strip(),
    )


def critique_to_json(critique: Critique) -> str:
    """Serialise the checklist for storage in the `critiques` table."""
    return json.dumps(critique.checklist, ensure_ascii=False)
