"""
Controlled self-improvement — Milestone 10.

The spec is blunt: "Never automatically deploy an improvement simply
because the model claims it is better." So nothing here trusts a claim.
A change is applied, the benchmark is re-run, and the numbers decide.

The loop:

    baseline -> propose a change -> re-run the benchmark -> accept or
    reject -> record either way

What is deliberately narrow: the only thing this can change is a prompt
string. It cannot edit logic, add dependencies, or touch the sandbox.
An agent that rewrites its own control flow is a much larger safety
problem than one that rewrites its own instructions, and the spec asks
for autonomy that stays bounded.

Rejections are stored alongside acceptances. A record of what did not
work is what stops the same idea being tried again in three months.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from benchmarks.runner import SuiteScore, score_bug_detection
from tools.llm_provider import LLMProvider

# A change must beat the baseline by more than this to be accepted.
# Without a margin, noise in one direction gets recorded as progress —
# the same mistake the evaluation engine exists to prevent.
MIN_IMPROVEMENT = 0.05

ACCEPTED = "accepted"
REJECTED = "rejected"

PROPOSER_SYSTEM_PROMPT = (
    "You improve the instructions given to a code reviewer. You will be "
    "shown the current instructions, its score on a fixed benchmark, and "
    "the specific cases it failed. Propose a revised instruction that "
    "would catch those cases. Do not make it longer for its own sake, and "
    "do not tell it to flag more things in general \u2014 a reviewer that "
    "flags everything scores no better than one that flags nothing."
)


@dataclass
class Attempt:
    """One proposed change and what the benchmark said about it."""

    description: str
    new_prompt: str
    baseline_score: float
    new_score: float
    verdict: str
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.new_score - self.baseline_score


def propose_prompt_change(
    llm: LLMProvider,
    current_prompt: str,
    score: SuiteScore,
    failures: list[str],
) -> tuple[str, str]:
    """
    Ask the model for a revised prompt.

    Returns (description, new_prompt). The description is stored so a
    human reading the record later knows what was tried, not just that
    something was.
    """
    failure_block = "\n".join(f"  - {f}" for f in failures) or "  (none recorded)"

    prompt = (
        f"Current reviewer instructions:\n{current_prompt}\n\n"
        f"Benchmark score: {score.mean:.1%}\n"
        f"Detail: {json.dumps(score.detail)}\n\n"
        f"Cases it failed:\n{failure_block}\n\n"
        "Return JSON with exactly this shape:\n"
        "{\n"
        '  "description": "one sentence on what you changed and why",\n'
        '  "new_prompt": "the full revised instructions"\n'
        "}"
    )

    reply = llm.generate_structured(prompt, system=PROPOSER_SYSTEM_PROMPT, max_tokens=1500)

    description = str(reply.get("description", "")).strip()
    new_prompt = str(reply.get("new_prompt", "")).strip()

    if not new_prompt:
        raise ValueError("The proposer returned no new prompt.")

    return description, new_prompt


def evaluate_change(
    llm: LLMProvider,
    baseline: SuiteScore,
    description: str,
    new_prompt: str,
    repeats: int = 3,
    min_improvement: float = MIN_IMPROVEMENT,
) -> Attempt:
    """
    Apply a proposed prompt, re-run the benchmark, and decide.

    The prompt is swapped in temporarily and always restored, including
    when the benchmark raises. A self-improvement loop that leaves a
    rejected change in place on failure would silently degrade the system
    it is meant to improve.
    """
    import agents.result_critic as critic_module

    original = critic_module.SYSTEM_PROMPT
    try:
        critic_module.SYSTEM_PROMPT = new_prompt
        new_score = score_bug_detection(llm, repeats=repeats)
    finally:
        critic_module.SYSTEM_PROMPT = original

    delta = new_score.mean - baseline.mean

    if delta > min_improvement:
        verdict = ACCEPTED
        reason = f"Beat the baseline by {delta:+.1%}, above the {min_improvement:.0%} margin."
    elif delta > 0:
        verdict = REJECTED
        reason = (
            f"Improved by {delta:+.1%}, which is inside the {min_improvement:.0%} "
            "margin and cannot be told apart from noise."
        )
    else:
        verdict = REJECTED
        reason = f"Scored {delta:+.1%} against the baseline."

    return Attempt(
        description=description,
        new_prompt=new_prompt,
        baseline_score=baseline.mean,
        new_score=new_score.mean,
        verdict=verdict,
        reason=reason,
        detail=new_score.detail,
    )


def format_attempt(attempt: Attempt) -> str:
    return (
        f"[{attempt.verdict.upper()}] {attempt.description}\n"
        f"  baseline {attempt.baseline_score:.1%} -> {attempt.new_score:.1%} "
        f"({attempt.delta:+.1%})\n"
        f"  {attempt.reason}\n"
        f"  detail: {attempt.detail}"
    )
