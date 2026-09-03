"""
Coding agent — Milestone 6.

Bridges the model and the sandbox: asks for code, runs it, and when it
fails, hands the error back and asks for a fix.

The retry loop is not politeness, it is necessity. A small local model
gets Python subtly wrong on the first attempt often enough that a
single-shot agent would be useless. Giving it the actual traceback is
far more effective than asking it to "try again" — the error text tells
it exactly what went wrong.

Attempts are capped. An agent that retries forever burns tokens and
wall-clock time on code that is never going to work, and the spec asks
for cost control, not persistence for its own sake.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from security.sandbox import ExecutionResult, run_code
from tools.llm_provider import LLMProvider

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You write small, self-contained Python scripts. Constraints you must "
    "respect: there is no network access, the filesystem is read-only "
    "except /tmp, and only the Python standard library is available. "
    "Print results to stdout. Do not write explanations."
)


@dataclass
class CodingOutcome:
    """
    The result of one coding task, including everything that failed first.

    `attempts` is kept rather than discarded because failed attempts are
    evidence: they show whether the model was close or lost, and the
    experiment memory the spec asks for should record that.
    """

    code: str
    result: ExecutionResult | None
    attempts: list[tuple[str, ExecutionResult]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.result.succeeded


def _extract_code(reply: str) -> str:
    """
    Pull Python out of the model's reply.

    Models wrap code in markdown fences despite being told not to, so
    strip them rather than feeding ```python to the interpreter.
    """
    text = reply.strip()
    if "```" not in text:
        return text

    # Take the content of the first fenced block.
    parts = text.split("```")
    if len(parts) < 2:
        return text
    block = parts[1]
    # Drop a leading language tag such as "python".
    if "\n" in block:
        first_line, rest = block.split("\n", 1)
        if first_line.strip().lower() in {"python", "py", ""}:
            return rest.strip()
    return block.strip()


def write_and_run(
    llm: LLMProvider,
    task: str,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: int = 30,
    output_check=None,
) -> CodingOutcome:
    """
    Ask the model for code that accomplishes `task`, run it sandboxed, and
    retry with the error text if it fails.

    Returns a CodingOutcome whether or not it eventually worked. The
    caller decides what a failure means — this function does not raise on
    code that simply does not run, because "the model could not solve
    this" is a legitimate research result, not an exception.
    """
    outcome = CodingOutcome(code="", result=None)
    prompt = f"Write a Python script that does the following:\n\n{task}"

    for attempt in range(max_attempts):
        reply = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1500)
        code = _extract_code(reply)

        result = run_code(code, timeout_seconds=timeout_seconds)
        outcome.attempts.append((code, result))
        outcome.code = code
        outcome.result = result

        # A script can exit 0 and still be useless. The runner needs
        # METRIC lines; without this check a run that printed the right
        # numbers in the wrong format counted as success and burned a
        # whole research cycle. Exit code alone is too weak a definition
        # of "worked".
        output_problem = ""
        if result.succeeded and output_check is not None:
            output_problem = output_check(result.stdout) or ""

        if result.succeeded and not output_problem:
            return outcome

        if attempt < max_attempts - 1:
            # Feed the real failure back. stderr is truncated because a
            # long traceback wastes context without adding information —
            # the last lines are the ones that matter.
            error = (output_problem or result.stderr or "timed out").strip()[-1500:]
            prompt = (
                f"This Python script failed:\n\n{code}\n\n"
                f"The error was:\n\n{error}\n\n"
                "Return a corrected version of the whole script."
            )

    return outcome
