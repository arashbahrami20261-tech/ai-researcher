"""
Tests for agents/coder.py.

The interesting behaviour here is the retry loop, and it is testable
without a real model or a real container: FakeLLM supplies the replies,
and run_code is patched so the tests control whether "execution"
succeeds. That keeps them fast and deterministic — the sandbox itself is
already covered by tests/test_sandbox.py.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.coder import _extract_code, write_and_run
from security.sandbox import ExecutionResult
from tests.conftest import FakeLLM


def _ok(stdout="done"):
    return ExecutionResult(stdout=stdout, stderr="", exit_code=0, timed_out=False)


def _fail(stderr="NameError: name 'x' is not defined"):
    return ExecutionResult(stdout="", stderr=stderr, exit_code=1, timed_out=False)


def test_working_code_runs_once_and_stops():
    llm = FakeLLM(["print(1)"])

    with patch("agents.coder.run_code", return_value=_ok("1")) as run:
        outcome = write_and_run(llm, "print one")

    assert outcome.succeeded
    assert len(outcome.attempts) == 1
    assert run.call_count == 1


def test_failure_triggers_a_retry():
    llm = FakeLLM(["broken code", "fixed code"])
    results = [_fail(), _ok()]

    with patch("agents.coder.run_code", side_effect=results):
        outcome = write_and_run(llm, "do a thing")

    assert outcome.succeeded
    assert len(outcome.attempts) == 2
    assert outcome.code == "fixed code"


def test_the_actual_error_is_fed_back_to_the_model():
    # This is the point of the retry loop. Asking the model to "try again"
    # without the traceback is far less effective than showing it what
    # went wrong, so the error text must reach the second prompt.
    llm = FakeLLM(["broken", "fixed"])

    with patch("agents.coder.run_code", side_effect=[_fail("ZeroDivisionError"), _ok()]):
        write_and_run(llm, "divide")

    second_prompt = llm.calls[1]["prompt"]
    assert "ZeroDivisionError" in second_prompt
    assert "broken" in second_prompt


def test_attempts_are_capped():
    # An agent that retries forever burns tokens on code that will never
    # work. Three failures means three attempts, not four.
    llm = FakeLLM(["a", "b", "c"])

    with patch("agents.coder.run_code", side_effect=[_fail(), _fail(), _fail()]) as run:
        outcome = write_and_run(llm, "impossible task")

    assert not outcome.succeeded
    assert run.call_count == 3
    assert len(outcome.attempts) == 3


def test_failed_attempts_are_kept_not_discarded():
    # Failed attempts are evidence about how close the model got, and the
    # spec treats failures as valuable research memory.
    llm = FakeLLM(["first try", "second try"])

    with patch("agents.coder.run_code", side_effect=[_fail(), _ok()]):
        outcome = write_and_run(llm, "task")

    assert outcome.attempts[0][0] == "first try"
    assert not outcome.attempts[0][1].succeeded


def test_a_timeout_is_reported_to_the_model_too():
    # A timed-out run has empty stderr, so without the fallback the model
    # would be asked to fix an error it was never shown.
    llm = FakeLLM(["while True: pass", "fixed"])
    timeout = ExecutionResult(stdout="", stderr="", exit_code=-1, timed_out=True)

    with patch("agents.coder.run_code", side_effect=[timeout, _ok()]):
        write_and_run(llm, "loop")

    assert "timed out" in llm.calls[1]["prompt"]


def test_markdown_fences_are_stripped():
    assert _extract_code("```python\nprint(1)\n```") == "print(1)"
    assert _extract_code("```\nprint(2)\n```") == "print(2)"


def test_bare_code_is_left_alone():
    assert _extract_code("print(3)") == "print(3)"


def test_output_check_can_reject_a_run_that_exited_zero():
    """
    A script can exit 0 and still be useless.

    This was a real failure: the model printed the right numbers in the
    wrong format, the sandbox reported success, and a whole research
    cycle was spent on a run that measured nothing. Exit code alone is
    too weak a definition of "worked".
    """
    llm = FakeLLM(["bad format", "good format"])
    calls = []

    def check(stdout):
        calls.append(stdout)
        return "" if stdout == "METRIC a=1" else "no METRIC lines found"

    with patch("agents.coder.run_code", side_effect=[_ok("wrong"), _ok("METRIC a=1")]):
        outcome = write_and_run(llm, "measure something", output_check=check)

    assert outcome.succeeded
    assert len(outcome.attempts) == 2
    assert len(calls) == 2


def test_the_output_problem_is_explained_to_the_model():
    # Telling the model "try again" without saying what was wrong wastes
    # the retry. The check's message must reach the next prompt.
    llm = FakeLLM(["bad", "good"])

    def check(stdout):
        return "" if "METRIC" in stdout else "UNIQUE_FORMAT_COMPLAINT"

    with patch("agents.coder.run_code", side_effect=[_ok("wrong"), _ok("METRIC a=1")]):
        write_and_run(llm, "task", output_check=check)

    assert "UNIQUE_FORMAT_COMPLAINT" in llm.calls[1]["prompt"]


def test_no_output_check_means_exit_code_alone_decides():
    # Existing callers that do not care about output format must keep
    # working unchanged.
    llm = FakeLLM(["anything"])

    with patch("agents.coder.run_code", return_value=_ok("whatever")):
        outcome = write_and_run(llm, "task")

    assert outcome.succeeded
    assert len(outcome.attempts) == 1
