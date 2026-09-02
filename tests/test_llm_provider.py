"""
Tests for tools/llm_provider.py — specifically the structured-output path.

These cover the failure modes that broke the old label-splitting approach:
the model wrapping its JSON in code fences, adding a stray sentence, or
returning something that isn't JSON at all.
"""

from __future__ import annotations

import pytest

from tests.conftest import FakeLLM


def test_generate_structured_parses_plain_json():
    llm = FakeLLM(['{"summary": "s", "hypothesis": "h"}'])

    result = llm.generate_structured("anything")

    assert result == {"summary": "s", "hypothesis": "h"}


def test_generate_structured_strips_markdown_fences():
    llm = FakeLLM(['```json\n{"summary": "s"}\n```'])

    result = llm.generate_structured("anything")

    assert result["summary"] == "s"


def test_generate_structured_recovers_from_surrounding_prose():
    llm = FakeLLM(['Sure, here you go: {"summary": "s"} Hope that helps!'])

    result = llm.generate_structured("anything")

    assert result["summary"] == "s"


def test_generate_structured_raises_on_non_json():
    llm = FakeLLM(["I'm afraid I can't do that."])

    with pytest.raises(ValueError, match="No JSON object found"):
        llm.generate_structured("anything")


def test_generate_structured_raises_on_malformed_json():
    llm = FakeLLM(['{"summary": "s", }{']);

    with pytest.raises(ValueError):
        llm.generate_structured("anything")


def test_generate_structured_rejects_json_arrays():
    # A list is valid JSON but not the object shape callers expect.
    llm = FakeLLM(['[1, 2, 3]'])

    with pytest.raises(ValueError):
        llm.generate_structured("anything")


def test_json_instruction_is_appended_to_system_prompt():
    llm = FakeLLM(['{"ok": true}'])

    llm.generate_structured("anything", system="You are a reviewer.")

    system = llm.calls[0]["system"]
    assert "You are a reviewer." in system
    assert "JSON" in system
