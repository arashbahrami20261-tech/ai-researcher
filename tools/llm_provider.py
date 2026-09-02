"""
LLM abstraction layer.

Why this file exists:
The research loop should never call the Anthropic SDK directly. Every other
part of the system (Research Director, Critic, Hypothesis Generator, ...)
talks to an `LLMProvider` interface instead. That means swapping models,
adding a second provider, or mocking the LLM in tests never requires
touching the research logic itself — only this file changes.
"""

from __future__ import annotations

import json
import os

import requests
from abc import ABC, abstractmethod

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the process
# environment. This is what lets ANTHROPIC_API_KEY be read below without
# ever hard-coding it anywhere in the codebase.
load_dotenv()


class LLMProvider(ABC):
    """
    Abstract base class every LLM backend must implement.

    Any future provider (a different vendor's API, or a local model) plugs
    in by subclassing this and implementing `generate`. Nothing else in the
    codebase needs to know which concrete provider is in use.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> str:
        """
        Send `prompt` to the model and return its plain-text reply.

        `system` is an optional system prompt (instructions that shape the
        model's behavior for this call, separate from the user-facing
        prompt). `max_tokens` caps the length of the response.
        """
        raise NotImplementedError

    def generate_structured(
        self, prompt: str, system: str | None = None, max_tokens: int = 1000
    ) -> dict:
        """
        Same as `generate`, but the model is asked for JSON and the reply is
        parsed into a Python dict.

        Why this exists: splitting a free-text reply on labels like
        "HYPOTHESIS" is fragile — it breaks the moment the model writes
        "## Hypothesis" or reorders the sections. Asking for JSON and
        parsing it gives a stable contract between the model and the code.

        Implemented once here on the base class rather than per-provider,
        since the JSON instruction and the parsing are provider-independent.

        Raises `ValueError` if the reply cannot be parsed as JSON. Callers
        must handle that rather than assume success — a malformed reply is
        a real failure mode, not an edge case.
        """
        json_instruction = (
            "Respond with a single valid JSON object and nothing else. "
            "No explanation before or after it, no markdown code fences."
        )
        full_system = f"{system}\n\n{json_instruction}" if system else json_instruction

        raw = self.generate(prompt, system=full_system, max_tokens=max_tokens)
        return _parse_json_reply(raw)


def _parse_json_reply(raw: str) -> dict:
    """
    Turn a model's text reply into a dict, tolerating the two things models
    do anyway despite being told not to: wrapping the JSON in ```json fences,
    and adding a sentence before or after it.
    """
    text = raw.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the opening ``` line
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]

    # Fall back to grabbing the outermost {...} span, in case the model added
    # a stray sentence around the object.
    text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON object found in model reply: {raw[:200]!r}")
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model reply was not valid JSON: {raw[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")

    return parsed


class ClaudeProvider(LLMProvider):
    """Concrete LLMProvider backed by the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        # Imported here (not at module level) so that importing this file
        # doesn't require the `anthropic` package unless a Claude call is
        # actually made — keeps the abstraction layer lightweight to import.
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and fill in your key, or export it in your shell."
            )

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> str:
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        # response.content is a list of content blocks; for a plain text
        # reply there is exactly one block of type "text". Joining handles
        # the (rare) case of multiple text blocks safely.
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
class OllamaProvider(LLMProvider):
    """
    LLMProvider backed by a local Ollama server.

    Exists so the whole research loop can be exercised end-to-end without
    a paid API key. Everything above this class is untouched: that is the
    point of the LLMProvider interface.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self._model = model
        self._url = f"{host}/api/generate"
        self._timeout = timeout

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> str:
        """
        Send one prompt to the local Ollama server and return its reply.

        Ollama's request shape differs from Anthropic's: the token cap lives
        under an "options" sub-object and is called num_predict, not
        max_tokens. Translating that here is exactly why this abstraction
        exists — nothing outside this class needs to know the difference.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        response = requests.post(self._url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()

        # Ollama reports a missing model with HTTP 200 and an "error" key in
        # the body, so raise_for_status() alone would let it through and the
        # failure would surface much later as unparseable JSON.
        if "error" in data:
            raise RuntimeError(f"Ollama returned an error: {data['error']}")

        return data.get("response", "")
