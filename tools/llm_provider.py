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

import os
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
