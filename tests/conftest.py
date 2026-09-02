"""
Shared test fixtures.

The point of `FakeLLM` is that no test in this suite ever makes a real API
call: tests must run offline, instantly, and for free. This is exactly what
the `LLMProvider` abstraction was built for — nothing in the research loop
knows or cares that it is talking to a fake.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base
from tools.llm_provider import LLMProvider


class FakeLLM(LLMProvider):
    """
    An LLMProvider that returns canned replies from a queue.

    `replies` is consumed in order, one per `generate` call. Passing a dict
    is a convenience — it gets serialised to JSON, since most calls in this
    codebase go through `generate_structured`.

    `calls` records every prompt received, so tests can assert on what the
    model was actually asked (e.g. that the critic really was shown the
    retrieved papers).
    """

    def __init__(self, replies: list) -> None:
        self._replies = [
            json.dumps(r) if isinstance(r, dict) else r for r in replies
        ]
        self.calls: list[dict] = []

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if not self._replies:
            raise AssertionError("FakeLLM ran out of canned replies")
        return self._replies.pop(0)


@pytest.fixture
def session():
    """A fresh in-memory database, torn down automatically after each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    yield s
    s.close()
