"""
CLI entrypoint:

    python scripts/run_research.py "your research question here"

This file is deliberately thin. All the actual logic lives in
`research/loop.py` so it can be tested and reused; this only parses
arguments, handles failure modes, and prints.
"""

from __future__ import annotations

import sys

# Add the project root to the path so `database.*`, `literature.*`,
# `research.*`, and `tools.*` resolve when this script is run directly.
sys.path.insert(0, ".")

import requests

from database.db import SessionLocal, ensure_migrated
from research.loop import format_report, run_research
from tools.llm_provider import ClaudeProvider, OllamaProvider


def build_provider(name: str):
    """
    Pick an LLM backend by name.

    Kept as a tiny factory rather than an if-statement inline, so adding
    a third provider later touches one place. Note what is NOT here:
    nothing about research, critics, or prompts. The caller receives an
    LLMProvider and does not care which one.
    """
    if name == "ollama":
        return OllamaProvider()
    if name == "claude":
        return ClaudeProvider()
    raise ValueError(f"Unknown provider: {name!r}. Use 'claude' or 'ollama'.")


def main(question: str, provider: str = "claude") -> int:
    """Returns a process exit code: 0 on success, non-zero on failure."""
    try:
        ensure_migrated()
    except RuntimeError as exc:
        print(f"Database not ready: {exc}", file=sys.stderr)
        return 1

    try:
        llm = build_provider(provider)
    except RuntimeError as exc:
        # Missing API key — the most common first-run failure.
        print(f"LLM setup failed: {exc}", file=sys.stderr)
        return 1

    session = SessionLocal()
    try:
        print(f"\nResearch question: {question}")
        print("Searching arXiv...")
        print("Asking the model to summarize and propose a hypothesis...")
        print("Running critic review...\n")

        result = run_research(llm, session, question)

    except LookupError as exc:
        print(f"{exc}\nTry rephrasing the question with more specific terms.", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        # A failed search must never be reported as "no relevant papers".
        print(f"arXiv request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"The model's reply could not be used: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    print(format_report(result))
    print(f"\nSaved to the database as project #{result.project_id}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]

    # Minimal flag parsing rather than argparse: one optional flag does not
    # justify the extra dependency on argparse's output formatting here.
    chosen = "claude"
    if "--provider" in args:
        i = args.index("--provider")
        if i + 1 >= len(args):
            print("--provider needs a value: claude or ollama")
            sys.exit(1)
        chosen = args[i + 1]
        args = args[:i] + args[i + 2:]

    if not args:
        print('Usage: python scripts/run_research.py [--provider claude|ollama] "your question"')
        sys.exit(1)

    sys.exit(main(" ".join(args), provider=chosen))
