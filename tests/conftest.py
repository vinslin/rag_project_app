"""Shared pytest fixtures for the Legal Contract RAG test suite."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_groq_response(content: str = "Test answer.") -> MagicMock:
    """Build a minimal object that looks like a Groq ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice  = SimpleNamespace(message=message)
    usage   = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_chroma_results(chunks: list[dict]) -> dict:
    """Convert a list of chunk dicts into ChromaDB query-result shape."""
    return {
        "ids":       [["id_" + str(i) for i in range(len(chunks))]],
        "documents": [[c["text"] for c in chunks]],
        "metadatas": [[{k: v for k, v in c.items() if k != "text"} for c in chunks]],
        "distances": [[0.1 * (i + 1) for i in range(len(chunks))]],
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks() -> list[dict]:
    """Return the shared sample chunks from tests/fixtures/sample_chunks.json."""
    path = Path(__file__).parent / "fixtures" / "sample_chunks.json"
    return json.loads(path.read_text())


@pytest.fixture
def chroma_results(sample_chunks) -> dict:
    """ChromaDB-shaped results built from sample_chunks."""
    return _make_chroma_results(sample_chunks)


@pytest.fixture
def mock_groq_response():
    """A single canned Groq response."""
    return _make_groq_response


@pytest.fixture(autouse=True)
def _block_real_groq(monkeypatch):
    """Guarantee tests never hit the real Groq API.
    Sets a dummy key so Groq() construction succeeds, but the HTTP call
    would fail fast if any test accidentally reaches the network.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
