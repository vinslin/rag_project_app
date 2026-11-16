"""Unit tests for retrieval/mmr.py — cosine similarity + MMR selection logic."""

import pytest
from unittest.mock import patch
from retrieval.mmr import _cosine_similarity, mmr_rerank


# ── _cosine_similarity ────────────────────────────────────────────────────────

def test_identical_vectors_similarity_is_one():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_orthogonal_vectors_similarity_is_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_opposite_vectors_similarity_is_minus_one():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6


def test_zero_vector_returns_zero():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_similarity(a, b) == 0.0


# ── mmr_rerank ────────────────────────────────────────────────────────────────

def _make_results(texts: list[str]) -> dict:
    """Build a minimal ChromaDB result dict for mmr_rerank."""
    return {
        "documents": [texts],
        "metadatas": [[{"source": f"doc_{i}"} for i in range(len(texts))]],
        "distances": [[0.1 * (i + 1) for i in range(len(texts))]],
    }


def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding: each unique text maps to a distinct direction."""
    _vocab = {t: i for i, t in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"])}
    dim = 5
    vec = [0.0] * dim
    for word in text.lower().split():
        if word in _vocab:
            vec[_vocab[word]] = 1.0
    norm = sum(x ** 2 for x in vec) ** 0.5
    return [x / norm if norm else 1.0 / dim for x in vec]


@patch("retrieval.mmr.create_embedding", side_effect=_fake_embedding)
def test_mmr_returns_k_results(mock_embed):
    texts = ["alpha text", "beta text", "gamma text", "delta text", "epsilon text"]
    results = _make_results(texts)
    out = mmr_rerank("alpha", results, mmr_k=3)
    assert len(out["documents"][0]) == 3
    assert len(out["metadatas"][0]) == 3
    assert len(out["distances"][0]) == 3


@patch("retrieval.mmr.create_embedding", side_effect=_fake_embedding)
def test_mmr_when_fewer_docs_than_k_returns_all(mock_embed):
    texts = ["alpha text", "beta text"]
    results = _make_results(texts)
    out = mmr_rerank("alpha", results, mmr_k=10)
    assert len(out["documents"][0]) == 2


@patch("retrieval.mmr.create_embedding", side_effect=_fake_embedding)
def test_mmr_empty_results_passthrough(mock_embed):
    results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    out = mmr_rerank("query", results, mmr_k=5)
    assert out["documents"] == [[]]


@patch("retrieval.mmr.create_embedding", side_effect=_fake_embedding)
def test_mmr_scores_present_in_output(mock_embed):
    texts = ["alpha text", "beta text", "gamma text"]
    results = _make_results(texts)
    out = mmr_rerank("alpha", results, mmr_k=2)
    assert "mmr_scores" in out
    assert len(out["mmr_scores"]) == 2


@patch("retrieval.mmr.create_embedding", side_effect=_fake_embedding)
def test_mmr_preserves_rrf_scores_for_selected(mock_embed):
    texts = ["alpha text", "beta text", "gamma text"]
    results = _make_results(texts)
    results["rrf_scores"] = [0.9, 0.5, 0.3]
    out = mmr_rerank("alpha", results, mmr_k=2)
    assert "rrf_scores" in out
    assert len(out["rrf_scores"]) == 2
