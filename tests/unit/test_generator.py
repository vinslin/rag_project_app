"""Unit tests for generation/generator.py — mocks Groq + ChromaDB."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from generation.generator import _detect_counterparty, _build_context


# ── _detect_counterparty — pure dict lookup ───────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    ("What did Northwind agree to?",            "Northwind Logistics Private Limited"),
    ("Show Vertex payment terms",               "Vertex Retail Solutions Limited"),
    ("Halcyon confidentiality clause",          "Halcyon Analytics GmbH"),
    ("What is the payment term?",               None),   # no counterparty
    ("Compare Northwind and Vertex clauses",    None),   # two counterparties → None
])
def test_detect_counterparty(query, expected):
    assert _detect_counterparty(query) == expected


def test_detect_counterparty_case_insensitive():
    assert _detect_counterparty("NORTHWIND payment") == "Northwind Logistics Private Limited"


# ── _build_context — sorts by effective_date descending ──────────────────────

def _make_chroma_results(chunks: list[dict]) -> dict:
    return {
        "documents": [[c["text"] for c in chunks]],
        "metadatas": [[{k: v for k, v in c.items() if k != "text"} for c in chunks]],
        "distances": [[0.1 * (i + 1) for i in range(len(chunks))]],
    }


def test_build_context_sorts_amendment_before_msa():
    chunks = [
        {"text": "MSA clause", "source": "MSA-001.pdf", "source_doc": "MSA-001",
         "counterparty": "Acme", "effective_date": "2026-01-01",
         "doc_type": "msa", "clause_ref": "Section 1", "page": 1, "heading": "Payment"},
        {"text": "Amendment overrides payment", "source": "AMD-001.pdf", "source_doc": "AMD-001",
         "counterparty": "Acme", "effective_date": "2026-06-01",
         "doc_type": "amendment", "clause_ref": "Section 1", "page": 1, "heading": "Payment"},
    ]
    context = _build_context(_make_chroma_results(chunks))
    # Amendment (2026-06-01) must appear before MSA (2026-01-01) in context string
    assert context.index("AMD-001") < context.index("MSA-001")


def test_build_context_includes_all_metadata_fields():
    chunks = [
        {"text": "Clause text here", "source": "MSA-001.pdf", "source_doc": "MSA-001",
         "counterparty": "Test Corp", "effective_date": "2026-01-01",
         "doc_type": "msa", "clause_ref": "Section 5", "page": 3, "heading": "Payment"},
    ]
    context = _build_context(_make_chroma_results(chunks))
    assert "MSA-001"   in context
    assert "Test Corp" in context
    assert "Section 5" in context
    assert "2026-01-01" in context
    assert "Clause text here" in context


def test_build_context_separator_between_chunks():
    chunks = [
        {"text": "First clause", "source": "A.pdf", "source_doc": "A",
         "counterparty": "X", "effective_date": "2026-01-01",
         "doc_type": "msa", "clause_ref": "Section 1", "page": 1, "heading": "S1"},
        {"text": "Second clause", "source": "B.pdf", "source_doc": "B",
         "counterparty": "X", "effective_date": "2026-02-01",
         "doc_type": "msa", "clause_ref": "Section 2", "page": 2, "heading": "S2"},
    ]
    context = _build_context(_make_chroma_results(chunks))
    assert "---" in context


# ── GeminiGenerator.generate — mocked retrieval + LLM ────────────────────────

def _groq_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice  = SimpleNamespace(message=message)
    usage   = SimpleNamespace(prompt_tokens=200, completion_tokens=80)
    return SimpleNamespace(choices=[choice], usage=usage)


def _chroma_results_for(texts: list[str], dates: list[str] | None = None) -> dict:
    dates = dates or ["2026-01-01"] * len(texts)
    return {
        "ids":       [["id_" + str(i) for i in range(len(texts))]],
        "documents": [texts],
        "metadatas": [[{
            "source": f"doc_{i}.pdf", "page": 1, "heading": "Section",
            "source_doc": f"DOC-{i}", "counterparty": "Acme",
            "effective_date": dates[i], "doc_type": "msa", "clause_ref": "Section 1",
        } for i in range(len(texts))]],
        "distances": [[0.1 * (i + 1) for i in range(len(texts))]],
    }


@patch("generation.generator.groq_call_with_retry")
@patch("generation.generator.get_client")
def test_generate_no_results_returns_not_found(mock_get_client, mock_llm):
    from generation.generator import GeminiGenerator
    gen = GeminiGenerator()

    empty = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    with patch("generation.generator.retrieve", return_value=empty), \
         patch("generation.generator.get_collection", return_value=MagicMock()):
        result = gen.generate("What is the deadline?", search_mode="vector")

    assert "could not find" in result["answer"].lower()
    assert result["confidence"] == "low"
    mock_llm.assert_not_called()


@patch("generation.generator.groq_call_with_retry")
@patch("generation.generator.get_client")
def test_generate_high_distance_returns_out_of_scope(mock_get_client, mock_llm):
    from generation.generator import GeminiGenerator
    gen = GeminiGenerator()

    far_results = {
        "ids":       [["id_0"]],
        "documents": [["Irrelevant text"]],
        "metadatas": [[{"source": "x.pdf", "page": 1, "heading": "", "source_doc": "",
                        "counterparty": "", "effective_date": "", "doc_type": "", "clause_ref": ""}]],
        "distances": [[2.0]],   # > 1.5 threshold
    }

    with patch("generation.generator.retrieve", return_value=far_results), \
         patch("generation.generator.get_collection", return_value=MagicMock()):
        result = gen.generate("Random unrelated query", search_mode="vector")

    assert result["out_of_scope"] is True
    assert result["confidence"] == "low"
    mock_llm.assert_not_called()


@patch("reranking.reranker.rerank")
@patch("retrieval.mmr.mmr_rerank")
@patch("generation.generator.groq_call_with_retry")
@patch("generation.generator.get_client")
def test_generate_returns_answer_and_sources(
    mock_get_client, mock_llm, mock_mmr, mock_rerank
):
    from generation.generator import GeminiGenerator
    gen = GeminiGenerator()

    texts = ["Payment due in 30 days."]
    chroma = _chroma_results_for(texts)
    mock_mmr.return_value    = chroma
    mock_rerank.return_value = chroma
    mock_llm.return_value    = _groq_response("Payment is due in 30 days per Section 5.")

    with patch("generation.generator.retrieve", return_value=chroma), \
         patch("generation.generator.get_collection", return_value=MagicMock()):
        result = gen.generate("When is payment due?", search_mode="vector")

    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "sources" in result
    assert result["confidence"] == "high"
    assert result["out_of_scope"] is False
    assert "_spans_ms" in result
    assert "_input_tokens" in result
