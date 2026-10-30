"""Unit tests for router/classifier.py — _parse() is pure; classify() mocks Groq."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from router.classifier import QueryClassifier, Route, Classification


# ── _parse — pure JSON parsing ────────────────────────────────────────────────

@pytest.mark.parametrize("route_str,expected_route", [
    ("rag",          Route.RAG),
    ("mcp",          Route.MCP),
    ("both",         Route.BOTH),
    ("full_doc",     Route.FULL_DOC),
    ("full_doc_mcp", Route.FULL_DOC_MCP),
])
def test_parse_valid_routes(route_str, expected_route):
    raw = json.dumps({"route": route_str, "reason": "test reason"})
    result = QueryClassifier._parse(raw, "test query")
    assert result.route == expected_route
    assert result.reason == "test reason"


def test_parse_invalid_json_falls_back_to_rag():
    result = QueryClassifier._parse("not json at all", "test query")
    assert result.route == Route.RAG
    assert "Parse error" in result.reason


def test_parse_unknown_route_value_falls_back_to_rag():
    raw = json.dumps({"route": "unknown_route", "reason": "some reason"})
    result = QueryClassifier._parse(raw, "test query")
    assert result.route == Route.RAG


def test_parse_strips_code_fence():
    raw = "```json\n" + json.dumps({"route": "mcp", "reason": "standards"}) + "\n```"
    result = QueryClassifier._parse(raw, "test query")
    assert result.route == Route.MCP


def test_parse_missing_reason_defaults_to_empty_string():
    raw = json.dumps({"route": "rag"})
    result = QueryClassifier._parse(raw, "test query")
    assert result.route == Route.RAG
    assert result.reason == ""


# ── classify — mocked Groq client ────────────────────────────────────────────

def _mock_groq_response(route: str, reason: str = "test"):
    content = json.dumps({"route": route, "reason": reason})
    message = SimpleNamespace(content=content, tool_calls=None)
    choice  = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_classifier_with_mock(route: str):
    """Return a QueryClassifier whose internal Groq client is mocked."""
    classifier = QueryClassifier.__new__(QueryClassifier)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(route)
    classifier._client    = mock_client
    classifier._model     = "test-model"
    classifier._max_retries = 1
    return classifier, mock_client


def test_classify_returns_correct_route_for_rag():
    classifier, _ = _make_classifier_with_mock("rag")
    result = classifier.classify("What is the payment term?")
    assert result.route == Route.RAG
    assert isinstance(result, Classification)


def test_classify_returns_correct_route_for_full_doc_mcp():
    classifier, _ = _make_classifier_with_mock("full_doc_mcp")
    result = classifier.classify("Run a full compliance check")
    assert result.route == Route.FULL_DOC_MCP


def test_classify_falls_back_to_rag_on_api_error():
    classifier = QueryClassifier.__new__(QueryClassifier)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")
    classifier._client     = mock_client
    classifier._model      = "test-model"
    classifier._max_retries = 1
    result = classifier.classify("What are the clauses?")
    assert result.route == Route.RAG
    assert "unavailable" in result.reason.lower() or "defaulting" in result.reason.lower()


def test_classify_passes_trace_id_through():
    classifier, mock_client = _make_classifier_with_mock("rag")
    result = classifier.classify("Query with trace", trace_id="abc-123")
    assert isinstance(result, Classification)
    mock_client.chat.completions.create.assert_called_once()
