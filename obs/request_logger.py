"""Production observability logger.

Writes one structured JSON line per request to data/logs/requests.jsonl.
Each record is self-contained: trace_id, timestamp, prompt_version,
route, classifier decision, query, answer, per-span latency, retrieved
context IDs, token counts, and cost broken down by stage.

Search helper:
    from obs.request_logger import search_logs
    hits = search_logs("termination", field="answer")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"

# Groq model pricing — USD per million tokens
INPUT_PRICE_PER_M  = 0.10
OUTPUT_PRICE_PER_M = 0.50


def _calc_cost(input_tok: int, output_tok: int) -> float:
    return (input_tok * INPUT_PRICE_PER_M + output_tok * OUTPUT_PRICE_PER_M) / 1_000_000


def log_request(
    *,
    query: str,
    answer: str,
    route: str,
    prompt_version: str,
    retrieved_context_ids: list,
    spans_ms: dict,
    input_tokens: int,
    output_tokens: int,
    mcp_tools_called: list | None = None,
    classifier_decision: dict | None = None,
    out_of_scope: bool = False,
) -> str:
    """Append one structured log record to requests.jsonl.

    Returns the generated trace_id so callers can cross-reference.

    Args:
        query:                 Raw user question.
        answer:                Final answer text sent to the user.
        route:                 Pipeline path taken (rag/mcp/both/full_doc/
                               full_doc_mcp/guardrail).
        prompt_version:        Version string from generation/prompts.py.
        retrieved_context_ids: ChromaDB chunk IDs used for generation.
        spans_ms:              Per-stage wall-clock milliseconds:
                               {retrieval, reranking, generation}.
        input_tokens:          Total prompt tokens across all LLM calls.
        output_tokens:         Total completion tokens across all LLM calls.
        mcp_tools_called:      Tool names invoked via MCP, if any.
        classifier_decision:   {route, reason} from the query classifier.
        out_of_scope:          True when guardrails blocked the query.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    trace_id  = str(uuid.uuid4())
    gen_cost  = _calc_cost(input_tokens, output_tokens)

    record = {
        "trace_id":             trace_id,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "prompt_version":       prompt_version,
        "route":                route,
        "out_of_scope":         out_of_scope,
        "classifier_decision":  classifier_decision,
        "query":                query,
        "answer":               answer,
        "spans_ms":             spans_ms,
        "retrieved_context_ids": retrieved_context_ids,
        "mcp_tools_called":     mcp_tools_called or [],
        "tokens": {
            "input":  input_tokens,
            "output": output_tokens,
            "total":  input_tokens + output_tokens,
        },
        "cost_usd": {
            "retrieval":  0.0,
            "generation": round(gen_cost, 8),
            "tools":      0.0,
            "total":      round(gen_cost, 8),
        },
    }

    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return trace_id


def search_logs(keyword: str, field: str = "answer") -> list[dict]:
    """Return all log records where `field` contains `keyword` (case-insensitive).

    Example:
        hits = search_logs("Section 8.2", field="answer")
        hits = search_logs("termination", field="query")
    """
    if not LOG_FILE.exists():
        return []

    kw   = keyword.lower()
    hits = []

    with open(LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if kw in str(rec.get(field, "")).lower():
                    hits.append(rec)
            except json.JSONDecodeError:
                continue

    return hits


def get_all_logs() -> list[dict]:
    """Return every log record, oldest first."""
    if not LOG_FILE.exists():
        return []
    records = []
    with open(LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
