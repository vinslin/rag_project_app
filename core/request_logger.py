"""Production observability logger.

Writes one structured JSON line per request to data/logs/requests.jsonl.
Each record is self-contained: trace_id, timestamp, prompt_version,
route, classifier decision, query, answer, per-span latency, retrieved
context IDs, token counts (generation + tools), and generation cost.

Search helpers:
    from core.request_logger import search_logs, iter_logs
    hits = search_logs("termination", field="answer", limit=20)
    for rec in iter_logs():          # O(1) memory — one record at a time
        process(rec)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

LOG_DIR  = Path(__file__).parent.parent / "data" / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"

# Create once at import time — not on every log call
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Groq model pricing — USD per million tokens (generation only)
INPUT_PRICE_PER_M  = 0.10
OUTPUT_PRICE_PER_M = 0.50

# JSONL rotation constants — rename-based, single-process safe
_MAX_JSONL_BYTES    = 50 * 1024 * 1024  # 50 MB per file
_JSONL_BACKUP_COUNT = 3


def _calc_cost(input_tok: int, output_tok: int) -> float:
    return (input_tok * INPUT_PRICE_PER_M + output_tok * OUTPUT_PRICE_PER_M) / 1_000_000


def _rotate_if_needed(path: Path) -> None:
    """Rename-based JSONL rotation: path → path.1 → path.2 … path.N."""
    if not path.exists() or path.stat().st_size < _MAX_JSONL_BYTES:
        return
    for i in range(_JSONL_BACKUP_COUNT - 1, 0, -1):
        src = path.parent / f"{path.name}.{i}"
        dst = path.parent / f"{path.name}.{i + 1}"
        if src.exists():
            src.rename(dst)
    path.rename(path.parent / f"{path.name}.1")


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
    tools_input_tokens: int = 0,
    tools_output_tokens: int = 0,
    mcp_tools_called: list | None = None,
    classifier_decision: dict | None = None,
    out_of_scope: bool = False,
    trace_id: str | None = None,
) -> str:
    """Append one structured log record to requests.jsonl.

    Returns the trace_id (passed in or auto-generated) for cross-reference.
    ``tools_input_tokens`` / ``tools_output_tokens`` count tokens consumed by
    the MCP tool-calling loop separately from generation tokens.
    """
    _rotate_if_needed(LOG_FILE)

    trace_id = trace_id or str(uuid.uuid4())
    gen_cost = _calc_cost(input_tokens, output_tokens)

    record = {
        "trace_id":              trace_id,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "prompt_version":        prompt_version,
        "route":                 route,
        "out_of_scope":          out_of_scope,
        "classifier_decision":   classifier_decision,
        "query":                 query,
        "answer":                answer,
        "spans_ms":              spans_ms,
        "retrieved_context_ids": retrieved_context_ids,
        "mcp_tools_called":      mcp_tools_called or [],
        "tokens": {
            "input":  input_tokens,
            "output": output_tokens,
            "total":  input_tokens + output_tokens,
            # MCP tool-calling loop tokens tracked separately from generation
            "tools": {
                "input":  tools_input_tokens,
                "output": tools_output_tokens,
            },
        },
        "cost_usd": {
            # retrieval uses local ChromaDB — no per-query API cost
            "generation": round(gen_cost, 8),
            "total":      round(gen_cost, 8),
        },
    }

    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return trace_id


def iter_logs() -> Iterator[dict]:
    """Yield every log record one at a time — O(1) memory."""
    if not LOG_FILE.exists():
        return
    with open(LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def search_logs(
    keyword: str,
    field: str = "answer",
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """Return records where ``field`` contains ``keyword`` (case-insensitive).

    Args:
        limit:  Maximum number of records to return. ``None`` means no cap.
        offset: Skip this many matching records before collecting results.
    """
    kw      = keyword.lower()
    hits: list[dict] = []
    skipped = 0

    for rec in iter_logs():
        if kw not in str(rec.get(field, "")).lower():
            continue
        if skipped < offset:
            skipped += 1
            continue
        hits.append(rec)
        if limit is not None and len(hits) >= limit:
            break

    return hits


def get_all_logs(limit: int | None = None, offset: int = 0) -> list[dict]:
    """Return log records oldest first.

    Args:
        limit:  Maximum number of records to return. ``None`` means no cap.
        offset: Skip this many records from the start of the file.
    """
    records: list[dict] = []
    count = 0

    for rec in iter_logs():
        count += 1
        if count <= offset:
            continue
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break

    return records
