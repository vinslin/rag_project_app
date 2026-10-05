"""LLM-based query classifier.

Determines which data source(s) should answer a user query:
  - "rag"          : specific content from uploaded contract documents (chunks)
  - "mcp"          : company contract standards from the MCP server
  - "both"         : validate a specific clause/section against standards
  - "full_doc"     : read and analyse the whole document (no MCP needed)
  - "full_doc_mcp" : compare the whole document against all MCP standards

Observability:
  Each classification decision is written to data/logs/classifications.jsonl
  so routing decisions can be audited independently of request logs.

Prompts:
  All prompts and version constants live in router/prompts.py.
  Bump CLASSIFIER_PROMPT_VERSION there when the classifier prompt changes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from core import config
from router.prompts import CLASSIFY_SYSTEM, CLASSIFIER_PROMPT_VERSION

load_dotenv()
logger = logging.getLogger(__name__)

_CLASSIFY_LOG = Path(__file__).parent.parent / "data" / "logs" / "classifications.jsonl"


def _log_classification(query: str, route: str, reason: str, latency_ms: float) -> None:
    """Append one classification decision to classifications.jsonl."""
    _CLASSIFY_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trace_id":                  str(uuid.uuid4()),
        "timestamp":                 datetime.now(timezone.utc).isoformat(),
        "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
        "query":                     query,
        "route":                     route,
        "reason":                    reason,
        "latency_ms":                round(latency_ms, 1),
    }
    with open(_CLASSIFY_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Route(str, Enum):
    RAG          = "rag"
    MCP          = "mcp"
    BOTH         = "both"
    FULL_DOC     = "full_doc"
    FULL_DOC_MCP = "full_doc_mcp"


@dataclass(frozen=True)
class Classification:
    route: Route
    reason: str
    raw_response: str


class QueryClassifier:
    """Classifies a user query into a routing decision using an LLM call."""

    def __init__(self, model: str | None = None, max_retries: int = 3):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model or config.GENERATION_MODEL
        self._max_retries = max_retries

    def classify(self, query: str) -> Classification:
        """Return a Classification for the given query.

        Falls back to Route.RAG if the LLM response cannot be parsed.
        Each decision is written to data/logs/classifications.jsonl.
        """
        prompt = f"Query: {query}"
        t_start = time.perf_counter()

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": CLASSIFY_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                result = self._parse(raw, query)
                _log_classification(
                    query=query,
                    route=str(result.route),
                    reason=result.reason,
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                )
                return result

            except Exception as exc:
                err = str(exc)
                retryable = any(c in err for c in ("429", "500", "502", "503", "504"))
                if not retryable or attempt == self._max_retries:
                    logger.error("Classifier LLM call failed: %s", exc)
                    break
                wait = 5.0 * (2 ** (attempt - 1))
                logger.warning(
                    "Classifier retry %d/%d in %.1fs", attempt, self._max_retries, wait
                )
                time.sleep(wait)

        fallback = Classification(
            route=Route.RAG,
            reason="Classifier unavailable — defaulting to RAG.",
            raw_response="",
        )
        _log_classification(
            query=query,
            route=str(fallback.route),
            reason=fallback.reason,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )
        return fallback

    @staticmethod
    def _parse(raw: str, query: str) -> Classification:
        text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(text)
            route_str = data.get("route", "rag").lower().strip()
            route = Route(route_str)
            reason = data.get("reason", "")
            logger.debug("Classified %r → %s (%s)", query[:60], route, reason)
            return Classification(route=route, reason=reason, raw_response=raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Classifier parse error (%s) for raw=%r — defaulting to RAG", exc, raw
            )
            return Classification(
                route=Route.RAG,
                reason="Parse error in classification — defaulted to RAG.",
                raw_response=raw,
            )
