"""LLM-based query classifier.

Determines which data source(s) should answer a user query:
  - "rag"  : content from uploaded contract documents
  - "mcp"  : company contract standards from the MCP server
  - "both" : compliance check — compare documents against standards
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum

from groq import Groq
from dotenv import load_dotenv

from rag import config

load_dotenv()
logger = logging.getLogger(__name__)


class Route(str, Enum):
    RAG = "rag"
    MCP = "mcp"
    BOTH = "both"


@dataclass(frozen=True)
class Classification:
    route: Route
    reason: str
    raw_response: str


_CLASSIFY_SYSTEM = """\
You are a query classifier for a legal contract analysis system that has two data sources:

1. RAG (document search): Searches through uploaded contract PDF documents to find specific content,
   clauses, terms, dates, parties, and language written in those documents.

2. MCP (standards server): A server that holds the company's approved contract standards —
   the active template version, required document sections, mandatory clauses, and
   detailed requirements for each clause.

Classify the user query into exactly ONE of these routes:
- "rag"  : The query asks about specific content, language, or details found inside uploaded contracts
- "mcp"  : The query asks about company standards, required clauses, active template, or compliance rules
- "both" : The query asks to validate or compare an uploaded contract against company standards

Rules:
- If the user asks "what does the contract say about X" → rag
- If the user asks "what clauses are required" or "what must a contract include" → mcp
- If the user asks "is my contract compliant" or "does it have all required clauses" → both
- When uncertain, prefer "both" over "rag" alone

Respond with ONLY a valid JSON object (no markdown, no explanation):
{"route": "rag" | "mcp" | "both", "reason": "<one concise sentence>"}
"""


class QueryClassifier:
    """Classifies a user query into a routing decision using an LLM call."""

    def __init__(self, model: str | None = None, max_retries: int = 3):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model or config.GENERATION_MODEL
        self._max_retries = max_retries

    def classify(self, query: str) -> Classification:
        """Return a Classification for the given query.

        Falls back to Route.RAG if the LLM response cannot be parsed.
        """
        prompt = f"Query: {query}"

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _CLASSIFY_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                return self._parse(raw, query)

            except Exception as exc:
                err = str(exc)
                retryable = any(c in err for c in ("429", "500", "502", "503", "504"))
                if not retryable or attempt == self._max_retries:
                    logger.error("Classifier LLM call failed: %s", exc)
                    break
                wait = 5.0 * (2 ** (attempt - 1))
                logger.warning("Classifier retry %d/%d in %.1fs", attempt, self._max_retries, wait)
                time.sleep(wait)

        # Safe fallback
        return Classification(
            route=Route.RAG,
            reason="Classifier unavailable — defaulting to RAG.",
            raw_response="",
        )

    @staticmethod
    def _parse(raw: str, query: str) -> Classification:
        # Strip markdown code fences if the model added them
        text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(text)
            route_str = data.get("route", "rag").lower().strip()
            route = Route(route_str)
            reason = data.get("reason", "")
            logger.debug("Classified %r → %s (%s)", query[:60], route, reason)
            return Classification(route=route, reason=reason, raw_response=raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Classifier parse error (%s) for raw=%r — defaulting to RAG", exc, raw)
            return Classification(
                route=Route.RAG,
                reason="Parse error in classification — defaulted to RAG.",
                raw_response=raw,
            )
