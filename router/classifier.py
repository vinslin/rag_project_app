"""LLM-based query classifier.

Determines which data source(s) should answer a user query:
  - "rag"          : specific content from uploaded contract documents (chunks)
  - "mcp"          : company contract standards from the MCP server
  - "both"         : validate a specific clause/section against standards
  - "full_doc"     : read and analyse the whole document (no MCP needed)
  - "full_doc_mcp" : compare the whole document against all MCP standards
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

from core import config

load_dotenv()
logger = logging.getLogger(__name__)


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


_CLASSIFY_SYSTEM = """\
You are a query classifier for a legal contract analysis system with these data sources:

1. RAG (chunk search): Retrieves specific passages from uploaded contract PDFs.
2. MCP (standards server): Holds company-approved contract standards — active template,
   required sections, mandatory clauses, and per-clause requirements.
3. FULL_DOC: Loads the entire contract document text (not just chunks).

Classify into exactly ONE route:

- "rag"
  The user asks about specific content, a named clause, a date, a party name,
  or any detail that can be answered from a few paragraphs of the document.
  Examples: "What is the payment term?", "Who are the parties?",
            "What does clause 8.2 say?"

- "mcp"
  The user asks about company standards, what the active template requires,
  which clauses are mandatory, or what a clause must contain per company policy.
  Examples: "What clauses are required?", "What is the active template version?",
            "What must the confidentiality clause include?"

- "both"
  The user asks whether a specific clause or section in their contract meets
  company standards — a targeted clause-level compliance check.
  Examples: "Is our termination clause compliant?",
            "Does section 6 meet the confidentiality requirements?"

- "full_doc"
  The user wants a broad analysis, summary, or overview of the whole document
  without comparing it against company standards.
  Examples: "Summarise this contract", "Give me an overview of the agreement",
            "What are the key points of this contract?",
            "Explain the entire document to me"

- "full_doc_mcp"
  The user wants to compare or validate the ENTIRE contract against the company
  template or all company standards — a full document-level compliance review.
  Examples: "Compare this contract with the active template",
            "Does this contract meet all company requirements?",
            "Full compliance review of this document",
            "Validate the entire contract against company standards",
            "Check all clauses against the template",
            "Is this contract compliant with our template?"

Rules:
- Specific question about one clause → "both" (not "full_doc_mcp")
- Whole document summary with NO mention of standards → "full_doc"
- Any comparison with template / company standards across the whole document → "full_doc_mcp"
- When uncertain between "both" and "full_doc_mcp", pick "full_doc_mcp"

Respond with ONLY a valid JSON object — no markdown, no explanation:
{"route": "rag"|"mcp"|"both"|"full_doc"|"full_doc_mcp", "reason": "<one concise sentence>"}
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
                logger.warning(
                    "Classifier retry %d/%d in %.1fs", attempt, self._max_retries, wait
                )
                time.sleep(wait)

        return Classification(
            route=Route.RAG,
            reason="Classifier unavailable — defaulting to RAG.",
            raw_response="",
        )

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
