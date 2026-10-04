"""Router Agent — orchestrates RAG, MCP, full-doc, and combined answering.

Entry point: ``route(query, full_doc_texts=None, **rag_kwargs) -> RouteResult``

Routes:
    rag          → RAG chunk retrieval only
    mcp          → MCP standards server only (tool-calling loop)
    both         → RAG chunks + MCP standards → synthesis LLM call
    full_doc     → whole document text → LLM (no MCP)
    full_doc_mcp → whole document text + ALL MCP standards → compliance report

MCP tool schemas are discovered dynamically from the server (tools/list)
and cached in-process. Nothing is hardcoded here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from groq import Groq
from dotenv import load_dotenv

from core import config
from pipeline.rag import answer_question
from router.classifier import QueryClassifier, Route
from mcp.client import MCPHttpClient
from router.doc_loader import load_full_text, DocumentLoadError

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = config.GENERATION_MODEL

# Token threshold: documents larger than this are split section-by-section
# rather than sent in a single prompt.
_SINGLE_PASS_TOKEN_LIMIT = 60_000


# ── System prompts ────────────────────────────────────────────────────────

_MCP_LOOP_SYSTEM = (
    "You are a legal contract standards advisor with access to the company's "
    "contract standards database. Answer the user's question by calling the "
    "relevant tools first, then give a precise, well-structured answer "
    "based solely on what the tools return."
)

_CLAUSE_SYNTHESIS_SYSTEM = (
    "You are a legal contract compliance analyst. "
    "You have been provided with the company's contract standards and the actual "
    "content retrieved from the contract under review. "
    "Produce a clear, structured compliance report: "
    "(1) what is present and compliant, "
    "(2) what is missing or non-compliant, "
    "(3) a concise overall verdict."
)

_FULL_DOC_SYSTEM = (
    "You are a senior legal analyst. You have been given the complete text of a "
    "contract. Provide a thorough analysis covering: key parties, scope of services, "
    "payment terms, termination conditions, confidentiality obligations, liability "
    "provisions, governing law, and any notable or unusual clauses. "
    "Be concise but comprehensive."
)

_FULL_DOC_MCP_SYSTEM = (
    "You are a senior legal contract compliance officer. "
    "You have been given the complete text of a contract AND the company's full "
    "contract standards (active template, required sections, required clauses, "
    "and per-clause requirements). "
    "Produce a structured compliance report with these exact sections:\n\n"
    "## 1. Template & Version\n"
    "State which template version applies and its effective date.\n\n"
    "## 2. Document Structure\n"
    "List each required section. Mark ✅ present / ❌ missing / ⚠️ out of order.\n\n"
    "## 3. Required Clauses\n"
    "For each mandatory clause code, mark ✅ found / ❌ missing.\n\n"
    "## 4. Clause Requirement Details\n"
    "For each clause, list its sub-requirements and mark each ✅ met / ❌ not met, "
    "with a one-line explanation citing the contract text.\n\n"
    "## 5. Overall Verdict\n"
    "State COMPLIANT / PARTIALLY COMPLIANT / NON-COMPLIANT with a 2-3 sentence summary."
)

_SECTION_ANALYSIS_SYSTEM = (
    "You are a legal contract compliance analyst reviewing one section of a contract. "
    "Given the company standards for this section and the contract text, "
    "state what is compliant and what is missing or non-compliant. Be concise."
)

_SECTION_SYNTHESIS_SYSTEM = (
    "You are a legal contract compliance officer. You have received section-by-section "
    "compliance analysis of a contract against company standards. "
    "Synthesise these into a single, well-structured final compliance report with an "
    "overall verdict: COMPLIANT / PARTIALLY COMPLIANT / NON-COMPLIANT."
)


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    args: dict
    result_preview: str = ""


@dataclass
class RouteResult:
    """Unified response returned by the router for every query."""

    answer: str
    route: str          # "rag"|"mcp"|"both"|"full_doc"|"full_doc_mcp"
    route_reason: str
    reasoning: str
    confidence: str
    out_of_scope: bool
    sources: list[dict] = field(default_factory=list)
    mcp_tools_called: list[ToolCall] = field(default_factory=list)
    # Present only when RAG ran
    vector_chunks: list[dict] = field(default_factory=list)
    bm25_chunks: list[dict] = field(default_factory=list)
    mmr_chunks: list[dict] = field(default_factory=list)
    # Present only when full-doc routes ran
    doc_filenames: list[str] = field(default_factory=list)
    doc_token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route,
            "route_reason": self.route_reason,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "out_of_scope": self.out_of_scope,
            "sources": self.sources,
            "mcp_tools_called": [
                {"name": t.name, "args": t.args, "result_preview": t.result_preview}
                for t in self.mcp_tools_called
            ],
            "vector_chunks": self.vector_chunks,
            "bm25_chunks": self.bm25_chunks,
            "mmr_chunks": self.mmr_chunks,
            "doc_filenames": self.doc_filenames,
            "doc_token_estimate": self.doc_token_estimate,
        }


# ── LLM helper ────────────────────────────────────────────────────────────

def _get_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _llm_call(
    client: Groq,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_retries: int = 4,
    base_delay: float = 5.0,
):
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict[str, Any] = {"model": MODEL, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            err = str(exc)
            retryable = any(c in err for c in ("429", "500", "502", "503", "504"))
            if not retryable or attempt == max_retries:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning("LLM retry %d/%d in %.1fs — %s", attempt, max_retries, wait, exc)
            time.sleep(wait)
    raise RuntimeError("_llm_call: all retries exhausted")


# ── MCP tool-calling loop ─────────────────────────────────────────────────

def _run_mcp_loop(
    client: Groq,
    mcp: MCPHttpClient,
    query: str,
    groq_tools: list[dict],
    max_iters: int = 8,
) -> tuple[str, list[ToolCall]]:
    """Run a Groq tool-calling loop using server-discovered MCP tools.

    Returns (final_answer_text, tools_called_list).
    """
    messages: list[dict] = [
        {"role": "system", "content": _MCP_LOOP_SYSTEM},
        {"role": "user", "content": query},
    ]
    tools_called: list[ToolCall] = []

    for _ in range(max_iters):
        resp = _llm_call(client, messages, tools=groq_tools)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or "", tools_called

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            logger.info("[MCP] → %s(%s)", tc.function.name, args)
            result_text = mcp.dispatch(tc.function.name, args)
            logger.debug("[MCP] ← %s: %s", tc.function.name, result_text[:200])
            tools_called.append(ToolCall(
                name=tc.function.name,
                args=args,
                result_preview=result_text[:300],
            ))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    logger.warning("MCP loop reached max_iters without a final answer")
    return "[Max iterations reached without a final answer]", tools_called


def _fetch_all_mcp_standards(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
) -> tuple[str, list[ToolCall]]:
    """Fetch the complete MCP standards in a single tool-calling pass.

    Instructs the LLM to call all four MCP tools so we get the full picture
    (template, structure, required clauses, and every clause's requirements)
    before running the document compliance analysis.
    """
    query = (
        "Retrieve ALL company contract standards in full detail: "
        "the active template, the required document sections in order, "
        "all required clause codes, and the detailed requirements for "
        "EVERY required clause. Call all necessary tools."
    )
    return _run_mcp_loop(client, mcp, query, groq_tools, max_iters=12)


# ── Full-document analysis helpers ────────────────────────────────────────

def _analyse_single_pass(
    client: Groq,
    doc_text: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Send the entire document in one LLM call."""
    return _llm_call(
        client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    ).choices[0].message.content


def _analyse_section_by_section(
    client: Groq,
    doc_text: str,
    standards_text: str,
    mcp_sections: list[str],
) -> str:
    """Split the document by MCP section codes and analyse each independently.

    Used when the full document exceeds _SINGLE_PASS_TOKEN_LIMIT tokens.
    Each section is analysed against its relevant standards, then a final
    synthesis call produces the consolidated compliance report.
    """
    section_reports: list[str] = []

    for section_code in mcp_sections:
        section_prompt = (
            f"COMPANY STANDARDS FOR SECTION '{section_code}':\n"
            f"{standards_text}\n\n"
            f"CONTRACT TEXT (full — focus on the {section_code} section):\n"
            f"{doc_text}\n\n"
            f"Analyse only the {section_code} section for compliance."
        )
        report = _llm_call(
            client,
            messages=[
                {"role": "system", "content": _SECTION_ANALYSIS_SYSTEM},
                {"role": "user", "content": section_prompt},
            ],
        ).choices[0].message.content
        section_reports.append(f"### Section: {section_code}\n{report}")
        logger.debug("Section analysis done: %s", section_code)

    # Synthesise all section reports into one final compliance report
    synthesis_prompt = (
        "SECTION-BY-SECTION ANALYSIS:\n\n"
        + "\n\n---\n\n".join(section_reports)
        + "\n\nSynthesize the above into a single structured compliance report."
    )
    return _llm_call(
        client,
        messages=[
            {"role": "system", "content": _SECTION_SYNTHESIS_SYSTEM},
            {"role": "user", "content": synthesis_prompt},
        ],
    ).choices[0].message.content


# ── Route handlers ────────────────────────────────────────────────────────

def _handle_rag(query: str, route_reason: str, **rag_kwargs) -> RouteResult:
    logger.info("[Router] RAG path: %r", query[:80])
    result = answer_question(query, **rag_kwargs)
    return RouteResult(
        answer=result["answer"],
        route=Route.RAG,
        route_reason=route_reason,
        reasoning=result.get("reasoning", ""),
        confidence=result.get("confidence", "high"),
        out_of_scope=result.get("out_of_scope", False),
        sources=result.get("sources", []),
        mcp_tools_called=[],
        vector_chunks=result.get("vector_chunks", []),
        bm25_chunks=result.get("bm25_chunks", []),
        mmr_chunks=result.get("mmr_chunks", []),
    )


def _handle_mcp(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
    query: str,
    route_reason: str,
) -> RouteResult:
    logger.info("[Router] MCP path: %r", query[:80])
    answer, tools_called = _run_mcp_loop(client, mcp, query, groq_tools)
    return RouteResult(
        answer=answer,
        route=Route.MCP,
        route_reason=route_reason,
        reasoning=f"Answered from MCP standards server. Tools: {[t.name for t in tools_called]}",
        confidence="high",
        out_of_scope=False,
        mcp_tools_called=tools_called,
    )


def _handle_both(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
    query: str,
    route_reason: str,
    **rag_kwargs,
) -> RouteResult:
    logger.info("[Router] BOTH path: %r", query[:80])

    rag_result = answer_question(query, **rag_kwargs)

    if rag_result.get("out_of_scope"):
        logger.warning("[Router] Query blocked by guardrails — skipping MCP")
        return RouteResult(
            answer=rag_result["answer"],
            route=Route.BOTH,
            route_reason=route_reason,
            reasoning=rag_result.get("reasoning", ""),
            confidence=rag_result.get("confidence", "high"),
            out_of_scope=True,
        )

    rag_answer = rag_result.get("answer", "No relevant content found in the documents.")

    mcp_query = f"What do the company standards require regarding: {query}"
    mcp_answer, tools_called = _run_mcp_loop(client, mcp, mcp_query, groq_tools)

    synthesis = _llm_call(
        client,
        messages=[
            {"role": "system", "content": _CLAUSE_SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"COMPANY CONTRACT STANDARDS:\n{mcp_answer}\n\n"
                    f"ACTUAL CONTRACT CONTENT:\n{rag_answer}\n\n"
                    f"USER QUESTION:\n{query}"
                ),
            },
        ],
    )

    return RouteResult(
        answer=synthesis.choices[0].message.content,
        route=Route.BOTH,
        route_reason=route_reason,
        reasoning=(
            f"[Router: both] RAG retrieved clause context; "
            f"MCP tools: {[t.name for t in tools_called]}; synthesis LLM called."
        ),
        confidence="high",
        out_of_scope=False,
        sources=rag_result.get("sources", []),
        mcp_tools_called=tools_called,
        vector_chunks=rag_result.get("vector_chunks", []),
        bm25_chunks=rag_result.get("bm25_chunks", []),
        mmr_chunks=rag_result.get("mmr_chunks", []),
    )


def _handle_full_doc(
    client: Groq,
    query: str,
    route_reason: str,
    full_doc_texts: dict[str, str] | None,
) -> RouteResult:
    """Load the full document and answer without MCP."""
    logger.info("[Router] FULL_DOC path: %r", query[:80])

    try:
        doc = load_full_text(full_doc_texts)
    except DocumentLoadError as exc:
        return RouteResult(
            answer=str(exc),
            route=Route.FULL_DOC,
            route_reason=route_reason,
            reasoning="Document load failed.",
            confidence="low",
            out_of_scope=False,
        )

    logger.info(
        "[Router] Full doc loaded: %d tokens from %s", doc.estimated_tokens, doc.filenames
    )

    user_prompt = f"CONTRACT TEXT:\n\n{doc.text}\n\nUSER REQUEST:\n{query}"
    answer = _analyse_single_pass(client, doc.text, _FULL_DOC_SYSTEM, user_prompt)

    return RouteResult(
        answer=answer,
        route=Route.FULL_DOC,
        route_reason=route_reason,
        reasoning=(
            f"Full document analysis (~{doc.estimated_tokens} tokens, "
            f"source: {doc.source}). No MCP call needed."
        ),
        confidence="high",
        out_of_scope=False,
        doc_filenames=doc.filenames,
        doc_token_estimate=doc.estimated_tokens,
    )


def _handle_full_doc_mcp(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
    query: str,
    route_reason: str,
    full_doc_texts: dict[str, str] | None,
) -> RouteResult:
    """Load the full document AND fetch ALL MCP standards → compliance report."""
    logger.info("[Router] FULL_DOC_MCP path: %r", query[:80])

    # Step 1 — Load full document text
    try:
        doc = load_full_text(full_doc_texts)
    except DocumentLoadError as exc:
        return RouteResult(
            answer=str(exc),
            route=Route.FULL_DOC_MCP,
            route_reason=route_reason,
            reasoning="Document load failed.",
            confidence="low",
            out_of_scope=False,
        )

    logger.info(
        "[Router] Full doc loaded: %d tokens from %s", doc.estimated_tokens, doc.filenames
    )

    # Step 2 — Fetch ALL MCP standards in one tool-calling pass
    standards_text, tools_called = _fetch_all_mcp_standards(client, mcp, groq_tools)
    logger.info(
        "[Router] MCP standards fetched via %d tool call(s)", len(tools_called)
    )

    # Step 3 — Compliance analysis: single pass or section-by-section
    if doc.estimated_tokens <= _SINGLE_PASS_TOKEN_LIMIT:
        logger.info("[Router] Single-pass compliance analysis (%d tokens)", doc.estimated_tokens)
        user_prompt = (
            f"COMPANY CONTRACT STANDARDS (from MCP server):\n{standards_text}\n\n"
            f"CONTRACT TEXT (complete):\n{doc.text}\n\n"
            f"USER REQUEST:\n{query}"
        )
        answer = _analyse_single_pass(client, doc.text, _FULL_DOC_MCP_SYSTEM, user_prompt)
    else:
        logger.info(
            "[Router] Document too large (%d tokens > %d) — section-by-section analysis",
            doc.estimated_tokens,
            _SINGLE_PASS_TOKEN_LIMIT,
        )
        # Extract section codes from the standards text for splitting
        mcp_sections = [
            "PARTIES", "DEFINITIONS", "SCOPE", "PAYMENT",
            "TERMINATION", "CONFIDENTIALITY", "DATA_PROTECTION",
            "LIABILITY", "GOVERNING_LAW", "DISPUTE",
        ]
        answer = _analyse_section_by_section(
            client, doc.text, standards_text, mcp_sections
        )

    return RouteResult(
        answer=answer,
        route=Route.FULL_DOC_MCP,
        route_reason=route_reason,
        reasoning=(
            f"Full document compliance review (~{doc.estimated_tokens} tokens, "
            f"source: {doc.source}). "
            f"MCP tools called: {[t.name for t in tools_called]}."
        ),
        confidence="high",
        out_of_scope=False,
        mcp_tools_called=tools_called,
        doc_filenames=doc.filenames,
        doc_token_estimate=doc.estimated_tokens,
    )


# ── Module-level singletons ───────────────────────────────────────────────

_classifier = QueryClassifier()
_mcp_client = MCPHttpClient()
_groq_tools_cache: list[dict] | None = None


def _get_groq_tools() -> list[dict]:
    """Fetch and cache sanitized MCP Groq tools from the server (tools/list).

    Cached in-process — no repeated network call per query.
    Call invalidate_tools_cache() to force a refresh (e.g. after server restart).
    """
    global _groq_tools_cache
    if _groq_tools_cache is None:
        _groq_tools_cache = _mcp_client.to_groq_tools()
        logger.info(
            "[Router] Cached %d Groq tools: %s",
            len(_groq_tools_cache),
            [t["function"]["name"] for t in _groq_tools_cache],
        )
    return _groq_tools_cache


def invalidate_tools_cache() -> None:
    """Force the next query to re-fetch tool schemas from the MCP server."""
    global _groq_tools_cache
    _groq_tools_cache = None


# ── Public API ────────────────────────────────────────────────────────────

def route(
    query: str,
    full_doc_texts: dict[str, str] | None = None,
    **rag_kwargs,
) -> RouteResult:
    """Classify a query and dispatch to the correct data source(s).

    Args:
        query:          The user's natural-language question.
        full_doc_texts: Mapping of {filename: full_text} stored by app.py
                        at index-build time. Required for full_doc and
                        full_doc_mcp routes; falls back to ChromaDB if None.
        **rag_kwargs:   Forwarded to ``rag.pipeline.answer_question``
                        (search_mode, retrieval_k, mmr_k, mmr_lambda,
                        final_k, conversation_history, document_type).

    Returns:
        RouteResult with the answer and full provenance metadata.
    """
    classification = _classifier.classify(query)
    client = _get_client()
    r = classification.route

    # ── MCP-dependent routes: check server availability first ─────────────
    mcp_needed = r in (Route.MCP, Route.BOTH, Route.FULL_DOC_MCP)
    if mcp_needed and not _mcp_client.is_available():
        fallback_reason = (
            f"MCP server unavailable — fell back to "
            f"{'RAG' if r != Route.FULL_DOC_MCP else 'full-doc only'}."
        )
        logger.warning("[Router] %s", fallback_reason)
        if r == Route.FULL_DOC_MCP:
            return _handle_full_doc(client, query, fallback_reason, full_doc_texts)
        return _handle_rag(query, fallback_reason, **rag_kwargs)

    groq_tools = _get_groq_tools() if mcp_needed else []

    if r == Route.MCP:
        return _handle_mcp(client, _mcp_client, groq_tools, query, classification.reason)

    if r == Route.BOTH:
        return _handle_both(
            client, _mcp_client, groq_tools, query, classification.reason, **rag_kwargs
        )

    if r == Route.FULL_DOC:
        return _handle_full_doc(client, query, classification.reason, full_doc_texts)

    if r == Route.FULL_DOC_MCP:
        return _handle_full_doc_mcp(
            client, _mcp_client, groq_tools, query, classification.reason, full_doc_texts
        )

    # Default: RAG
    return _handle_rag(query, classification.reason, **rag_kwargs)
