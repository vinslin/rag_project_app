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

Observability:
    Every route call is logged to data/logs/requests.jsonl via obs.request_logger.
    Each log record contains the route taken, classifier decision, per-span latency,
    token counts across ALL LLM calls in the route, and the final answer text.
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
from generation.prompts import PROMPT_VERSION
from pipeline.rag import answer_question
from router.classifier import QueryClassifier, Route
from router.prompts import (
    AGENT_PROMPT_VERSION,
    CLASSIFIER_PROMPT_VERSION,
    MCP_LOOP_SYSTEM,
    CLAUSE_SYNTHESIS_SYSTEM,
    FULL_DOC_SYSTEM,
    FULL_DOC_MCP_SYSTEM,
    SECTION_ANALYSIS_SYSTEM,
    SECTION_SYNTHESIS_SYSTEM,
)
from mcp.client import MCPHttpClient
from router.doc_loader import load_full_text, DocumentLoadError
from obs.request_logger import log_request

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = config.GENERATION_MODEL

_SINGLE_PASS_TOKEN_LIMIT = 60_000


# ── Token accumulator ─────────────────────────────────────────────────────

class _UsageAccum:
    """Accumulates input/output token counts across multiple LLM calls."""

    __slots__ = ("input_tokens", "output_tokens")

    def __init__(self) -> None:
        self.input_tokens  = 0
        self.output_tokens = 0

    def add(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens  += getattr(usage, "prompt_tokens",     0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0


# ── System prompts (imported from router/prompts.py) ─────────────────────
# To change any prompt, edit router/prompts.py and bump AGENT_PROMPT_VERSION.


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
    vector_chunks: list[dict] = field(default_factory=list)
    bm25_chunks: list[dict] = field(default_factory=list)
    mmr_chunks: list[dict] = field(default_factory=list)
    doc_filenames: list[str] = field(default_factory=list)
    doc_token_estimate: int = 0
    # Observability fields — populated by each handler
    input_tokens: int = 0
    output_tokens: int = 0
    spans_ms: dict = field(default_factory=dict)
    retrieved_context_ids: list = field(default_factory=list)

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
    _acc: _UsageAccum | None = None,
):
    """Call the Groq LLM with retry. Accumulates token usage into _acc if provided."""
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict[str, Any] = {"model": MODEL, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
            if _acc is not None:
                _acc.add(response)
            return response
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
    _acc: _UsageAccum | None = None,
) -> tuple[str, list[ToolCall]]:
    """Run a Groq tool-calling loop using server-discovered MCP tools."""
    messages: list[dict] = [
        {"role": "system", "content": MCP_LOOP_SYSTEM},
        {"role": "user", "content": query},
    ]
    tools_called: list[ToolCall] = []

    for _ in range(max_iters):
        resp = _llm_call(client, messages, tools=groq_tools, _acc=_acc)
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
    _acc: _UsageAccum | None = None,
) -> tuple[str, list[ToolCall]]:
    """Fetch the complete MCP standards in a single tool-calling pass."""
    query = (
        "Retrieve ALL company contract standards in full detail: "
        "the active template, the required document sections in order, "
        "all required clause codes, and the detailed requirements for "
        "EVERY required clause. Call all necessary tools."
    )
    return _run_mcp_loop(client, mcp, query, groq_tools, max_iters=12, _acc=_acc)


# ── Full-document analysis helpers ────────────────────────────────────────

def _analyse_single_pass(
    client: Groq,
    doc_text: str,
    system_prompt: str,
    user_prompt: str,
    _acc: _UsageAccum | None = None,
) -> str:
    """Send the entire document in one LLM call."""
    return _llm_call(
        client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        _acc=_acc,
    ).choices[0].message.content


def _analyse_section_by_section(
    client: Groq,
    doc_text: str,
    standards_text: str,
    mcp_sections: list[str],
    _acc: _UsageAccum | None = None,
) -> str:
    """Split the document by MCP section codes and analyse each independently."""
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
                {"role": "system", "content": SECTION_ANALYSIS_SYSTEM},
                {"role": "user", "content": section_prompt},
            ],
            _acc=_acc,
        ).choices[0].message.content
        section_reports.append(f"### Section: {section_code}\n{report}")
        logger.debug("Section analysis done: %s", section_code)

    synthesis_prompt = (
        "SECTION-BY-SECTION ANALYSIS:\n\n"
        + "\n\n---\n\n".join(section_reports)
        + "\n\nSynthesize the above into a single structured compliance report."
    )
    return _llm_call(
        client,
        messages=[
            {"role": "system", "content": SECTION_SYNTHESIS_SYSTEM},
            {"role": "user", "content": synthesis_prompt},
        ],
        _acc=_acc,
    ).choices[0].message.content


# ── Route handlers ────────────────────────────────────────────────────────

def _handle_rag(
    query: str,
    route_reason: str,
    _acc: _UsageAccum,
    **rag_kwargs,
) -> RouteResult:
    logger.info("[Router] RAG path: %r", query[:80])
    # log=False: the router logs this request with full route context instead
    result = answer_question(query, log=False, **rag_kwargs)

    # Carry observability data from the generator into the accumulator
    _acc.input_tokens  += result.get("_input_tokens",  0)
    _acc.output_tokens += result.get("_output_tokens", 0)

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
        spans_ms=result.get("_spans_ms", {}),
        retrieved_context_ids=result.get("_retrieved_context_ids", []),
    )


def _handle_mcp(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
    query: str,
    route_reason: str,
    _acc: _UsageAccum,
) -> RouteResult:
    logger.info("[Router] MCP path: %r", query[:80])
    t0 = time.perf_counter()
    answer, tools_called = _run_mcp_loop(client, mcp, query, groq_tools, _acc=_acc)
    generation_ms = (time.perf_counter() - t0) * 1000

    return RouteResult(
        answer=answer,
        route=Route.MCP,
        route_reason=route_reason,
        reasoning=f"Answered from MCP standards server. Tools: {[t.name for t in tools_called]}",
        confidence="high",
        out_of_scope=False,
        mcp_tools_called=tools_called,
        spans_ms={"retrieval": 0.0, "reranking": 0.0, "generation": round(generation_ms, 1)},
    )


def _handle_both(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
    query: str,
    route_reason: str,
    _acc: _UsageAccum,
    **rag_kwargs,
) -> RouteResult:
    logger.info("[Router] BOTH path: %r", query[:80])
    t0 = time.perf_counter()

    rag_result = answer_question(query, log=False, **rag_kwargs)
    _acc.input_tokens  += rag_result.get("_input_tokens",  0)
    _acc.output_tokens += rag_result.get("_output_tokens", 0)

    if rag_result.get("out_of_scope"):
        logger.warning("[Router] Query blocked by guardrails — skipping MCP")
        return RouteResult(
            answer=rag_result["answer"],
            route=Route.BOTH,
            route_reason=route_reason,
            reasoning=rag_result.get("reasoning", ""),
            confidence=rag_result.get("confidence", "high"),
            out_of_scope=True,
            spans_ms=rag_result.get("_spans_ms", {}),
        )

    rag_answer = rag_result.get("answer", "No relevant content found in the documents.")

    mcp_query = f"What do the company standards require regarding: {query}"
    mcp_answer, tools_called = _run_mcp_loop(client, mcp, mcp_query, groq_tools, _acc=_acc)

    synthesis = _llm_call(
        client,
        messages=[
            {"role": "system", "content": CLAUSE_SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"COMPANY CONTRACT STANDARDS:\n{mcp_answer}\n\n"
                    f"ACTUAL CONTRACT CONTENT:\n{rag_answer}\n\n"
                    f"USER QUESTION:\n{query}"
                ),
            },
        ],
        _acc=_acc,
    )

    total_ms = (time.perf_counter() - t0) * 1000

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
        retrieved_context_ids=rag_result.get("_retrieved_context_ids", []),
        spans_ms={"retrieval": 0.0, "reranking": 0.0, "generation": round(total_ms, 1)},
    )


def _handle_full_doc(
    client: Groq,
    query: str,
    route_reason: str,
    full_doc_texts: dict[str, str] | None,
    _acc: _UsageAccum,
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

    logger.info("[Router] Full doc loaded: %d tokens from %s", doc.estimated_tokens, doc.filenames)

    t0 = time.perf_counter()
    user_prompt = f"CONTRACT TEXT:\n\n{doc.text}\n\nUSER REQUEST:\n{query}"
    answer = _analyse_single_pass(client, doc.text, FULL_DOC_SYSTEM, user_prompt, _acc=_acc)
    generation_ms = (time.perf_counter() - t0) * 1000

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
        spans_ms={"retrieval": 0.0, "reranking": 0.0, "generation": round(generation_ms, 1)},
    )


def _handle_full_doc_mcp(
    client: Groq,
    mcp: MCPHttpClient,
    groq_tools: list[dict],
    query: str,
    route_reason: str,
    full_doc_texts: dict[str, str] | None,
    _acc: _UsageAccum,
) -> RouteResult:
    """Load the full document AND fetch ALL MCP standards → compliance report."""
    logger.info("[Router] FULL_DOC_MCP path: %r", query[:80])

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

    logger.info("[Router] Full doc loaded: %d tokens from %s", doc.estimated_tokens, doc.filenames)

    t0 = time.perf_counter()

    standards_text, tools_called = _fetch_all_mcp_standards(client, mcp, groq_tools, _acc=_acc)
    logger.info("[Router] MCP standards fetched via %d tool call(s)", len(tools_called))

    if doc.estimated_tokens <= _SINGLE_PASS_TOKEN_LIMIT:
        logger.info("[Router] Single-pass compliance analysis (%d tokens)", doc.estimated_tokens)
        user_prompt = (
            f"COMPANY CONTRACT STANDARDS (from MCP server):\n{standards_text}\n\n"
            f"CONTRACT TEXT (complete):\n{doc.text}\n\n"
            f"USER REQUEST:\n{query}"
        )
        answer = _analyse_single_pass(client, doc.text, FULL_DOC_MCP_SYSTEM, user_prompt, _acc=_acc)
    else:
        logger.info(
            "[Router] Document too large (%d tokens > %d) — section-by-section analysis",
            doc.estimated_tokens, _SINGLE_PASS_TOKEN_LIMIT,
        )
        mcp_sections = [
            "PARTIES", "DEFINITIONS", "SCOPE", "PAYMENT",
            "TERMINATION", "CONFIDENTIALITY", "DATA_PROTECTION",
            "LIABILITY", "GOVERNING_LAW", "DISPUTE",
        ]
        answer = _analyse_section_by_section(
            client, doc.text, standards_text, mcp_sections, _acc=_acc
        )

    generation_ms = (time.perf_counter() - t0) * 1000

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
        spans_ms={"retrieval": 0.0, "reranking": 0.0, "generation": round(generation_ms, 1)},
    )


# ── Module-level singletons ───────────────────────────────────────────────

_classifier = QueryClassifier()
_mcp_client = MCPHttpClient()
_groq_tools_cache: list[dict] | None = None


def _get_groq_tools() -> list[dict]:
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
    """Classify a query, dispatch to the correct data source(s), and log.

    Every call writes one structured record to data/logs/requests.jsonl.
    Token counts span ALL LLM calls made by the route handler (classifier,
    retrieval LLM, synthesis LLM, MCP tool-calling loop, etc.).
    """
    t_route_start = time.perf_counter()
    _acc = _UsageAccum()

    # ── Classify ──────────────────────────────────────────────────────────
    classification = _classifier.classify(query)
    client = _get_client()
    r = classification.route

    classifier_decision = {
        "route":                     classification.route,
        "reason":                    classification.reason,
        "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
        "agent_prompt_version":      AGENT_PROMPT_VERSION,
    }

    # ── MCP-dependent routes: check server availability first ─────────────
    mcp_needed = r in (Route.MCP, Route.BOTH, Route.FULL_DOC_MCP)
    if mcp_needed and not _mcp_client.is_available():
        fallback_reason = (
            f"MCP server unavailable — fell back to "
            f"{'RAG' if r != Route.FULL_DOC_MCP else 'full-doc only'}."
        )
        logger.warning("[Router] %s", fallback_reason)
        if r == Route.FULL_DOC_MCP:
            result = _handle_full_doc(client, query, fallback_reason, full_doc_texts, _acc)
        else:
            result = _handle_rag(query, fallback_reason, _acc, **rag_kwargs)
    else:
        groq_tools = _get_groq_tools() if mcp_needed else []

        if r == Route.MCP:
            result = _handle_mcp(client, _mcp_client, groq_tools, query, classification.reason, _acc)
        elif r == Route.BOTH:
            result = _handle_both(client, _mcp_client, groq_tools, query, classification.reason, _acc, **rag_kwargs)
        elif r == Route.FULL_DOC:
            result = _handle_full_doc(client, query, classification.reason, full_doc_texts, _acc)
        elif r == Route.FULL_DOC_MCP:
            result = _handle_full_doc_mcp(client, _mcp_client, groq_tools, query, classification.reason, full_doc_texts, _acc)
        else:
            result = _handle_rag(query, classification.reason, _acc, **rag_kwargs)

    total_route_ms = (time.perf_counter() - t_route_start) * 1000

    # Merge route-level span total if handler didn't populate spans_ms
    if not result.spans_ms:
        result.spans_ms = {"retrieval": 0.0, "reranking": 0.0, "generation": round(total_route_ms, 1)}

    # ── Log every route — single log point for all router paths ───────────
    log_request(
        query=query,
        answer=result.answer,
        route=str(result.route),
        prompt_version=PROMPT_VERSION,
        retrieved_context_ids=result.retrieved_context_ids,
        spans_ms=result.spans_ms,
        input_tokens=_acc.input_tokens,
        output_tokens=_acc.output_tokens,
        mcp_tools_called=[t.name for t in result.mcp_tools_called],
        classifier_decision=classifier_decision,
        out_of_scope=result.out_of_scope,
    )

    return result
