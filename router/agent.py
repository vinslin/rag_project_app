"""Router Agent — orchestrates RAG, MCP, and combined answering.

Entry point: ``route(query, **rag_kwargs) -> RouteResult``

Flow:
    1. QueryClassifier decides the route (rag | mcp | both).
    2. The agent dispatches to the correct path(s).
    3. For "both", a synthesis LLM call produces a compliance verdict.

All MCP communication goes through MCPHttpClient (HTTP + SSE).
All RAG communication goes through rag.pipeline.answer_question.
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

from rag import config
from rag.pipeline import answer_question
from router.classifier import QueryClassifier, Route
from router.mcp_http_client import MCPHttpClient, MCPClientError

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = config.GENERATION_MODEL

# Tools are NOT hardcoded here. They are discovered from the MCP server at
# runtime via MCPHttpClient.to_groq_tools(). This means adding or changing
# a tool on the server side requires zero changes to this agent.

_MCP_LOOP_SYSTEM = (
    "You are a legal contract standards advisor with access to the company's "
    "contract standards database. Answer the user's question by calling the "
    "relevant tools first, then give a precise, well-structured answer "
    "based solely on what the tools return."
)

_SYNTHESIS_SYSTEM = (
    "You are a legal contract compliance analyst. "
    "You have been provided with the company's contract standards and the actual "
    "content retrieved from the contract under review. "
    "Produce a clear, structured compliance report: "
    "(1) what is present and compliant, "
    "(2) what is missing or non-compliant, "
    "(3) a concise overall verdict."
)


# ── Result type ───────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    args: dict
    result_preview: str = ""


@dataclass
class RouteResult:
    """Unified response returned by the router for every query."""

    answer: str
    route: str                            # "rag" | "mcp" | "both"
    route_reason: str
    reasoning: str
    confidence: str
    out_of_scope: bool
    sources: list[dict] = field(default_factory=list)
    mcp_tools_called: list[ToolCall] = field(default_factory=list)
    # RAG debug chunks (present only when RAG ran)
    vector_chunks: list[dict] = field(default_factory=list)
    bm25_chunks: list[dict] = field(default_factory=list)
    mmr_chunks: list[dict] = field(default_factory=list)

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
        }


# ── Internal LLM helpers ──────────────────────────────────────────────────

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
    """Run a Groq tool-calling loop using tools discovered from the MCP server.

    Args:
        groq_tools: Tool schemas fetched from the server via
                    ``MCPHttpClient.to_groq_tools()`` — not hardcoded.

    Returns the final answer text and the list of tool calls made.
    """
    messages: list[dict] = [
        {"role": "system", "content": _MCP_LOOP_SYSTEM},
        {"role": "user", "content": query},
    ]
    tools_called: list[ToolCall] = []

    for iteration in range(max_iters):
        resp = _llm_call(client, messages, tools=groq_tools)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # LLM produced a final answer — exit loop
            return msg.content or "", tools_called

        # Append assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and append results
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    logger.warning("MCP loop reached max_iters=%d without a final answer", max_iters)
    return "[Max iterations reached without a final answer]", tools_called


# ── Route handlers ────────────────────────────────────────────────────────

def _handle_rag(query: str, route_reason: str, **rag_kwargs) -> RouteResult:
    logger.info("[Router] RAG path for query: %r", query[:80])
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
    client: Groq, mcp: MCPHttpClient, groq_tools: list[dict], query: str, route_reason: str
) -> RouteResult:
    logger.info("[Router] MCP path for query: %r", query[:80])
    answer, tools_called = _run_mcp_loop(client, mcp, query, groq_tools)
    tool_names = [t.name for t in tools_called]
    return RouteResult(
        answer=answer,
        route=Route.MCP,
        route_reason=route_reason,
        reasoning=f"Answered from MCP contract standards server. Tools used: {tool_names}",
        confidence="high",
        out_of_scope=False,
        sources=[],
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
    logger.info("[Router] BOTH path for query: %r", query[:80])

    # Step 1 — RAG: retrieve actual contract content
    rag_result = answer_question(query, **rag_kwargs)

    # Guardrails block: propagate immediately, no need to call MCP
    if rag_result.get("out_of_scope"):
        logger.warning("[Router] Query blocked by RAG guardrails — skipping MCP")
        return RouteResult(
            answer=rag_result["answer"],
            route=Route.BOTH,
            route_reason=route_reason,
            reasoning=rag_result.get("reasoning", ""),
            confidence=rag_result.get("confidence", "high"),
            out_of_scope=True,
            sources=[],
            mcp_tools_called=[],
        )

    rag_answer = rag_result.get("answer", "No relevant content found in the uploaded documents.")

    # Step 2 — MCP: fetch applicable company standards using server-discovered tools
    mcp_query = (
        f"What do the company contract standards require regarding the following topic: {query}"
    )
    mcp_answer, tools_called = _run_mcp_loop(client, mcp, mcp_query, groq_tools)

    # Step 3 — Synthesise a compliance verdict
    synthesis_user = (
        f"COMPANY CONTRACT STANDARDS (from MCP server):\n"
        f"{mcp_answer}\n\n"
        f"ACTUAL CONTRACT CONTENT (from document search):\n"
        f"{rag_answer}\n\n"
        f"USER QUESTION:\n{query}"
    )
    synthesis_resp = _llm_call(
        client,
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": synthesis_user},
        ],
    )
    final_answer = synthesis_resp.choices[0].message.content

    tool_names = [t.name for t in tools_called]
    return RouteResult(
        answer=final_answer,
        route=Route.BOTH,
        route_reason=route_reason,
        reasoning=(
            f"[Router: both] "
            f"RAG retrieved document context; "
            f"MCP tools used: {tool_names}; "
            f"LLM synthesised a compliance verdict."
        ),
        confidence="high",
        out_of_scope=False,
        sources=rag_result.get("sources", []),
        mcp_tools_called=tools_called,
        vector_chunks=rag_result.get("vector_chunks", []),
        bm25_chunks=rag_result.get("bm25_chunks", []),
        mmr_chunks=rag_result.get("mmr_chunks", []),
    )


# ── Public API ────────────────────────────────────────────────────────────

_classifier = QueryClassifier()
_mcp_client = MCPHttpClient()

# Cache discovered Groq tools so we only call tools/list once per process
# lifetime. Tools change only when the server is redeployed.
_groq_tools_cache: list[dict] | None = None


def _get_groq_tools() -> list[dict]:
    """Fetch and cache MCP tools in Groq format.

    Calls MCPHttpClient.to_groq_tools() which in turn calls the server's
    tools/list endpoint. The result is cached in-process so subsequent
    requests pay no extra network cost.
    """
    global _groq_tools_cache
    if _groq_tools_cache is None:
        _groq_tools_cache = _mcp_client.to_groq_tools()
        logger.info(
            "[Router] Cached %d Groq tools from MCP server: %s",
            len(_groq_tools_cache),
            [t["function"]["name"] for t in _groq_tools_cache],
        )
    return _groq_tools_cache


def route(query: str, **rag_kwargs) -> RouteResult:
    """Classify a query and route it to the right source(s).

    Args:
        query: The user's natural-language question.
        **rag_kwargs: Forwarded verbatim to ``rag.pipeline.answer_question``
                      (search_mode, retrieval_k, mmr_k, mmr_lambda, final_k,
                       conversation_history, document_type).

    Returns:
        A ``RouteResult`` with the answer and full provenance metadata.
    """
    classification = _classifier.classify(query)
    client = _get_client()

    if classification.route == Route.MCP:
        if not _mcp_client.is_available():
            logger.warning("[Router] MCP server unavailable — falling back to RAG")
            return _handle_rag(
                query,
                route_reason="MCP server unavailable — fell back to RAG.",
                **rag_kwargs,
            )
        groq_tools = _get_groq_tools()
        return _handle_mcp(client, _mcp_client, groq_tools, query, classification.reason)

    if classification.route == Route.BOTH:
        if not _mcp_client.is_available():
            logger.warning("[Router] MCP server unavailable — falling back to RAG only")
            return _handle_rag(
                query,
                route_reason="MCP server unavailable — fell back to RAG only.",
                **rag_kwargs,
            )
        groq_tools = _get_groq_tools()
        return _handle_both(client, _mcp_client, groq_tools, query, classification.reason, **rag_kwargs)

    # Default: RAG
    return _handle_rag(query, classification.reason, **rag_kwargs)
