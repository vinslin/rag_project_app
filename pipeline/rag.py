"""RAG pipeline use case — entry point for question answering.

Orchestrates: guardrails → generator → log → response.

When called directly (router OFF), log=True so every request is logged here.
When called from router/agent.py, log=False so the router logs instead,
ensuring each query is logged exactly once with full route context.
"""

import time

from core import config
from generation.generator import get_generator
from generation.prompts import PROMPT_VERSION
from guardrails.checker import screen_query
from obs.request_logger import log_request


def _detect_metadata_filter(query: str) -> dict | None:
    if "amendment" in query.lower():
        return {"document_type": "amendment"}
    return None


def answer_question(
    query: str,
    document_type: str | None = None,
    backend: str | None = None,
    search_mode: str | None = None,
    conversation_history: list | None = None,
    retrieval_k: int | None = None,
    mmr_k: int | None = None,
    mmr_lambda: float | None = None,
    final_k: int | None = None,
    log: bool = True,
) -> dict:
    """Run the full RAG pipeline: guardrails → retrieval → reranking → generation.

    Args:
        log: When True (default), write a structured log record for this request.
             Set to False when the router agent is handling logging instead.

    Returns a dict matching RESPONSE_SCHEMA plus raw chunk lists for the UI.
    """
    t_start = time.perf_counter()

    allowed, reason = screen_query(query)
    if not allowed:
        answer = config.GUARDRAIL_ANSWER if reason == "injection" else config.NO_DRAFTING_ANSWER
        result = {
            "answer": answer,
            "reasoning": (
                f"The question was rejected by the input guardrails before "
                f"retrieval: {reason} requests are answered with a fixed safe "
                "response and never passed to the model."
            ),
            "sources":      [],
            "confidence":   "high",
            "out_of_scope": True,
        }
        if log:
            log_request(
                query=query,
                answer=answer,
                route="guardrail",
                prompt_version=PROMPT_VERSION,
                retrieved_context_ids=[],
                spans_ms={"retrieval": 0.0, "reranking": 0.0, "generation": 0.0},
                input_tokens=0,
                output_tokens=0,
                out_of_scope=True,
            )
        return result

    where = {"document_type": document_type} if document_type else _detect_metadata_filter(query)
    generator = get_generator(backend)

    kwargs: dict = {
        "where":                where,
        "search_mode":          search_mode or config.SEARCH_MODE,
        "conversation_history": conversation_history,
    }
    if retrieval_k is not None:
        kwargs["retrieval_k"] = retrieval_k
    if mmr_k is not None:
        kwargs["mmr_k"] = mmr_k
    if mmr_lambda is not None:
        kwargs["mmr_lambda"] = mmr_lambda
    if final_k is not None:
        kwargs["final_k"] = final_k

    result = generator.generate(query, **kwargs)

    if log:
        log_request(
            query=query,
            answer=result.get("answer", ""),
            route="rag",
            prompt_version=result.get("_prompt_version", PROMPT_VERSION),
            retrieved_context_ids=result.get("_retrieved_context_ids", []),
            spans_ms=result.get("_spans_ms", {}),
            input_tokens=result.get("_input_tokens", 0),
            output_tokens=result.get("_output_tokens", 0),
            out_of_scope=result.get("out_of_scope", False),
        )

    return result
