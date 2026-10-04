"""RAG pipeline use case — entry point for question answering.

Orchestrates: guardrails → generator → response.
"""

from core import config
from generation.generator import get_generator
from guardrails.checker import screen_query


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
) -> dict:
    """Run the full RAG pipeline: guardrails → retrieval → reranking → generation.

    Returns a dict matching RESPONSE_SCHEMA plus raw chunk lists for the UI.
    """
    allowed, reason = screen_query(query)
    if not allowed:
        answer = config.GUARDRAIL_ANSWER if reason == "injection" else config.NO_DRAFTING_ANSWER
        return {
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

    return generator.generate(query, **kwargs)
