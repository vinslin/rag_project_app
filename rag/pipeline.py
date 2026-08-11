
from . import config
from .generator import get_generator
from .guardrails import screen_query


def detect_metadata_filter(query):
  
    if "amendment" in query.lower():
        return {"document_type": "amendment"}
    return None


def answer_question(query, document_type=None, backend=None):

    # Screen the query through guardrails before anything else;
    # rejections carry out_of_scope=true with a fixed rejection message
    # so the app can show them verbatim.
    allowed, reason = screen_query(query)
    if not allowed:
        answer = config.GUARDRAIL_ANSWER if reason == "injection" else config.NO_DRAFTING_ANSWER
        return {
            "answer": answer,
            "reasoning": (
                "The question was rejected by the input guardrails before "
                f"retrieval: {reason} requests are answered with a fixed safe "
                "response and never passed to the model."
            ),
            "sources": [],
            "confidence": "high",
            "out_of_scope": True,
        }

    where = {"document_type": document_type} if document_type else detect_metadata_filter(query)

    generator = get_generator(backend)
    return generator.generate(query, where=where)
