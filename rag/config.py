"""Default configuration constants for the RAG pipeline."""


# Chunking

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100


# Retrieval

TOP_K = 5
COLLECTION_NAME = "legal_contracts"


# Generation

GENERATION_MODEL = "gemini-3.5-flash"


# Guardrails

GUARDRAIL_ANSWER = (
    "I'm sorry, but I cannot process this request. "
    "Your input appears to contain instructions that conflict with my guidelines. "
    "Please rephrase your question about the contract documents."
)

NO_DRAFTING_ANSWER = (
    "I'm sorry, but I can only answer questions about existing contract documents. "
    "I cannot draft, write, or compose new contract language, clauses, or provisions. "
    "Please ask a question about what is already in the documents."
)


# Response Schema

RESPONSE_SCHEMA = {
    "answer": str,
    "reasoning": str,
    "sources": list,
    "confidence": str,
    "out_of_scope": bool,
}
