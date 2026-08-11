"""Default configuration constants for the RAG pipeline."""


# Chunking

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100


# Retrieval

TOP_K = 5
RETRIEVAL_K = 20   # candidates fetched from Chroma (Stage 1: fast)
FINAL_K = 5         # chunks kept after cross-encoder reranking (Stage 2: accurate)
COLLECTION_NAME = "legal_contracts"


# Reranking (Cross-Encoder)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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
