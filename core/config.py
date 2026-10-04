"""Application-wide configuration constants.

Single source of truth — all packages import from here.
Previously lived at rag/config.py (kept as a shim for backward compat).
"""

# Chunking
CHUNK_SIZE_TOKENS   = 500
CHUNK_OVERLAP_TOKENS = 100

# Retrieval
TOP_K          = 5
RETRIEVAL_K    = 50     # candidates from each search method (Stage 1: RRF fusion)
FINAL_K        = 5      # chunks kept after cross-encoder reranking (Stage 4)
COLLECTION_NAME = "legal_contracts"

# BM25 & Hybrid Search
BM25_CORPUS_PATH = "./data/bm25_corpus.json"
RRF_K            = 60                      # RRF constant (standard default)
SEARCH_MODE      = "hybrid"                # "hybrid", "vector", or "bm25"

# MMR (Maximal Marginal Relevance)
MMR_K      = 20    # diverse chunks to keep after MMR (Stage 2)
MMR_LAMBDA = 0.7   # relevance-vs-diversity tradeoff

# Reranking (Cross-Encoder)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation
GENERATION_MODEL = "openai/gpt-oss-20b"   # served via Groq

# Guardrails
GUARDRAIL_ANSWER    = "I'm sorry, but I cannot process this request."
NO_DRAFTING_ANSWER  = (
    "I'm sorry, but I can only answer questions about existing contract documents. "
    "Drafting new contract language is outside my scope."
)

# Response Schema
RESPONSE_SCHEMA = {
    "answer":      str,
    "reasoning":   str,
    "sources":     list,
    "confidence":  str,
    "out_of_scope": bool,
}
