"""Default configuration constants for the RAG pipeline."""


# Chunking

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100


# Retrieval

TOP_K = 5
RETRIEVAL_K = 50    # candidates from each search method (Stage 1: RRF fusion)
FINAL_K = 5         # chunks kept after cross-encoder reranking (Stage 4: accurate)
COLLECTION_NAME = "legal_contracts"


# BM25 & Hybrid Search

BM25_CORPUS_PATH = "./data/bm25_corpus.json"  # persisted corpus for BM25 indexing
RRF_K = 60                                     # RRF constant (standard default)
SEARCH_MODE = "hybrid"                          # "hybrid", "vector", or "bm25"


# MMR (Maximal Marginal Relevance)

MMR_K = 20          # diverse chunks to keep after MMR (Stage 2)
MMR_LAMBDA = 0.7    # relevance-vs-diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity)


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
