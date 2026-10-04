# Backward-compatibility shim — real config lives in core.config
from core.config import *  # noqa: F401, F403
from core.config import (
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOP_K, RETRIEVAL_K, FINAL_K,
    COLLECTION_NAME, BM25_CORPUS_PATH, RRF_K, SEARCH_MODE, MMR_K, MMR_LAMBDA,
    CROSS_ENCODER_MODEL, GENERATION_MODEL, GUARDRAIL_ANSWER, NO_DRAFTING_ANSWER,
    RESPONSE_SCHEMA,
)
