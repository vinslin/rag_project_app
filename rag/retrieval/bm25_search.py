# Backward-compatibility shim
from retrieval.bm25_search import (  # noqa: F401
    bm25_retrieve, save_corpus, clear_corpus, load_corpus,
)
