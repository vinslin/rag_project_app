"""Retrieval package for vector store, BM25, hybrid search, and MMR diversity filtering."""

from .vector_store import build_index, clear_index, retrieve, get_collection
from .bm25_search import bm25_retrieve, save_corpus, clear_corpus, load_corpus
from .hybrid_search import hybrid_retrieve, rrf_fuse
from .mmr import mmr_rerank

__all__ = [
    "build_index",
    "clear_index",
    "retrieve",
    "get_collection",
    "bm25_retrieve",
    "save_corpus",
    "clear_corpus",
    "load_corpus",
    "hybrid_retrieve",
    "rrf_fuse",
    "mmr_rerank",
]
