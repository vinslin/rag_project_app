"""Retrieval layer — embeddings, vector store, BM25, hybrid search, and MMR."""

from retrieval.vector_store import build_index, clear_index, retrieve, get_collection, chroma_client
from retrieval.bm25_search import bm25_retrieve, save_corpus, clear_corpus, load_corpus
from retrieval.hybrid_search import hybrid_retrieve
from retrieval.mmr import mmr_rerank
from retrieval.embeddings import create_embedding

__all__ = [
    "build_index", "clear_index", "retrieve", "get_collection", "chroma_client",
    "bm25_retrieve", "save_corpus", "clear_corpus", "load_corpus",
    "hybrid_retrieve",
    "mmr_rerank",
    "create_embedding",
]
