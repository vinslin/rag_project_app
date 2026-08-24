"""Hybrid search: combine vector and BM25 results using Reciprocal Rank Fusion.

RRF score for a document d:
    RRF(d) = Σ  1 / (k + rank_r(d))    for each ranker r

where k is a constant (default 60).  Documents appearing in both result sets
receive contributions from both, boosting them above single-source hits.
"""

from . import config
from .vector_store import retrieve as vector_retrieve, get_collection
from .bm25_search import bm25_retrieve


def rrf_fuse(vector_results, bm25_results, top_k, rrf_k=config.RRF_K):
    """Merge two result sets using Reciprocal Rank Fusion.

    Each document is identified by its text content.  Metadata is taken from
    whichever source first contributed the document.

    Args:
        vector_results: ChromaDB-style result dict from vector search.
        bm25_results: ChromaDB-style result dict from BM25 search.
        top_k: Number of fused results to return.
        rrf_k: RRF smoothing constant (default 60).

    Returns:
        dict with keys:
            documents  – list[list[str]]
            metadatas  – list[list[dict]]
            distances  – list[list[float]]  (original vector distances where available)
            rrf_scores – list[float]
    """
    # doc_text → {rrf_score, metadata, distance, origin}
    doc_map = {}

    # --- Score vector results ---
    v_docs = vector_results.get("documents", [[]])[0]
    v_metas = vector_results.get("metadatas", [[]])[0]
    v_dists = vector_results.get("distances", [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(v_docs, v_metas, v_dists)):
        rrf_score = 1.0 / (rrf_k + rank + 1)  # rank is 0-based, formula uses 1-based
        doc_map[doc] = {
            "rrf_score": rrf_score,
            "metadata": meta,
            "distance": dist,
            "origin": "vector",
        }

    # --- Score BM25 results ---
    b_docs = bm25_results.get("documents", [[]])[0]
    b_metas = bm25_results.get("metadatas", [[]])[0]
    b_dists = bm25_results.get("distances", [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(b_docs, b_metas, b_dists)):
        rrf_score = 1.0 / (rrf_k + rank + 1)
        if doc in doc_map:
            # Document found in both — add RRF scores (the fusion boost)
            doc_map[doc]["rrf_score"] += rrf_score
            doc_map[doc]["origin"] = "both"
        else:
            doc_map[doc] = {
                "rrf_score": rrf_score,
                "metadata": meta,
                "distance": dist,
                "origin": "bm25",
            }

    # --- Sort by fused RRF score descending and take top_k ---
    ranked = sorted(doc_map.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:top_k]

    documents = [item[0] for item in ranked]
    metadatas = [item[1]["metadata"] for item in ranked]
    distances = [item[1]["distance"] for item in ranked]
    rrf_scores = [round(item[1]["rrf_score"], 6) for item in ranked]
    retrieval_origins = [item[1]["origin"] for item in ranked]

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
        "rrf_scores": rrf_scores,
        "retrieval_origins": retrieval_origins,
    }


def hybrid_retrieve(query, top_k=config.RETRIEVAL_K, where=None, rrf_k=config.RRF_K):
    """Run both vector and BM25 search, then fuse with RRF.

    Args:
        query: The user's question string.
        top_k: Number of fused results to return.
        where: Optional ChromaDB metadata filter dict.
        rrf_k: RRF smoothing constant.

    Returns:
        Fused result dict (same shape as ChromaDB results + rrf_scores).
    """
    collection = get_collection(config.COLLECTION_NAME)

    # Stage 1a — Vector (semantic) search
    vector_results = vector_retrieve(collection, query, top_k, where=where)

    # Stage 1b — BM25 (keyword) search
    bm25_results = bm25_retrieve(query, top_k)

    # Stage 1c — Reciprocal Rank Fusion
    fused = rrf_fuse(vector_results, bm25_results, top_k, rrf_k)

    # Attach raw results so the UI can show all retrieved chunks
    fused["vector_results"] = vector_results
    fused["bm25_results"] = bm25_results

    return fused
