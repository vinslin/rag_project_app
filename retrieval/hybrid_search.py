"""Hybrid search: combine vector and BM25 results using Reciprocal Rank Fusion.

RRF score for a document d:
    RRF(d) = Σ  1 / (k + rank_r(d))    for each ranker r
"""

from core import config
from retrieval.vector_store import retrieve as vector_retrieve, get_collection
from retrieval.bm25_search import bm25_retrieve


def rrf_fuse(
    vector_results: dict,
    bm25_results: dict,
    top_k: int,
    rrf_k: int = config.RRF_K,
) -> dict:
    """Merge two result sets using Reciprocal Rank Fusion."""
    doc_map: dict = {}

    v_docs  = vector_results.get("documents", [[]])[0]
    v_metas = vector_results.get("metadatas",  [[]])[0]
    v_dists = vector_results.get("distances",  [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(v_docs, v_metas, v_dists)):
        rrf_score = 1.0 / (rrf_k + rank + 1)
        doc_map[doc] = {"rrf_score": rrf_score, "metadata": meta, "distance": dist, "origin": "vector"}

    b_docs  = bm25_results.get("documents", [[]])[0]
    b_metas = bm25_results.get("metadatas",  [[]])[0]
    b_dists = bm25_results.get("distances",  [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(b_docs, b_metas, b_dists)):
        rrf_score = 1.0 / (rrf_k + rank + 1)
        if doc in doc_map:
            doc_map[doc]["rrf_score"] += rrf_score
            doc_map[doc]["origin"] = "both"
        else:
            doc_map[doc] = {"rrf_score": rrf_score, "metadata": meta, "distance": dist, "origin": "bm25"}

    ranked = sorted(doc_map.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:top_k]

    return {
        "documents":        [[item[0] for item in ranked]],
        "metadatas":        [[item[1]["metadata"] for item in ranked]],
        "distances":        [[item[1]["distance"] for item in ranked]],
        "rrf_scores":       [round(item[1]["rrf_score"], 6) for item in ranked],
        "retrieval_origins": [item[1]["origin"] for item in ranked],
    }


def hybrid_retrieve(
    query: str,
    top_k: int = config.RETRIEVAL_K,
    where: dict | None = None,
    rrf_k: int = config.RRF_K,
) -> dict:
    """Run both vector and BM25 search, then fuse with RRF."""
    collection = get_collection(config.COLLECTION_NAME)
    vector_results = vector_retrieve(collection, query, top_k, where=where)
    bm25_results   = bm25_retrieve(query, top_k)
    fused = rrf_fuse(vector_results, bm25_results, top_k, rrf_k)
    fused["vector_results"] = vector_results
    fused["bm25_results"]   = bm25_results
    return fused
