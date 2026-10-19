"""Maximal Marginal Relevance (MMR) for diversity-aware chunk selection.

    MMR = argmax [ λ · sim(d, q) - (1-λ) · max sim(d, d_selected) ]
"""

import numpy as np

from core import config
from retrieval.embeddings import create_embedding


def _cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def mmr_rerank(
    query: str,
    results: dict,
    mmr_k: int = config.MMR_K,
    mmr_lambda: float = config.MMR_LAMBDA,
) -> dict:
    """Apply MMR to select diverse, relevant chunks from retrieval results."""
    documents        = results["documents"][0]
    metadatas        = results["metadatas"][0]
    distances        = results["distances"][0]
    rrf_scores       = results.get("rrf_scores", [])
    retrieval_origins = results.get("retrieval_origins", [])

    n = len(documents)
    if n == 0:
        return results
    if n <= mmr_k:
        results["mmr_scores"] = [1.0] * n
        return results

    query_embedding = create_embedding(query)
    doc_embeddings  = [create_embedding(doc) for doc in documents]
    relevance_scores = [_cosine_similarity(e, query_embedding) for e in doc_embeddings]

    selected_indices: list[int] = []
    remaining_indices = list(range(n))
    mmr_scores: list[float] = []

    for _ in range(mmr_k):
        best_idx, best_score = None, -float("inf")
        for idx in remaining_indices:
            relevance = relevance_scores[idx]
            max_sim = (
                max(_cosine_similarity(doc_embeddings[idx], doc_embeddings[s]) for s in selected_indices)
                if selected_indices else 0.0
            )
            score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if score > best_score:
                best_score, best_idx = score, idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            mmr_scores.append(round(best_score, 6))

    filtered: dict = {
        "documents": [[documents[i] for i in selected_indices]],
        "metadatas": [[metadatas[i] for i in selected_indices]],
        "distances": [[distances[i] for i in selected_indices]],
        "mmr_scores": mmr_scores,
    }
    if rrf_scores:
        filtered["rrf_scores"] = [rrf_scores[i] for i in selected_indices]
    if retrieval_origins:
        filtered["retrieval_origins"] = [retrieval_origins[i] for i in selected_indices]
    return filtered
