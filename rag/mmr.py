"""Maximal Marginal Relevance (MMR) for diversity-aware chunk selection.

After RRF fusion returns top-N candidates, MMR iteratively selects chunks
that are both relevant to the query AND diverse from already-selected chunks:

    MMR = argmax [ λ · sim(d, q) - (1-λ) · max sim(d, d_selected) ]

where λ controls the relevance-vs-diversity tradeoff:
    λ = 1.0  → pure relevance (no diversity)
    λ = 0.0  → pure diversity (ignores relevance)
    λ = 0.7  → balanced (default)
"""

import numpy as np

from . import config
from .embeddings import create_embedding


def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def mmr_rerank(query, results, mmr_k=config.MMR_K, mmr_lambda=config.MMR_LAMBDA):
    """Apply MMR to select diverse, relevant chunks from retrieval results.

    Args:
        query: The user's question string.
        results: ChromaDB-style result dict (documents, metadatas, distances).
                 May also contain rrf_scores and retrieval_origins from hybrid search.
        mmr_k: Number of diverse chunks to select.
        mmr_lambda: Relevance-vs-diversity tradeoff (0.0–1.0).

    Returns:
        Filtered result dict with the same structure, keeping only the top mmr_k
        diverse chunks.  Adds an 'mmr_scores' key with the MMR score for each
        selected chunk.
    """
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    rrf_scores = results.get("rrf_scores", [])
    retrieval_origins = results.get("retrieval_origins", [])

    n = len(documents)
    if n == 0:
        return results

    # If we have fewer documents than mmr_k, return them all
    if n <= mmr_k:
        results["mmr_scores"] = [1.0] * n
        return results

    # Compute embeddings for query and all candidate chunks
    query_embedding = create_embedding(query)
    doc_embeddings = [create_embedding(doc) for doc in documents]

    # Compute relevance scores: cosine similarity between each doc and query
    relevance_scores = [
        _cosine_similarity(doc_emb, query_embedding)
        for doc_emb in doc_embeddings
    ]

    # Iterative MMR selection
    selected_indices = []
    remaining_indices = list(range(n))
    mmr_scores = []

    for _ in range(mmr_k):
        best_idx = None
        best_mmr_score = -float("inf")

        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component: max similarity to any already-selected doc
            if selected_indices:
                max_sim_to_selected = max(
                    _cosine_similarity(doc_embeddings[idx], doc_embeddings[sel])
                    for sel in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            # MMR score
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            mmr_scores.append(round(best_mmr_score, 6))

    # Build filtered results preserving the same structure
    filtered = {
        "documents": [[documents[i] for i in selected_indices]],
        "metadatas": [[metadatas[i] for i in selected_indices]],
        "distances": [[distances[i] for i in selected_indices]],
        "mmr_scores": mmr_scores,
    }

    # Carry forward rrf_scores if present
    if rrf_scores:
        filtered["rrf_scores"] = [rrf_scores[i] for i in selected_indices]

    # Carry forward retrieval_origins if present
    if retrieval_origins:
        filtered["retrieval_origins"] = [retrieval_origins[i] for i in selected_indices]

    return filtered
