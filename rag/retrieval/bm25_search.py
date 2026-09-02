"""BM25 keyword search using rank_bm25.

Persists the document corpus to a JSON file so it survives app restarts,
and builds a BM25Okapi index at query time for keyword-based retrieval.
"""

import json
import os

from rank_bm25 import BM25Okapi

from rag import config


# ---------------------------------------------------------------------------
# Corpus persistence
# ---------------------------------------------------------------------------

def clear_corpus():
    """Delete the persisted BM25 corpus file (called when rebuilding the index)."""
    if os.path.exists(config.BM25_CORPUS_PATH):
        os.remove(config.BM25_CORPUS_PATH)


def save_corpus(chunks, source="unknown", page=0):
    """Append chunk texts and metadata to the BM25 corpus JSON file.

    Each entry stores the raw text, source filename, page number, and heading
    so that BM25 results carry the same metadata as vector-store results.
    """
    os.makedirs(os.path.dirname(config.BM25_CORPUS_PATH), exist_ok=True)

    # Load existing corpus (if any)
    existing = []
    if os.path.exists(config.BM25_CORPUS_PATH):
        with open(config.BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Append new chunks
    for chunk in chunks:
        existing.append({
            "text": chunk.text,
            "source": source,
            "page": page,
            "heading": chunk.heading,
        })

    with open(config.BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_corpus():
    """Load the full BM25 corpus from disk.

    Returns:
        list[dict]: Each dict has keys: text, source, page, heading.
        Returns an empty list if the corpus file doesn't exist.
    """
    if not os.path.exists(config.BM25_CORPUS_PATH):
        return []

    with open(config.BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(text):
    """Simple lowercase whitespace tokenizer (matches BM25Okapi defaults)."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def bm25_retrieve(query, top_k=config.RETRIEVAL_K):
    """Run BM25 keyword search over the persisted corpus.

    Args:
        query: The user's question string.
        top_k: Number of results to return.

    Returns:
        dict matching the ChromaDB result format:
            documents  – list[list[str]]
            metadatas  – list[list[dict]]
            distances  – list[list[float]]  (BM25 scores, higher = better)
    """
    corpus = load_corpus()
    if not corpus:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    tokenized_corpus = [_tokenize(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top-K indices sorted by score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    documents = []
    metadatas = []
    distances = []

    for idx in ranked_indices:
        doc = corpus[idx]
        documents.append(doc["text"])
        metadatas.append({
            "source": doc["source"],
            "page": doc["page"],
            "heading": doc.get("heading", ""),
        })
        # Store BM25 score as "distance" (higher = more relevant)
        distances.append(round(float(scores[idx]), 4))

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
