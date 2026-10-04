"""BM25 keyword search using rank_bm25.

Persists the document corpus to a JSON file so it survives app restarts.
"""

import json
import os

from rank_bm25 import BM25Okapi

from core import config


def clear_corpus() -> None:
    """Delete the persisted BM25 corpus file."""
    if os.path.exists(config.BM25_CORPUS_PATH):
        os.remove(config.BM25_CORPUS_PATH)


def save_corpus(chunks, source: str = "unknown", page: int = 0) -> None:
    """Append chunk texts and metadata to the BM25 corpus JSON file."""
    os.makedirs(os.path.dirname(config.BM25_CORPUS_PATH), exist_ok=True)

    existing = []
    if os.path.exists(config.BM25_CORPUS_PATH):
        with open(config.BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    for chunk in chunks:
        existing.append({
            "text":    chunk.text,
            "source":  source,
            "page":    page,
            "heading": chunk.heading,
        })

    with open(config.BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_corpus() -> list[dict]:
    """Load the full BM25 corpus from disk."""
    if not os.path.exists(config.BM25_CORPUS_PATH):
        return []
    with open(config.BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_retrieve(query: str, top_k: int = config.RETRIEVAL_K) -> dict:
    """Run BM25 keyword search over the persisted corpus."""
    corpus = load_corpus()
    if not corpus:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    tokenized_corpus = [_tokenize(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    documents, metadatas, distances = [], [], []
    for idx in ranked_indices:
        doc = corpus[idx]
        documents.append(doc["text"])
        metadatas.append({
            "source":  doc["source"],
            "page":    doc["page"],
            "heading": doc.get("heading", ""),
        })
        distances.append(round(float(scores[idx]), 4))

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
