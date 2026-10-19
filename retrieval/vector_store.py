import re

import chromadb

from retrieval.embeddings import create_embedding
from retrieval.bm25_search import save_corpus, clear_corpus


chroma_client = chromadb.PersistentClient(path="./data/chroma")


def _stable_chunk_id(chunk, source: str) -> str:
    """Build a stable, ChromaDB-safe chunk ID from metadata + position.

    Format: {source_doc}_{clause_ref}_{index}
    Falls back to {source}_p{page}_{index} when metadata is absent.
    """
    def _safe(s: str) -> str:
        return re.sub(r"[^a-z0-9\-]", "-", s.lower()).strip("-")

    if hasattr(chunk, "source_doc") and chunk.source_doc:
        doc_part    = _safe(chunk.source_doc)
        clause_part = _safe(chunk.clause_ref or chunk.heading or "chunk")
        return f"{doc_part}_{clause_part}_{chunk.index}"

    return f"{_safe(source)}_chunk_{chunk.index}"


def clear_index(collection_name: str) -> None:
    """Delete and recreate a ChromaDB collection, then clear the BM25 corpus."""
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    chroma_client.get_or_create_collection(name=collection_name)
    clear_corpus()


def build_index(chunks, collection_name: str, source: str = "unknown", page: int = 0):
    """Add chunks to a ChromaDB collection (append-only).

    Stores 8 metadata fields per chunk:
        existing: source, page, heading
        new:      source_doc, counterparty, effective_date, doc_type, clause_ref
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for chunk in chunks:
        chunk_id  = _stable_chunk_id(chunk, source)
        embedding = create_embedding(chunk.text)

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[{
                "source":         source,
                "page":           page,
                "heading":        chunk.heading,
                # new metadata fields
                "source_doc":     getattr(chunk, "source_doc",     ""),
                "counterparty":   getattr(chunk, "counterparty",   ""),
                "effective_date": getattr(chunk, "effective_date", ""),
                "doc_type":       getattr(chunk, "doc_type",       ""),
                "clause_ref":     getattr(chunk, "clause_ref",     ""),
            }],
        )

    save_corpus(chunks, source=source, page=page)
    return collection


def retrieve(collection, question: str, top_k: int, where: dict | None = None) -> dict:
    """Query the collection and return the top-K most relevant chunks."""
    query_embedding = create_embedding(question)
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        top_k,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    return collection.query(**query_kwargs)


def get_collection(collection_name: str):
    """Get an existing ChromaDB collection by name."""
    return chroma_client.get_collection(collection_name)
