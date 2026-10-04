import chromadb

from retrieval.embeddings import create_embedding
from retrieval.bm25_search import save_corpus, clear_corpus


chroma_client = chromadb.PersistentClient(path="./data/chroma")


def clear_index(collection_name: str) -> None:
    """Delete and recreate a ChromaDB collection, then clear the BM25 corpus."""
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    chroma_client.get_or_create_collection(name=collection_name)
    clear_corpus()


def build_index(chunks, collection_name: str, source: str = "unknown", page: int = 0):
    """Add chunks to a ChromaDB collection (append-only)."""
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for chunk in chunks:
        chunk_id = f"{source}_p{page}_{chunk.index}"
        embedding = create_embedding(chunk.text)
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[{"source": source, "page": page, "heading": chunk.heading}],
        )

    save_corpus(chunks, source=source, page=page)
    return collection


def retrieve(collection, question: str, top_k: int, where: dict | None = None) -> dict:
    """Query the collection and return the top-K most relevant chunks."""
    query_embedding = create_embedding(question)
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    return collection.query(**query_kwargs)


def get_collection(collection_name: str):
    """Get an existing ChromaDB collection by name."""
    return chroma_client.get_collection(collection_name)
