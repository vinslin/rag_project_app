import chromadb
from rag.embeddings import create_embedding
from rag.bm25_search import save_corpus, clear_corpus


chroma_client = chromadb.PersistentClient(path="./data/chroma")


def clear_index(collection_name):
    """Clear an existing ChromaDB collection and the BM25 corpus.

    Call this ONCE before indexing new documents, not per-page.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    clear_corpus()


def build_index(chunks, collection_name, source="unknown", page=0):
    """Add chunks to a ChromaDB collection (append-only).

    Call clear_index() once before the first call to build_index()
    when rebuilding from scratch.
    """

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    for chunk in chunks:

        chunk_id = f"{source}_p{page}_{chunk.index}"

        embedding = create_embedding(chunk.text)

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[{
                "source": source,
                "page": page,
                "heading": chunk.heading
            }]
        )

    # Persist chunks for BM25 keyword search
    save_corpus(chunks, source=source, page=page)

    return collection


def retrieve(collection, question, top_k, where=None):
    """Query the collection and return the top-K most relevant chunks.

    Args:
        collection: ChromaDB collection to query.
        question: The user's question string.
        top_k: Number of results to return.
        where: Optional ChromaDB metadata filter dict (e.g. {"document_type": "amendment"}).
    """

    query_embedding = create_embedding(question)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    return results


def get_collection(collection_name):
    """Get an existing ChromaDB collection by name."""

    return chroma_client.get_collection(collection_name)
