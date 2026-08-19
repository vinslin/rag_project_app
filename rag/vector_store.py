import chromadb
from rag.embeddings import create_embedding
from rag.bm25_search import save_corpus, clear_corpus


chroma_client = chromadb.PersistentClient(path="./data/chroma")


def build_index(chunks, collection_name, source="unknown", page=0):
    """Build (or rebuild) a ChromaDB collection from Chunk dataclass objects."""

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    # Clear existing collection
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        # Also clear the BM25 corpus when rebuilding
        clear_corpus()

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
