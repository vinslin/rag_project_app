# Backward-compatibility shim
from retrieval.vector_store import (  # noqa: F401
    chroma_client, build_index, clear_index, retrieve, get_collection,
)
