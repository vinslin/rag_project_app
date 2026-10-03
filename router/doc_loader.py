"""Full document text loader for the router agent.

When the router needs the entire document (not just top-k chunks), this
module provides it via two strategies tried in priority order:

1. Session store (fast path)
   app.py passes ``full_doc_texts: dict[filename, text]`` when it calls
   route(). The loader joins all stored texts and returns them directly.
   No embedding or network call needed.

2. ChromaDB reconstruction (fallback)
   If no session texts are available (e.g. CLI usage), the loader fetches
   every chunk from the collection, sorts them by (source, page) order,
   and concatenates them to approximate the original document.
   This is a best-effort reconstruction — chunk order within a page
   follows insertion order, which matches the chunker's output order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag import config
from rag.retrieval.vector_store import chroma_client

logger = logging.getLogger(__name__)

_WORDS_PER_TOKEN = 1.3   # rough estimate used for context-window sizing


@dataclass(frozen=True)
class DocumentText:
    text: str
    source: str            # "session" | "chroma"
    filenames: list[str]
    estimated_tokens: int


def load_full_text(
    full_doc_texts: dict[str, str] | None = None,
    collection_name: str = config.COLLECTION_NAME,
) -> DocumentText:
    """Return the full text of all indexed documents.

    Args:
        full_doc_texts: Mapping of {filename: full_text} stored by app.py
                        when the user builds the index. If supplied and
                        non-empty, this is used directly (fast path).
        collection_name: ChromaDB collection to fall back to if no session
                         texts are provided.

    Returns:
        A DocumentText with the joined text and provenance metadata.

    Raises:
        DocumentLoadError: If neither session texts nor ChromaDB chunks
                           are available.
    """
    if full_doc_texts:
        return _from_session(full_doc_texts)
    return _from_chroma(collection_name)


def _from_session(full_doc_texts: dict[str, str]) -> DocumentText:
    """Fast path: join texts that app.py stored at index-build time."""
    filenames = list(full_doc_texts.keys())
    separator = "\n\n" + ("─" * 60) + "\n\n"
    joined = separator.join(
        f"[Document: {name}]\n\n{text}"
        for name, text in full_doc_texts.items()
    )
    tokens = int(len(joined.split()) * _WORDS_PER_TOKEN)
    logger.info(
        "Loaded %d document(s) from session store (~%d tokens): %s",
        len(filenames), tokens, filenames,
    )
    return DocumentText(
        text=joined,
        source="session",
        filenames=filenames,
        estimated_tokens=tokens,
    )


def _from_chroma(collection_name: str) -> DocumentText:
    """Fallback: reconstruct full text from all ChromaDB chunks."""
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as exc:
        raise DocumentLoadError(
            f"ChromaDB collection '{collection_name}' not found: {exc}"
        ) from exc

    # Fetch every chunk — ChromaDB .get() with no filter returns all documents
    result = collection.get(include=["documents", "metadatas"])
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or []

    if not documents:
        raise DocumentLoadError(
            f"No chunks found in collection '{collection_name}'. "
            "Build the index before requesting full-document analysis."
        )

    # Sort by (source filename, page number) to restore reading order
    paired = sorted(
        zip(documents, metadatas),
        key=lambda x: (x[1].get("source", ""), int(x[1].get("page", 0))),
    )

    # Group by source so each document is clearly separated
    from itertools import groupby
    filenames: list[str] = []
    doc_sections: list[str] = []

    for source, group in groupby(paired, key=lambda x: x[1].get("source", "unknown")):
        filenames.append(source)
        chunks_text = "\n\n".join(chunk for chunk, _ in group)
        doc_sections.append(f"[Document: {source}]\n\n{chunks_text}")

    separator = "\n\n" + ("─" * 60) + "\n\n"
    joined = separator.join(doc_sections)
    tokens = int(len(joined.split()) * _WORDS_PER_TOKEN)

    logger.info(
        "Reconstructed %d document(s) from ChromaDB (~%d tokens): %s",
        len(filenames), tokens, filenames,
    )
    return DocumentText(
        text=joined,
        source="chroma",
        filenames=filenames,
        estimated_tokens=tokens,
    )


class DocumentLoadError(Exception):
    """Raised when full document text cannot be loaded from any source."""
