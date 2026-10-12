#!/usr/bin/env python3
"""Rebuild ChromaDB index and BM25 corpus from the 6 corpus PDFs.

Usage:
    python rebuild_index.py

Excludes test/chunking documents — only the 6 shared-corpus contracts.
"""

import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from dotenv import load_dotenv
load_dotenv()

CORPUS_DOCS = [
    "AMD-2026-014-01_Northwind_Amendment_No_1.pdf",
    "AMD-2026-014-02_Northwind_Amendment_No_2.pdf",
    "MSA-2026-014_Northwind_Master_Services_Agreement.pdf",
    "MSA-2026-022_Vertex_Master_Services_Agreement.pdf",
    "NDA-2026-007_Mutual_Non_Disclosure_Agreement.pdf",
    "SCH-2026-014_Northwind_Schedules.pdf",
]

def main():
    from ingestion.pdf_loader import load_document
    from ingestion.chunker import chunk_document
    from retrieval.vector_store import build_index, clear_index, chroma_client
    from core import config

    docs_dir = os.path.join(APP_ROOT, "documents")

    print("Clearing existing index and BM25 corpus...")
    clear_index(config.COLLECTION_NAME)

    total_chunks = 0
    for filename in CORPUS_DOCS:
        doc_path = os.path.join(docs_dir, filename)
        if not os.path.exists(doc_path):
            print(f"  WARNING: {filename} not found, skipping.")
            continue

        print(f"\nIndexing {filename}...")
        pages = load_document(doc_path)
        doc_chunks = 0
        for page in pages:
            meta = {k: page.get(k, "") for k in
                    ["source_doc", "counterparty", "effective_date", "doc_type"]}
            chunks = chunk_document(page["text"], metadata=meta)
            build_index(chunks, config.COLLECTION_NAME,
                        source=page["source"], page=page["page"])
            doc_chunks += len(chunks)
        print(f"  → {doc_chunks} chunks from {len(pages)} pages")
        total_chunks += doc_chunks

    col = chroma_client.get_collection(config.COLLECTION_NAME)
    print(f"\nDone. Total chunks in index: {col.count()} (expected ~{total_chunks})")

if __name__ == "__main__":
    main()
