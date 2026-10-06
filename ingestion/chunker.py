"""Document chunking: markdown-aware section splitting with overlapping windows.

Each Chunk now carries 5 metadata fields parsed from the document header
at load time (source_doc, counterparty, effective_date, doc_type, clause_ref).
These fields are stored in ChromaDB at index time and used for filtering
and context building at retrieval time.
"""

import re
from dataclasses import dataclass, field

from core import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Matches legal section/schedule references in headings, e.g.:
#   "Section 7.3", "Section 3", "Schedule B-2", "Schedule C", "Clause 5.1"
_CLAUSE_RE = re.compile(
    r"\b(Section|Clause|Schedule|Appendix|Exhibit|Annex)\s+[\w][\w.\-]*",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    text:           str
    heading:        str   # nearest preceding markdown heading (or "Preamble")
    index:          int   # position within the parent document
    source_doc:     str = ""   # Agreement / Document number  (e.g. "MSA-2026-014")
    counterparty:   str = ""   # Counterparty name            (e.g. "Northwind Logistics Private Limited")
    effective_date: str = ""   # ISO date                     (e.g. "2026-01-14")
    doc_type:       str = ""   # "msa" | "amendment" | "nda" | "schedule"
    clause_ref:     str = ""   # normalised section reference (e.g. "Section 7.3")


def _extract_clause_ref(heading: str) -> str:
    """Return the first legal section/schedule reference found in a heading.

    Falls back to the first 80 characters of the heading itself.
    """
    m = _CLAUSE_RE.search(heading)
    if m:
        return m.group(0).strip()
    return heading[:80].strip()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) pairs at heading lines."""
    sections: list[tuple[str, str]] = []
    heading, lines = "Preamble", []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            body = "\n".join(lines).strip()
            if body:
                sections.append((heading, body))
            heading, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        sections.append((heading, body))
    return sections


def chunk_document(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping word-window chunks.

    Parameters
    ----------
    text        : document text (plain or markdown-formatted)
    chunk_size  : target chunk size in words (approximate tokens)
    overlap     : overlap between consecutive windows in words
    metadata    : dict with keys source_doc, counterparty, effective_date,
                  doc_type — carried onto every Chunk produced from this text
    """
    if metadata is None:
        metadata = {}

    source_doc     = metadata.get("source_doc", "")
    counterparty   = metadata.get("counterparty", "")
    effective_date = metadata.get("effective_date", "")
    doc_type       = metadata.get("doc_type", "")

    chunks: list[Chunk] = []

    for heading, body in split_into_sections(text):
        clause_ref = _extract_clause_ref(heading)
        words = body.split()

        if len(words) <= chunk_size:
            chunks.append(Chunk(
                text=body,
                heading=heading,
                index=len(chunks),
                source_doc=source_doc,
                counterparty=counterparty,
                effective_date=effective_date,
                doc_type=doc_type,
                clause_ref=clause_ref,
            ))
            continue

        step = chunk_size - overlap
        for start in range(0, len(words), step):
            window = words[start:start + chunk_size]
            chunks.append(Chunk(
                text=" ".join(window),
                heading=heading,
                index=len(chunks),
                source_doc=source_doc,
                counterparty=counterparty,
                effective_date=effective_date,
                doc_type=doc_type,
                clause_ref=clause_ref,
            ))
            if start + chunk_size >= len(words):
                break

    return chunks
