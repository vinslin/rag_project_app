"""Document loader: PDF, DOCX, and Markdown.

Parses a 5-field metadata header block from every document so that chunks
carry counterparty, effective_date, doc_type, source_doc, and clause_ref
into ChromaDB at index time.

Public API
----------
load_document(path) -> list[dict]
    Each dict: {text, page, source, source_doc, counterparty,
                effective_date, doc_type}

load_pdf(path) -> list[dict]
    Legacy alias kept for backward compatibility with run_eval.py.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


# ── Header key → metadata field mapping ──────────────────────────────────────

_KEY_MAP: dict[str, str] = {
    "agreement no":   "source_doc",   # rstrip(".") strips trailing dot before lookup
    "document no":    "source_doc",
    "amendment no":   "source_doc",
    "counterparty":   "counterparty",
    "effective date": "effective_date",
    "status":         "_status_raw",   # processed further into doc_type
}

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03",    "april": "04",
    "may":     "05", "june":     "06", "july":  "07",    "august": "08",
    "september":"09","october":  "10", "november":"11",  "december":"12",
}


def _parse_date(raw: str) -> str:
    """Convert '14 January 2026' or '2026-01-14' → ISO 'YYYY-MM-DD'.

    Returns the original string unchanged when parsing fails.
    """
    raw = raw.strip()
    # Already ISO
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    # "14 January 2026" or "1 April 2026"
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        day, month_word, year = m.groups()
        month = _MONTH_MAP.get(month_word.lower())
        if month:
            return f"{year}-{month}-{int(day):02d}"
    return raw


def _status_to_doc_type(status: str) -> str:
    """Derive a short doc_type token from the Status header value."""
    s = status.lower()
    if "amendment" in s:
        return "amendment"
    if "non-disclosure" in s or "nda" in s:
        return "nda"
    if "schedule" in s:
        return "schedule"
    return "msa"


def _doc_type_from_filename(filename: str) -> str:
    """Derive doc_type from filename prefix when status header is absent."""
    base = os.path.basename(filename).upper()
    if base.startswith("AMD"):
        return "amendment"
    if base.startswith("NDA"):
        return "nda"
    if base.startswith("SCH"):
        return "schedule"
    return "msa"


def _parse_header(text: str) -> dict:
    """Extract metadata from a document's header block.

    Reads the first 30 lines looking for 'Key: Value' pairs.
    Returns a dict with keys: source_doc, counterparty, effective_date, doc_type.
    """
    result: dict[str, str] = {
        "source_doc":     "",
        "counterparty":   "",
        "effective_date": "",
        "doc_type":       "",
    }
    for line in text.splitlines()[:30]:
        line = line.strip()
        if ":" not in line:
            continue
        raw_key, _, raw_val = line.partition(":")
        key = raw_key.strip().lower().rstrip(".")
        val = raw_val.strip()
        if not val:
            continue
        field = _KEY_MAP.get(key)
        if field == "source_doc":
            result["source_doc"] = val
        elif field == "counterparty":
            result["counterparty"] = val
        elif field == "effective_date":
            result["effective_date"] = _parse_date(val)
        elif field == "_status_raw":
            result["doc_type"] = _status_to_doc_type(val)

    return result


# ── Table → readable text ─────────────────────────────────────────────────────

def _table_to_text(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to pipe-separated key:value text.

    Empty cells are skipped. The first row is treated as headers.
    """
    if not table:
        return ""
    headers = [str(h).strip() if h else "" for h in table[0]]
    rows: list[str] = []
    for row in table[1:]:
        pairs = []
        for h, v in zip(headers, row):
            cell = str(v).strip() if v else ""
            if cell and h:
                pairs.append(f"{h}: {cell}")
            elif cell:
                pairs.append(cell)
        if pairs:
            rows.append(" | ".join(pairs))
    return "\n".join(rows)


# ── Format-specific loaders ───────────────────────────────────────────────────

def _load_pdf(path: str) -> list[dict]:
    """Load a PDF using pdfplumber. Tables are converted to readable text."""
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError(
            "pdfplumber is required for PDF loading. "
            "Run: pip install pdfplumber"
        ) from e

    source = os.path.basename(path)
    pages_out: list[dict] = []
    meta: dict[str, str] | None = None

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Extract plain text
            text = page.extract_text() or ""

            # Replace tables with structured text
            tables = page.extract_tables() or []
            for table in tables:
                table_text = _table_to_text(table)
                if table_text:
                    text = text + "\n\n" + table_text

            if not text.strip():
                continue

            # Parse header once from page 1
            if meta is None:
                meta = _parse_header(text)
                # Filename fallbacks when header parsing misses fields
                if not meta["source_doc"]:
                    stem = Path(path).stem
                    meta["source_doc"] = stem.split("_")[0] if "_" in stem else stem
                # Filename prefix wins for AMD/NDA/SCH — "Status: Executed" gives
                # a generic "msa" default that the filename can override
                fn_type = _doc_type_from_filename(path)
                if fn_type != "msa":
                    meta["doc_type"] = fn_type
                elif not meta["doc_type"]:
                    meta["doc_type"] = "msa"

            pages_out.append({
                "text":           text,
                "page":           page_number,
                "source":         source,
                "source_doc":     meta.get("source_doc", ""),
                "counterparty":   meta.get("counterparty", ""),
                "effective_date": meta.get("effective_date", ""),
                "doc_type":       meta.get("doc_type", ""),
            })

    return pages_out


def _load_docx(path: str) -> list[dict]:
    """Load a DOCX file using python-docx. Returns a single 'page'."""
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX loading. "
            "Run: pip install python-docx"
        ) from e

    source = os.path.basename(path)
    doc = Document(path)

    parts: list[str] = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            # Paragraph
            text_parts = [run.text for run in block.iterchildren()
                          if run.tag.split("}")[-1] == "r"
                          and hasattr(run, "text") and run.text]
            line = "".join(text_parts).strip()
            if line:
                parts.append(line)
        elif tag == "tbl":
            # Table — convert rows to pipe text
            rows: list[list[str]] = []
            for row in block.iterchildren():
                if row.tag.split("}")[-1] != "tr":
                    continue
                cells = []
                for cell in row.iterchildren():
                    if cell.tag.split("}")[-1] == "tc":
                        cell_text = "".join(
                            n.text or "" for n in cell.iter()
                            if n.tag.split("}")[-1] == "t"
                        ).strip()
                        cells.append(cell_text)
                if cells:
                    rows.append(cells)
            if rows:
                parts.append(_table_to_text(rows))

    full_text = "\n".join(parts)
    if not full_text.strip():
        return []

    meta = _parse_header(full_text)
    return [{
        "text":           full_text,
        "page":           1,
        "source":         source,
        "source_doc":     meta.get("source_doc", ""),
        "counterparty":   meta.get("counterparty", ""),
        "effective_date": meta.get("effective_date", ""),
        "doc_type":       meta.get("doc_type", ""),
    }]


def _load_md(path: str) -> list[dict]:
    """Load a Markdown file as plain text. Returns a single 'page'."""
    source = os.path.basename(path)
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    meta = _parse_header(text)
    return [{
        "text":           text,
        "page":           1,
        "source":         source,
        "source_doc":     meta.get("source_doc", ""),
        "counterparty":   meta.get("counterparty", ""),
        "effective_date": meta.get("effective_date", ""),
        "doc_type":       meta.get("doc_type", ""),
    }]


# ── Public API ────────────────────────────────────────────────────────────────

def load_document(path: str) -> list[dict]:
    """Load any supported document (PDF, DOCX, MD) and return page dicts.

    Each dict contains:
        text          — extracted text (tables converted to readable rows)
        page          — 1-based page number (DOCX/MD always return page 1)
        source        — basename of the file
        source_doc    — Agreement / Document number from header block
        counterparty  — Counterparty name from header block
        effective_date — ISO date string (YYYY-MM-DD) from header block
        doc_type      — "msa" | "amendment" | "nda" | "schedule"
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".docx", ".doc"):
        return _load_docx(path)
    elif ext in (".md", ".txt"):
        return _load_md(path)
    else:
        raise ValueError(f"Unsupported file format: {ext!r} ({path})")


def load_pdf(path: str) -> list[dict]:
    """Legacy alias — loads a PDF via load_document()."""
    return _load_pdf(path)
