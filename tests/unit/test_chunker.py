"""Unit tests for ingestion/chunker.py — pure logic, no mocks needed."""

import pytest
from ingestion.chunker import chunk_document, split_into_sections, _extract_clause_ref, Chunk


# ── split_into_sections ───────────────────────────────────────────────────────

def test_split_markdown_headings():
    text = "# Introduction\nSome preamble.\n\n## Section 1\nFirst section body."
    sections = split_into_sections(text)
    headings = [h for h, _ in sections]
    assert "Introduction" in headings
    assert "Section 1" in headings


def test_split_preamble_before_first_heading():
    text = "Opening text.\n\n# Section A\nContent here."
    sections = split_into_sections(text)
    assert sections[0][0] == "Preamble"
    assert "Opening text." in sections[0][1]


def test_split_no_headings_returns_preamble():
    text = "This document has no headings at all."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0][0] == "Preamble"


def test_split_legal_section_numbers():
    text = "1.  Definitions\nAll capitalised terms have meanings.\n2.  Payment\nInvoices due in 30 days."
    sections = split_into_sections(text)
    assert len(sections) >= 2


def test_empty_string_returns_no_sections():
    sections = split_into_sections("")
    assert sections == []


# ── _extract_clause_ref ───────────────────────────────────────────────────────

def test_extract_clause_ref_recognises_section():
    assert _extract_clause_ref("Section 7.3 Payment Terms") == "Section 7.3"


def test_extract_clause_ref_recognises_schedule():
    assert _extract_clause_ref("Schedule B-2 SLA") == "Schedule B-2"


def test_extract_clause_ref_falls_back_to_heading():
    result = _extract_clause_ref("Preamble and Definitions")
    assert result == "Preamble and Definitions"


# ── chunk_document ────────────────────────────────────────────────────────────

def test_short_text_produces_one_chunk():
    text = "# Section 1\nThis is a short section."
    chunks = chunk_document(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_long_text_produces_multiple_chunks():
    long_body = " ".join(["word"] * 1200)
    text = f"# Big Section\n{long_body}"
    chunks = chunk_document(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_overlap_means_words_shared_between_chunks():
    long_body = " ".join([str(i) for i in range(700)])
    text = f"# Numbers\n{long_body}"
    chunks = chunk_document(text, chunk_size=500, overlap=100)
    if len(chunks) >= 2:
        words_0 = set(chunks[0].text.split())
        words_1 = set(chunks[1].text.split())
        assert len(words_0 & words_1) > 0


def test_chunk_index_is_sequential():
    text = "# A\nFirst.\n# B\nSecond.\n# C\nThird."
    chunks = chunk_document(text)
    for i, c in enumerate(chunks):
        assert c.index == i


def test_metadata_propagated_to_chunks():
    text = "# Payment\nDue in 30 days."
    meta = {
        "source_doc": "MSA-001",
        "counterparty": "Acme Corp",
        "effective_date": "2026-01-01",
        "doc_type": "msa",
    }
    chunks = chunk_document(text, metadata=meta)
    for c in chunks:
        assert c.source_doc == "MSA-001"
        assert c.counterparty == "Acme Corp"
        assert c.effective_date == "2026-01-01"
        assert c.doc_type == "msa"


def test_empty_text_produces_no_chunks():
    chunks = chunk_document("")
    assert chunks == []


def test_single_line_clause_is_not_dropped():
    text = "1.  Entire Agreement. This Agreement constitutes the entire agreement."
    chunks = chunk_document(text)
    assert len(chunks) >= 1
    assert any("Entire Agreement" in c.text for c in chunks)
