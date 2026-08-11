"""Document chunking: markdown-aware section splitting with overlapping windows."""

import re
from dataclasses import dataclass

from . import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    text: str
    heading: str  # nearest preceding markdown heading (or "Preamble")
    index: int    # position within the parent document


def split_into_sections(text):
    """Split markdown text into (heading, body) pairs at heading lines.

    Text before the first heading is grouped under "Preamble".
    """
    sections = []
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


def chunk_document(text, chunk_size=config.CHUNK_SIZE_TOKENS, overlap=config.CHUNK_OVERLAP_TOKENS):
   
    chunks = []
    for heading, body in split_into_sections(text):
        words = body.split()
        if len(words) <= chunk_size:
            chunks.append(Chunk(text=body, heading=heading, index=len(chunks)))
            continue
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            window = words[start:start + chunk_size]
            chunks.append(Chunk(text=" ".join(window), heading=heading, index=len(chunks)))
            if start + chunk_size >= len(words):
                break
    return chunks
