# === stage3_ragkg/app/rag/chunker.py ===
from __future__ import annotations

import logging
import re
from typing import List

from ..logging_utils import get_logger, log_call

logger = get_logger(__name__)


@log_call(level=logging.DEBUG, include_result=False)
def split_by_headings(text: str, heading_patterns: List[str]) -> List[str]:
    """
    Split text into sections based on heading patterns.

    If no heading patterns are provided, the entire text is returned as a single section.
    """
    if not heading_patterns:
        logger.debug("split_by_headings: no heading_patterns provided; returning full text as one part")
        return [text]
    pattern = re.compile("|".join(heading_patterns), flags=re.MULTILINE)
    parts: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()].strip())
        last = m.start()
    parts.append(text[last:].strip())
    parts = [p for p in parts if p.strip()]
    logger.debug(
        "split_by_headings: patterns=%d sections=%d text_len=%d",
        len(heading_patterns),
        len(parts),
        len(text),
    )
    return parts


@log_call(level=logging.INFO, include_result=False)
def make_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
    heading_patterns: List[str] | None = None,
    method: str = "words",
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Main chunking entrypoint.

    Supports:
      - word-based chunking with overlap
      - sentence-based chunking with overlap
      - optional pre-splitting by headings
    """

    def chunk_by_words(t: str, size: int, overlap: int) -> List[str]:
        words = t.split()
        if not words:
            return []
        chunks: List[str] = []
        i = 0
        while i < len(words):
            j = min(i + size, len(words))
            ch = " ".join(words[i:j]).strip()
            if ch:
                chunks.append(ch)
            if j == len(words):
                break
            i = max(0, j - overlap)
        return chunks

    def split_sentences(s: str) -> List[str]:
        s = re.sub(r"\s+", " ", s.strip())
        parts = re.split(r"(?<=[\.!\?…])\s+(?=[A-ZÀ-Ỵ0-9])", s)
        if len(parts) == 1:
            parts = re.split(r"\n+", s)
        return [p.strip() for p in parts if p.strip()]

    def chunk_by_sentences(s: str, n: int, overlap: int) -> List[str]:
        sent = split_sentences(s)
        out: List[str] = []
        i = 0
        while i < len(sent):
            j = min(i + n, len(sent))
            ch = " ".join(sent[i:j]).strip()
            if ch:
                out.append(ch)
            if j == len(sent):
                break
            i = max(0, j - overlap)
        return out

    sections = split_by_headings(text, heading_patterns or [])
    chunks: List[str] = []
    for s in sections:
        if method == "sentence":
            chunks.extend(chunk_by_sentences(s, sentences_per_chunk, overlap_sentences))
        else:
            chunks.extend(chunk_by_words(s, chunk_size, chunk_overlap))

    logger.info(
        "make_chunks: method=%s sections=%d chunks=%d text_len=%d",
        method,
        len(sections),
        len(chunks),
        len(text),
    )
    return chunks
