from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


def split_by_headings(text: str, heading_patterns: List[str]) -> List[str]:
    if not heading_patterns:
        return [text]
    pattern = re.compile("|".join(heading_patterns), flags=re.MULTILINE)
    parts: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()].strip())
        last = m.start()
    parts.append(text[last:].strip())
    return [p for p in parts if p.strip()]


def chunk_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j == len(words):
            break
        i = max(0, j - overlap)
    return chunks


def make_chunks(
    text: str,
    chunk_size_words: int = 1000,
    chunk_overlap_words: int = 120,
    heading_patterns: List[str] | None = None,
) -> List[str]:
    if heading_patterns:
        sections = split_by_headings(text, heading_patterns)
        chunks: List[str] = []
        for s in sections:
            chunks.extend(chunk_by_words(s, chunk_size_words, chunk_overlap_words))
        return chunks
    else:
        return chunk_by_words(text, chunk_size_words, chunk_overlap_words)
