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
    method: str = "words",
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Create chunks from raw text. Supports word-level and sentence-level chunking.
    """
    def split_sentences(t: str) -> List[str]:
        # Basic sentence split for VI/EN using ., !, ?, … and newlines.
        # Minimizes false splits on abbreviations with a simple lookbehind.
        t = re.sub(r"\s+", " ", t.strip())
        parts = re.split(r"(?<=[\.!\?…])\s+(?=[A-ZÀ-Ỵ0-9])", t)
        if len(parts) == 1:
            parts = re.split(r"\n+", t)
        return [p.strip() for p in parts if p.strip()]

    def chunk_by_sentences(text: str, n: int, overlap: int) -> List[str]:
        sents = split_sentences(text)
        if not sents:
            return []
        out: List[str] = []
        i = 0
        while i < len(sents):
            j = min(i + n, len(sents))
            ch = " ".join(sents[i:j]).strip()
            if ch:
                out.append(ch)
            if j == len(sents):
                break
            i = max(0, j - overlap)
        return out

    # Split by headings first (if any), then chunk each section.
    texts = split_by_headings(text, heading_patterns) if heading_patterns else [text]
    chunks: List[str] = []
    for s in texts:
        if method == "sentence":
            chunks.extend(chunk_by_sentences(s, sentences_per_chunk, overlap_sentences))
        else:
            chunks.extend(chunk_by_words(s, chunk_size_words, chunk_overlap_words))
    return chunks