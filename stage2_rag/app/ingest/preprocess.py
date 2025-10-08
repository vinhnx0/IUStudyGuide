from __future__ import annotations

from typing import Dict, List, Tuple

from app.retriever.utils_text import normalize_vietnamese_text
from app.retriever.chunker import make_chunks


def preprocess_text(
    text: str,
    source_name: str,
    heading_patterns: List[str],
    chunk_size_words: int,
    chunk_overlap_words: int,
    page: int | None = None,
    url: str | None = None,
    section_hint: str | None = None,
) -> list[dict]:
    clean = normalize_vietnamese_text(text)
    chunks = make_chunks(clean, chunk_size_words, chunk_overlap_words, heading_patterns)
    out = []
    for ch in chunks:
        out.append(
            {
                "text": ch,
                "metadata": {
                    "source": source_name,
                    "page": page,
                    "url": url,
                    "section_path": section_hint or "",
                },
            }
        )
    return out
