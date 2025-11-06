from __future__ import annotations

from typing import Dict, List, Tuple

from app.retriever.utils_text import normalize_vietnamese_text
from app.retriever.chunker import make_chunks
import re
from collections import Counter

def preprocess_text(
    text: str,
    source_name: str,
    heading_patterns: List[str],
    chunk_size_words: int,
    chunk_overlap_words: int,
    page: int | None = None,
    url: str | None = None,
    section_hint: str | None = None,
    method: str = "words",
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
    enrich: bool = False,
    keywords_per_chunk: int = 6,
    hypotheticals_per_chunk: int = 2,
) -> list[dict]:
    clean = normalize_vietnamese_text(text)
    chunks = make_chunks(
        clean,
        chunk_size_words,
        chunk_overlap_words,
        heading_patterns,
        method=method,
        sentences_per_chunk=sentences_per_chunk,
        overlap_sentences=overlap_sentences,
    )

    # --- helpers for metadata enrichment (English only) ---
    def guess_title(s: str) -> str:
        line = s.split("\n", 1)[0].strip()
        return line[:120]

    STOP = set("""
        and or for the a an of in on with from by as at into to than then that this those these which what when where how why whose while who whom
        is are was were be been being have has had do does did can could should would may might must shall will
    """.split())
    word_re = re.compile(r"[A-Za-zÀ-Ỵà-ỵ0-9_’-]+")

    def extract_keywords(s: str, k: int) -> list[str]:
        words = [w.lower() for w in word_re.findall(s)]
        words = [w for w in words if len(w) > 2 and w not in STOP]
        common = [w for w, _ in Counter(words).most_common(k * 2)]
        out: list[str] = []
        for w in common:
            if any(c.isalpha() for c in w) and w not in out:
                out.append(w)
            if len(out) >= k:
                break
        return out

    def make_hypotheticals(title: str, kws: list[str], max_q: int) -> list[str]:
        base: list[str] = []
        if title:
            base.append(f"What is the main idea of “{title}”?")
        if kws:
            base.append(f"Clarify related concepts: {', '.join(kws[:3])}?")
        base.append("What are the applications, prerequisites, or real-world links of this passage?")
        return base[:max_q]
    out = []
    for ch in chunks:
        meta = {
            "source": source_name,
            "page": page,
            "url": url,
            "section_path": section_hint or "",
        }
        if enrich:
            t = guess_title(ch)
            kws = extract_keywords(ch, keywords_per_chunk)
            hypos = make_hypotheticals(t, kws, hypotheticals_per_chunk)
            meta.update({"title": t, "keywords": kws, "hypothetical_questions": hypos})
        out.append({"text": ch, "metadata": meta})
    return out
