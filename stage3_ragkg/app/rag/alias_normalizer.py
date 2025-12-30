# === stage3_ragkg/app/rag/alias_normalizer.py ===
from __future__ import annotations

import json
import re
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rapidfuzz import fuzz

from .embedder import load_embedder, embed_texts
from ..logging_utils import get_logger, log_call

logger = get_logger(__name__)


ROMAN_TO_ARABIC = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
}


def _strip_accents(s: str) -> str:
    """Remove accents and normalize whitespace."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def _roman_to_arabic_tokens(s: str) -> str:
    """Convert isolated roman numerals to arabic (CALCULUS II -> CALCULUS 2)."""

    def repl(m: re.Match[str]) -> str:
        token = m.group(1).upper()
        return ROMAN_TO_ARABIC.get(token, token)

    return re.sub(r"\b(I|II|III|IV|V|VI|VII|VIII|IX|X)\b", repl, s, flags=re.IGNORECASE)


def normalize_name(s: str) -> str:
    """Apply lightweight text normalization for alias matching."""
    s0 = _strip_accents(s)
    s0 = s0.lower()
    s0 = _roman_to_arabic_tokens(s0)
    s0 = re.sub(r"[_\-]", " ", s0)
    s0 = re.sub(r"[^a-z0-9\s\.]", " ", s0)
    s0 = re.sub(r"\s+", " ", s0)
    return s0.strip()


@log_call(level=logging.DEBUG, include_result=False)
def extract_course_tokens(s: str) -> List[str]:
    """
    Extract potential course-like tokens, for example 'CS101', 'MATH 201', or 'giai tich 1'.

    This is intentionally permissive; canonicalization happens later.
    """
    s_norm = normalize_name(s)
    tokens: List[str] = []
    patterns = [
        r"\b[a-z]{2,4}\s?\d{2,3}[a-z]{0,2}\b",  # cs101, math201a, it 101
        r"\b(calculus|giai tich|dai so)\s*\d\b",  # localized names + index
    ]
    for p in patterns:
        tokens += re.findall(p, s_norm)
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    logger.debug("extract_course_tokens: input_len=%d tokens=%r", len(s), out)
    return out


class AliasNormalizer:
    """
    Resolve user-provided course aliases to canonical course IDs.

    Configuration expects:
      cfg["paths"]["aliases_json"]: JSON file mapping canonical IDs to lists of aliases, e.g.:
        { "CS101": ["cs 101", "intro to cs", "tin hoc dai cuong"], ... }

      cfg["embedding"]["model_name"] and ["normalize_embeddings"] are used for the semantic fallback.

    Resolution order:
      1) exact (case/spacing-insensitive)
      2) fuzzy ratio (RapidFuzz) against canonical IDs and aliases
      3) semantic cosine similarity using sentence-transformer embeddings
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        path = Path(cfg["paths"]["aliases_json"])
        self.alias_map: Dict[str, List[str]] = {}

        if path.exists():
            try:
                self.alias_map = json.loads(path.read_text(encoding="utf-8"))
                total_aliases = sum(len(v) for v in self.alias_map.values())
                logger.info(
                    "AliasNormalizer: loaded aliases from %s canon_ids=%d total_aliases=%d",
                    path,
                    len(self.alias_map),
                    total_aliases,
                )
            except Exception:
                logger.exception("AliasNormalizer: failed to load aliases JSON from %s", path)
                self.alias_map = {}
        else:
            logger.warning("AliasNormalizer: aliases file not found at %s", path)

        # Precompute normalized candidates
        self._canon_norm: Dict[str, str] = {k: normalize_name(k) for k in self.alias_map}
        self._alias_norm: Dict[str, List[str]] = {
            k: [normalize_name(a) for a in v] for k, v in self.alias_map.items()
        }
        # Lazy embedder: only load if semantic fallback is reached
        self._embedder = None

        # Thresholds (can be exposed in cfg if needed)
        retrieval_cfg = self.cfg.get("retrieval", {}) or {}
        self._fuzzy_threshold = int(retrieval_cfg.get("fuzzy_threshold", 85))
        # Stricter default semantic threshold; can be overridden in cfg.
        self._semantic_threshold = float(retrieval_cfg.get("semantic_threshold", 0.45))

        logger.debug(
            "AliasNormalizer: initialized with fuzzy_threshold=%d, "
            "semantic_threshold=%.3f, canon_norm=%d",
            self._fuzzy_threshold,
            self._semantic_threshold,
            len(self._canon_norm),
        )

    @property
    def embedder(self):
        if self._embedder is None:
            mname = self.cfg["embedding"]["model_name"]
            norm = self.cfg["embedding"]["normalize_embeddings"]
            logger.info(
                "AliasNormalizer: lazy-loading embedder model=%s normalize_embeddings=%s",
                mname,
                norm,
            )
            self._embedder = load_embedder(mname, norm)
        return self._embedder

    # ---- Stage-2 compatibility wrapper ----
    @log_call(level=logging.DEBUG, include_result=False)
    def normalize(self, text: str) -> str:
        """
        Normalize a user-provided course description and return a canonical course ID or message.

        If a course is confidently resolved, the canonical ID is returned. If the resolution
        determines that the course code does not exist in the knowledge graph, a user-facing
        message describing that situation is returned instead.
        """
        info = self.resolve(text)

        if info.get("method") == "not-found":
            msg = info.get("message") or f"Mã môn {info.get('matched') or text} không tồn tại trong knowledge graph."
            logger.info(
                "AliasNormalizer.normalize: course_not_found raw=%r msg=%s",
                text,
                msg,
            )
            return msg

        canonical = info.get("canonical") or text
        logger.debug(
            "AliasNormalizer.normalize: raw=%r canonical=%r method=%s conf=%.3f",
            text,
            canonical,
            info.get("method"),
            float(info.get("confidence", 0.0)),
        )
        return canonical

    # ---- Main API ----
    @log_call(level=logging.INFO, include_result=False)
    def resolve(self, text: str) -> Dict[str, Any]:
        """
        Resolve a user-provided text to a canonical course.

        Returns a dictionary with keys:
          - "canonical": canonical course ID or None
          - "matched": the token from the input that was used for matching, or None
          - "method": one of {"none", "exact", "exact-alias", "fuzzy", "semantic", "not-found"}
          - "confidence": float score in [0, 1] indicating match confidence
          - "message": optional string describing the situation (for example, when a course
                       code is considered not present in the knowledge graph)
        """
        tokens = extract_course_tokens(text)
        if not tokens:
            logger.debug("AliasNormalizer.resolve: no course-like tokens found for %r", text)
            return {"canonical": None, "matched": None, "method": "none", "confidence": 0.0}

        # 1) Exact / normalized match
        for t in tokens:
            t_norm = normalize_name(t)
            for canonical, cn in self._canon_norm.items():
                if t_norm == cn:
                    logger.info(
                        "AliasNormalizer.resolve: exact canonical match raw=%r canonical=%s",
                        text,
                        canonical,
                    )
                    return {
                        "canonical": canonical,
                        "matched": t,
                        "method": "exact",
                        "confidence": 1.0,
                    }
            for canonical, aliases in self._alias_norm.items():
                if t_norm in aliases:
                    logger.info(
                        "AliasNormalizer.resolve: exact alias match raw=%r canonical=%s",
                        text,
                        canonical,
                    )
                    return {
                        "canonical": canonical,
                        "matched": t,
                        "method": "exact-alias",
                        "confidence": 0.98,
                    }

        # 2) Fuzzy match (max score across canon + aliases)
        best: Optional[tuple[str, str]] = None
        best_score = 0.0
        for t in tokens:
            t_norm = normalize_name(t)
            # compare against canon
            for canonical, cn in self._canon_norm.items():
                sc = fuzz.token_sort_ratio(t_norm, cn)
                if sc > best_score:
                    best_score = sc
                    best = (canonical, t)
            # compare against aliases
            for canonical, aliases in self._alias_norm.items():
                for a in aliases:
                    sc = fuzz.token_sort_ratio(t_norm, a)
                    if sc > best_score:
                        best_score = sc
                        best = (canonical, t)

        if best and best_score >= self._fuzzy_threshold:
            logger.info(
                "AliasNormalizer.resolve: fuzzy match raw=%r canonical=%s score=%d threshold=%d",
                text,
                best[0],
                best_score,
                self._fuzzy_threshold,
            )
            return {
                "canonical": best[0],
                "matched": best[1],
                "method": "fuzzy",
                "confidence": min(0.9, best_score / 100.0),
            }

        # 2b) Code-like tokens that do not exist in alias_map -> treat as not-found.
        unknown_codes: List[str] = []
        for t in tokens:
            token_compact = re.sub(r"\s+", "", t).upper()
            if re.match(r"^[A-Z]{2,4}\d{2,3}[A-Z]{0,3}$", token_compact) and token_compact not in self.alias_map:
                unknown_codes.append(token_compact)

        if unknown_codes:
            matched_code = unknown_codes[0]
            msg = f"Mã môn {matched_code} không tồn tại trong knowledge graph."
            logger.info(
                "AliasNormalizer.resolve: code-like token not in alias_map; treating as not-found "
                "raw=%r matched=%s",
                text,
                matched_code,
            )
            return {
                "canonical": None,
                "matched": matched_code,
                "method": "not-found",
                "confidence": 0.0,
                "message": msg,
            }

        # 3) Semantic fallback (cosine via dot because vectors are normalized)
        if not self.alias_map:
            logger.info(
                "AliasNormalizer.resolve: no alias_map available; semantic fallback disabled."
            )
            return {"canonical": None, "matched": None, "method": "none", "confidence": 0.0}

        cands: List[str] = []
        cmap: List[str] = []
        for canonical, aliases in self.alias_map.items():
            cands.append(canonical)
            cmap.append(canonical)
            for a in aliases:
                cands.append(a)
                cmap.append(canonical)

        logger.debug(
            "AliasNormalizer.resolve: semantic fallback for raw=%r tokens=%r candidates=%d",
            text,
            tokens,
            len(cands),
        )

        emb_cands = embed_texts(self.embedder, cands)
        emb_tokens = embed_texts(self.embedder, tokens)
        # cosine because embedder uses normalized embeddings (see config)
        sims = emb_tokens @ emb_cands.T
        i, j = divmod(int(np.argmax(sims)), sims.shape[1])

        canonical = cmap[j]
        matched = tokens[i]
        confidence = float(sims[i, j])

        if confidence < self._semantic_threshold:
            msg = f"Mã môn {matched} không tồn tại trong knowledge graph."
            logger.warning(
                "AliasNormalizer.resolve: semantic score below threshold "
                "raw=%r matched=%s conf=%.3f threshold=%.3f",
                text,
                matched,
                confidence,
                self._semantic_threshold,
            )
            return {
                "canonical": None,
                "matched": matched,
                "method": "not-found",
                "confidence": confidence,
                "message": msg,
            }

        logger.info(
            "AliasNormalizer.resolve: semantic match raw=%r canonical=%s matched=%s conf=%.3f",
            text,
            canonical,
            matched,
            confidence,
        )

        return {
            "canonical": canonical,
            "matched": matched,
            "method": "semantic",
            "confidence": confidence,
        }
