from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz
import numpy as np

from app.retriever.embedder import load_embedder, embed_texts


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


def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(s: str) -> str:
    s0 = s.strip().lower()
    s0 = strip_diacritics(s0)
    s0 = re.sub(r"[-_]", " ", s0)
    # MA-101 -> ma101
    s0 = re.sub(r"\b([a-z]{2,4})\s*[-]?\s*(\d{2,3})\b", r"\1\2", s0)
    # Roman numerals -> Arabic
    for r, a in ROMAN_TO_ARABIC.items():
        s0 = re.sub(rf"\b{r.lower()}\b", a, s0)
    s0 = re.sub(r"\s+", " ", s0)
    return s0.strip()


def extract_course_tokens(s: str) -> List[str]:
    # grab likely tokens like "giai tich 1", "ma101", "calculus 1"
    s_norm = normalize_name(s)
    tokens = []
    # patterns
    patterns = [
        r"\b[a-z]{2,4}\d{2,3}\b",
        r"\b[g|giai|giai tich|giai tich]\s*\d\b",
        r"\bcalculus\s*\d\b",
        r"\b\d{1,2}\b",
    ]
    for p in patterns:
        tokens += re.findall(p, s_norm)
    if not tokens and len(s_norm.split()) <= 6:
        tokens.append(s_norm)
    return list(dict.fromkeys(tokens))


class AliasNormalizer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.alias_path = Path(self.paths["aliases_json"])
        self.alias_map = self._load_aliases()
        self.embedder = load_embedder(cfg["embedding"]["model_name"], cfg["embedding"]["normalize_embeddings"])

    def _load_aliases(self) -> Dict[str, List[str]]:
        if self.alias_path.exists():
            with open(self.alias_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # default small seed map (will be extended by ingest)
        base = {
            "Calculus I": ["Giải tích 1", "Giai tich 1", "Giải Tích I", "MA101", "MA-101", "Calculus 1"],
            "Calculus II": ["Giải tích 2", "Giai tich 2", "Giải Tích II", "MA102", "Calculus 2"],
        }
        self._save_aliases(base)
        return base

    def _save_aliases(self, data: Dict[str, List[str]]) -> None:
        self.alias_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alias_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_aliases(self, new_map: Dict[str, List[str]]) -> None:
        base = self._load_aliases()
        for k, aliases in new_map.items():
            s = set(base.get(k, []))
            for a in aliases:
                s.add(a)
            base[k] = sorted(s)
        self._save_aliases(base)
        self.alias_map = base

    def resolve(self, text: str) -> Dict[str, Any]:
        tokens = extract_course_tokens(text)
        if not tokens:
            return {}

        # Exact / normalized exact
        for canonical, aliases in self.alias_map.items():
            # exact
            for a in [canonical] + aliases:
                if a.lower() in text.lower():
                    return {"canonical": canonical, "matched": a, "method": "exact", "confidence": 0.99}
            # normalized exact
            aliases_norm = [normalize_name(a) for a in [canonical] + aliases]
            for t in tokens:
                if normalize_name(t) in aliases_norm:
                    return {
                        "canonical": canonical,
                        "matched": t,
                        "method": "normalized_exact",
                        "confidence": 0.97,
                    }

        # Fuzzy
        best = None
        best_score = 0
        threshold = int(self.cfg["retrieval"].get("fuzzy_threshold", 85))
        for canonical, aliases in self.alias_map.items():
            for a in [canonical] + aliases:
                for t in tokens:
                    score = fuzz.ratio(normalize_name(a), normalize_name(t))
                    if score > best_score:
                        best_score = score
                        best = (canonical, t)
        if best and best_score >= threshold:
            return {
                "canonical": best[0],
                "matched": best[1],
                "method": "fuzzy",
                "confidence": min(0.9, best_score / 100.0),
            }

        # Semantic (embedding cosine)
        # Build small candidate list (canonical + aliases flattened)
        cands = []
        cand_map = []
        for canonical, aliases in self.alias_map.items():
            for a in [canonical] + aliases:
                cands.append(a)
                cand_map.append(canonical)
        emb_cands = embed_texts(self.embedder, cands)
        emb_tokens = embed_texts(self.embedder, tokens)
        sims = (emb_tokens @ emb_cands.T)  # cosine if normalized
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        canonical = cand_map[int(j)]
        matched = tokens[int(i)]
        conf = float(sims[i, j])
        return {"canonical": canonical, "matched": matched, "method": "semantic", "confidence": conf}
