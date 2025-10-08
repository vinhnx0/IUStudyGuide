from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, storage_path: Path):
        self.path = storage_path
        self._bm25 = None
        self._docs: List[str] = []
        self._tok_docs: List[List[str]] = []

    def build(self, docs: List[str]) -> None:
        self._docs = docs
        self._tok_docs = [d.lower().split() for d in docs]
        self._bm25 = BM25Okapi(self._tok_docs)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"docs": self._docs, "tok": self._tok_docs, "bm25": self._bm25}, f)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self._docs = data["docs"]
        self._tok_docs = data["tok"]
        self._bm25 = data["bm25"]
        return True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self._bm25:
            return []
        toks = query.lower().split()
        scores = self._bm25.get_scores(toks)
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
