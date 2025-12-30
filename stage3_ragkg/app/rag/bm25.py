# === stage3_ragkg/app/rag/bm25.py ===
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.logging_utils import get_logger, log_call

logger = get_logger(__name__)


def _simple_tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer.

    If you have a better tokenizer in utils_text, you can import and use it here.
    """
    return [t for t in text.split() if t]


@dataclass
class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi with persistent storage and doc_id mapping."""

    path: Path

    _bm25: Optional[BM25Okapi] = None
    _doc_ids: Optional[List[str]] = None
    _corpus_tokens: Optional[List[List[str]]] = None

    # ---------- helpers ----------

    @staticmethod
    def exists(path: Path) -> bool:
        """Check whether a BM25 index artifact exists on disk."""
        return Path(path).exists()

    # ---------- lifecycle ----------

    @log_call(level=logging.INFO, include_result=False)
    def build(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """Build BM25 index from documents."""
        logger.info("BM25Index.build: building index for %d documents", len(texts))

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have the same length")

        corpus_tokens = [_simple_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._doc_ids = list(doc_ids)
        self._corpus_tokens = corpus_tokens

        logger.debug("BM25Index.build: built index with %d token lists", len(self._corpus_tokens))

    @log_call(level=logging.INFO, include_result=False)
    def save(self) -> None:
        """Persist the index to disk (pickle)."""
        if self._bm25 is None or self._doc_ids is None or self._corpus_tokens is None:
            raise RuntimeError("BM25 index not built; nothing to save.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "doc_ids": self._doc_ids,
            "corpus": self._corpus_tokens,
            "bm25": self._bm25,
        }
        with self.path.open("wb") as f:
            pickle.dump(data, f)
        logger.info("BM25Index.save: saved BM25 index to %s", self.path)

    @log_call(level=logging.INFO, include_result=True)
    def load(self) -> bool:
        """Load the index from disk if exists."""
        if not self.path.exists():
            logger.warning("BM25Index.load: index path does not exist: %s", self.path)
            return False

        try:
            with self.path.open("rb") as f:
                data = pickle.load(f)
            self._doc_ids = data["doc_ids"]
            self._corpus_tokens = data["corpus"]
            self._bm25 = data["bm25"]
            logger.info(
                "BM25Index.load: loaded BM25 index from %s (documents=%d)",
                self.path,
                len(self._doc_ids) if self._doc_ids is not None else 0,
            )
            return True
        except Exception:
            logger.exception("BM25Index.load: failed to load BM25 index from %s", self.path)
            return False

    # ---------- query ----------

    @log_call(level=logging.DEBUG, include_result=False)
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search with BM25. Returns list of (doc_id, score) sorted by score desc."""
        if self._bm25 is None or self._doc_ids is None:
            raise RuntimeError("BM25 index not loaded/built.")

        logger.debug("BM25Index.search: query=%r top_k=%d", query, top_k)

        scores = self._bm25.get_scores(_simple_tokenize(query))
        doc_scores = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        results = [(doc_id, float(score)) for doc_id, score in doc_scores[:top_k]]

        logger.debug("BM25Index.search: returning %d hits (top_k=%d)", len(results), top_k)
        return results
