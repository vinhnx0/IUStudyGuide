# === stage3_ragkg/app/rag/hybrid.py ===
from __future__ import annotations

import logging
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient  # type: ignore

from ..logging_utils import get_logger, log_call
from .bm25 import BM25Index
from .embedder import embed_texts, load_embedder
from .indexing import IndexNotReadyError, _make_qdrant_client

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
logger = get_logger(__name__)


@log_call(level=logging.DEBUG, include_result=False)
def mmr(doc_embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int, lambda_mult: float) -> List[int]:
    n_docs = int(doc_embeddings.shape[0])
    if top_k <= 0 or n_docs == 0:
        return []

    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10
        return v / norm

    doc_embeddings = _normalize(doc_embeddings)
    query_embedding = _normalize(query_embedding.reshape(1, -1))[0]

    sim_to_query = doc_embeddings @ query_embedding
    sim_between_docs = doc_embeddings @ doc_embeddings.T

    selected: List[int] = []
    candidate_indices = list(range(n_docs))

    for _ in range(min(top_k, len(candidate_indices))):
        mmr_scores: List[Tuple[int, float]] = []
        for idx in candidate_indices:
            s_q = sim_to_query[idx]
            s_div = max(sim_between_docs[idx, j] for j in selected) if selected else 0.0
            score = lambda_mult * s_q - (1.0 - lambda_mult) * s_div
            mmr_scores.append((idx, float(score)))
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected


class HybridRetriever:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        bm25_path = Path(cfg["paths"]["bm25_index"])
        self.bm25 = BM25Index(bm25_path)
        loaded = self.bm25.load()
        if not loaded:
            logger.error("HybridRetriever: BM25 index not found at %s", bm25_path)
            raise RuntimeError(f"BM25 index not found at {bm25_path}")

        self.collection_name = cfg["paths"]["qdrant_collection"]
        self.qdrant_client: QdrantClient = _make_qdrant_client(cfg)

        embedding_cfg = cfg.get("embedding", {}) or {}
        model_name = embedding_cfg.get("model_name")
        normalize_embeddings = embedding_cfg.get("normalize_embeddings", True)
        device = embedding_cfg.get("device")
        self.embedder = load_embedder(model_name, normalize_embeddings, device=device)

        retrieval_cfg = cfg.get("retrieval", {}) or {}
        self.alpha_dense = retrieval_cfg.get("alpha_dense", 0.6)
        self.mmr_lambda = retrieval_cfg.get("mmr_lambda", 0.5)
        self.mmr_candidates = retrieval_cfg.get("mmr_candidates", 20)
        self.mode = retrieval_cfg.get("mode", "hybrid")

    @log_call(level=logging.DEBUG, include_result=False)
    def _qdrant_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        q_vec = embed_texts(self.embedder, [query])[0]
        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=q_vec.tolist(),
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "collection" in msg and ("not found" in msg or "does not exist" in msg or "404" in msg):
                raise IndexNotReadyError(
                    f"Qdrant collection '{self.collection_name}' is missing. Please rebuild the index."
                ) from exc
            raise

        hits: List[Dict[str, Any]] = []
        for p in search_result:
            payload = p.payload or {}
            chunk_id = payload.get("chunk_id")
            if not chunk_id:
                continue
            vec = np.array(p.vector, dtype=float) if p.vector is not None else None
            hits.append({"chunk_id": chunk_id, "score": float(p.score), "payload": payload, "vector": vec})
        return hits

    @log_call(level=logging.DEBUG, include_result=False)
    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        raw_hits = self.bm25.search(query, top_k=limit)
        return [{"chunk_id": cid, "score": score} for cid, score in raw_hits]

    @staticmethod
    def _min_max_normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        mn = min(scores)
        mx = max(scores)
        if mx <= mn:
            return [0.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    @log_call(level=logging.INFO, include_result=False)
    def retrieve(self, query: str, top_k: int = 6, mmr_k: int = 12) -> List[Dict[str, Any]]:
        bm25_hits: List[Dict[str, Any]] = []
        qdrant_hits: List[Dict[str, Any]] = []

        if self.mode in ("hybrid", "sparse-only"):
            bm25_hits = self._bm25_search(query, limit=mmr_k)
        if self.mode in ("hybrid", "dense-only"):
            qdrant_hits = self._qdrant_search(query, limit=self.mmr_candidates)

        combined: Dict[str, Dict[str, Any]] = {}

        for h in bm25_hits:
            combined.setdefault(h["chunk_id"], {})["bm25_score"] = h["score"]

        for h in qdrant_hits:
            d = combined.setdefault(h["chunk_id"], {})
            d["dense_score"] = h["score"]
            d.setdefault("payload", h.get("payload"))
            d.setdefault("vector", h.get("vector"))

        if not combined:
            return []

        bm25_scores = [v.get("bm25_score", 0.0) for v in combined.values()]
        dense_scores = [v.get("dense_score", 0.0) for v in combined.values()]
        bm25_norm = self._min_max_normalize(bm25_scores)
        dense_norm = self._min_max_normalize(dense_scores)

        for (cid, v), b_n, d_n in zip(combined.items(), bm25_norm, dense_norm):
            if self.mode == "dense-only":
                v["final_score"] = d_n
            elif self.mode == "sparse-only":
                v["final_score"] = b_n
            else:
                v["final_score"] = self.alpha_dense * d_n + (1.0 - self.alpha_dense) * b_n

        chunk_ids = list(combined.keys())
        vectors: List[np.ndarray] = []
        for cid in chunk_ids:
            v = combined[cid]
            if v.get("vector") is not None:
                vectors.append(v["vector"])
            else:
                payload = v.get("payload", {}) or {}
                text = payload.get("text", "")
                vec = embed_texts(self.embedder, [text])[0]
                vectors.append(np.array(vec, dtype=float))

        doc_embeddings = np.stack(vectors, axis=0)
        query_vec = np.array(embed_texts(self.embedder, [query])[0], dtype=float)
        selected_indices = mmr(doc_embeddings, query_vec, top_k=min(mmr_k, len(chunk_ids)), lambda_mult=self.mmr_lambda)

        picked: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx in selected_indices:
            cid = chunk_ids[idx]
            picked.append((cid, float(combined[cid]["final_score"]), combined[cid]))

        picked.sort(key=lambda x: x[1], reverse=True)
        picked = picked[:top_k]

        results: List[Dict[str, Any]] = []
        for cid, score, v in picked:
            payload = v.get("payload", {}) or {}
            results.append({"chunk_id": cid, "text": payload.get("text", ""), "score": score, "metadata": payload})
        return results
