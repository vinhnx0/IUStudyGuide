from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.retriever.embedder import load_embedder, embed_texts
from app.retriever.bm25 import BM25Index


def mmr(doc_embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int, lambda_mult: float) -> List[int]:
    # Maximal Marginal Relevance (simple implementation)
    selected: List[int] = []
    candidates = list(range(len(doc_embeddings)))
    if len(candidates) == 0:
        return []
    sim_to_query = cosine_similarity(doc_embeddings, query_embedding.reshape(1, -1)).flatten()

    while len(selected) < min(top_k, len(candidates)):
        if not selected:
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            candidates.remove(idx)
            continue
        # compute diversity term
        selected_embs = doc_embeddings[selected]
        sim_to_selected = cosine_similarity(doc_embeddings, selected_embs).max(axis=1)
        # MMR score
        mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
        # do not reselect already selected
        mmr_score[selected] = -1e9
        idx = int(np.argmax(mmr_score))
        selected.append(idx)
    return selected


class HybridRetriever:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.top_k = cfg["retrieval"]["top_k"]
        self.use_bm25 = cfg["retrieval"]["use_bm25"]
        self.mmr_lambda = cfg["retrieval"]["mmr_lambda"]
        self.mmr_candidates = cfg["retrieval"]["mmr_candidates"]

        # load chunks & metadata
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        chunks_file = Path(self.paths["chunks_jsonl"])
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.chunks.append(obj["text"])
                    self.metadata.append(obj["metadata"])

        # load embeddings / faiss
        self.faiss_path = Path(self.paths["faiss_index"])
        self.embed_model_name = cfg["embedding"]["model_name"]
        self.normalize_embeddings = bool(cfg["embedding"]["normalize_embeddings"])
        self.embedder = load_embedder(self.embed_model_name, self.normalize_embeddings)
        self.index = None
        self.embeddings = None
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            # attempt to load embeddings matrix (store sidecar if present)
            emb_sidecar = self.faiss_path.with_suffix(".npy")
            if emb_sidecar.exists():
                self.embeddings = np.load(emb_sidecar).astype("float32")

        # BM25
        self.bm25 = None
        if self.use_bm25:
            self.bm25 = BM25Index(Path(self.paths["bm25_index"]))
            if not self.bm25.load() and self.chunks:
                self.bm25.build(self.chunks)
                self.bm25.save()

    def _dense_search(self, queries: List[str], top_k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        q_emb = embed_texts(self.embedder, queries)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        # merge result of multiple queries
        scores = np.zeros(len(self.chunks), dtype="float32")
        for qe in q_emb:
            D, I = self.index.search(qe.reshape(1, -1), min(self.mmr_candidates, len(self.chunks)))
            for i, d in zip(I[0], D[0]):
                if i >= 0:
                    scores[i] = max(scores[i], 1.0 / (1.0 + d))  # convert distance to similarity-ish
        ranked = [(int(i), float(s)) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[: top_k * 2]

    def _bm25_search(self, queries: List[str], top_k: int) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        scores = {}
        for q in queries:
            for idx, s in self.bm25.search(q, top_k=top_k * 2):
                scores[idx] = max(scores.get(idx, 0.0), float(s))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: top_k * 2]

    def retrieve(self, queries: List[str]) -> tuple[list[dict], dict]:
        dense = self._dense_search(queries, self.top_k)
        bm25 = self._bm25_search(queries, self.top_k) if self.use_bm25 else []

        # union
        indices = list({i for i, _ in dense} | {i for i, _ in bm25})
        if not indices:
            return [], {"dense": dense, "bm25": bm25, "mmr_selected": []}

        # MMR re-rank using stored embeddings (if present); otherwise use embed on the fly
        if self.embeddings is None:
            self.embeddings = embed_texts(self.embedder, self.chunks)
            np.save(self.faiss_path.with_suffix(".npy"), self.embeddings)

        doc_emb = self.embeddings[indices]
        q_emb = embed_texts(self.embedder, [" ".join(queries)])[0]
        selected_local = mmr(doc_emb, q_emb, top_k=self.top_k, lambda_mult=float(self.mmr_lambda))
        selected_global = [indices[i] for i in selected_local]

        docs = []
        for idx in selected_global:
            docs.append({"text": self.chunks[idx], "metadata": self.metadata[idx]})

        dbg = {"dense": dense, "bm25": bm25, "mmr_selected": selected_global}
        return docs, dbg
