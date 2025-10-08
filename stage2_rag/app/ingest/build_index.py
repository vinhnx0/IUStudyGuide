from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from tqdm import tqdm

from app.ingest.loaders import dispatch_load
from app.ingest.preprocess import preprocess_text
from app.retriever.embedder import load_embedder, embed_texts
from app.retriever.bm25 import BM25Index


class IndexBuilder:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.embedder = load_embedder(cfg["embedding"]["model_name"], cfg["embedding"]["normalize_embeddings"])

    def build(self, data_dir: Path, rebuild: bool = False) -> None:
        storage_dir = Path(self.paths["storage_dir"])
        storage_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = Path(self.paths["chunks_jsonl"])
        faiss_path = Path(self.paths["faiss_index"])
        bm25_path = Path(self.paths["bm25_index"])

        if rebuild:
            for p in [chunks_path, faiss_path, faiss_path.with_suffix(".npy"), bm25_path]:
                if Path(p).exists():
                    Path(p).unlink(missing_ok=True)

        # Load & preprocess
        docs = []
        for path in data_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            source_name = str(path.relative_to(data_dir))
            for (text, extra) in dispatch_load(path):
                docs.append((source_name, text, extra))

        # Chunk
        chunk_size = int(self.cfg["chunking"]["chunk_size_words"])
        overlap = int(self.cfg["chunking"]["chunk_overlap_words"])
        heading_patterns = self.cfg["chunking"]["heading_patterns"]
        all_chunks = []
        with open(chunks_path, "w", encoding="utf-8") as f:
            for source_name, text, extra in tqdm(docs, desc="Chunking"):
                page = extra.get("page")
                url = extra.get("url")
                items = preprocess_text(
                    text,
                    source_name=source_name,
                    heading_patterns=heading_patterns,
                    chunk_size_words=chunk_size,
                    chunk_overlap_words=overlap,
                    page=page,
                    url=url,
                    section_hint=source_name,
                )
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    all_chunks.append(it["text"])

        # Embeddings + FAISS
        if all_chunks:
            embs = embed_texts(self.embedder, all_chunks)
            faiss_idx = faiss.IndexFlatIP(embs.shape[1])
            faiss_idx.add(embs)
            faiss.write_index(faiss_idx, str(faiss_path))
            np.save(faiss_path.with_suffix(".npy"), embs)

        # BM25
        if self.cfg["retrieval"]["use_bm25"] and all_chunks:
            bm25 = BM25Index(bm25_path)
            bm25.build(all_chunks)
            bm25.save()
