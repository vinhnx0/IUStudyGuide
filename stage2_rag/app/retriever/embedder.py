from __future__ import annotations

from typing import List, Tuple, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedder(model_name: str, normalize_embeddings: bool = True) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    model._normalize_embeddings = normalize_embeddings  # type: ignore[attr-defined]
    return model


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    arr = model.encode(list(texts), normalize_embeddings=getattr(model, "_normalize_embeddings", True))
    if isinstance(arr, list):
        arr = np.array(arr)
    return arr.astype("float32")
