from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, models

from app.logging_utils import get_logger, log_call

logger = get_logger(__name__)


def _safe_transformer_model_args() -> Dict[str, Any]:
    """
    Try to prevent Transformers/Accelerate from initializing weights on the 'meta' device.
    These args are passed to transformers.AutoModel.from_pretrained via Sentence-Transformers'
    lower-level modules API (models.Transformer), which is compatible with older ST versions.
    """
    return {
        "device_map": None,
        "low_cpu_mem_usage": False,
        # keep conservative defaults
        "trust_remote_code": False,
    }


def _move_to_device_safely(model: SentenceTransformer, device: str) -> SentenceTransformer:
    """
    Move a model to device, but if parameters are on 'meta', use to_empty() when available.
    """
    try:
        model.to(device)  # type: ignore[attr-defined]
        return model
    except NotImplementedError as exc:
        msg = str(exc).lower()
        if "meta tensor" not in msg and "to_empty" not in msg:
            raise

    # Try to_empty if available (newer torch)
    try:
        to_empty = getattr(model, "to_empty", None)
        if callable(to_empty):
            to_empty(device)  # type: ignore[misc]
            return model
    except Exception:
        pass

    raise NotImplementedError(
        "SentenceTransformer parameters are on 'meta' device and cannot be moved with .to(). "
        "Try upgrading/downgrading torch/transformers/accelerate/sentence-transformers, or "
        "use the manual ST modules builder path (already attempted)."
    )


def _build_sentence_transformer_via_modules(
    model_name: str,
    *,
    device: Optional[str],
    normalize_embeddings: bool,
) -> SentenceTransformer:
    """
    Build a SentenceTransformer pipeline using sentence_transformers.models.*
    This path works on older sentence-transformers versions that do NOT support
    SentenceTransformer(..., model_kwargs=...).
    """
    model_args = _safe_transformer_model_args()

    # NOTE: models.Transformer passes model_args into AutoModel.from_pretrained.
    transformer = models.Transformer(model_name, model_args=model_args)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    st_model = SentenceTransformer(modules=[transformer, pooling])

    # Move after construction (avoid ST __init__ forcing .to() too early)
    target_device = device or "cpu"
    st_model = _move_to_device_safely(st_model, target_device)

    setattr(st_model, "_normalize_embeddings", bool(normalize_embeddings))
    return st_model


@log_call(level=logging.INFO, include_result=False)
def load_embedder(
    model_name: str,
    normalize_embeddings: bool = True,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """
    Load a SentenceTransformer model robustly across ST versions.

    Fixes two classes of issues:
    1) Older sentence-transformers: SentenceTransformer.__init__ does NOT accept model_kwargs
       -> we fall back to building via sentence_transformers.models.Transformer(...)
    2) Newer torch/transformers combos may load on 'meta' then crash on .to(device)
       -> we disable meta-loading via model_args and use safe device moves.
    """
    if not model_name:
        raise ValueError("model_name must be provided")

    logger.info(
        "load_embedder: model_name=%s normalize=%s device=%s",
        model_name,
        normalize_embeddings,
        device,
    )

    # ---- Attempt 1: If this ST version supports model_kwargs, use it (best control) ----
    model_args = _safe_transformer_model_args()
    try:
        # Some ST versions support model_kwargs; others will raise TypeError.
        if device:
            st_model = SentenceTransformer(model_name, device=device, model_kwargs=model_args)  # type: ignore[arg-type]
        else:
            st_model = SentenceTransformer(model_name, model_kwargs=model_args)  # type: ignore[arg-type]

        # Tag normalization preference
        setattr(st_model, "_normalize_embeddings", bool(normalize_embeddings))
        return st_model

    except TypeError as exc:
        # model_kwargs not supported in this ST version -> fallback to modules path
        logger.warning(
            "SentenceTransformer(model_kwargs=...) not supported (%s). Falling back to modules builder.",
            exc,
        )

    except NotImplementedError as exc:
        # meta-tensor move error on some stacks -> fallback to modules path
        msg = str(exc).lower()
        if "meta tensor" in msg or "to_empty" in msg:
            logger.warning(
                "Meta-tensor error during SentenceTransformer init/move (%s). Falling back to modules builder.",
                exc,
            )
        else:
            raise

    # ---- Attempt 2: Build via sentence_transformers.models (compatible with older ST) ----
    try:
        return _build_sentence_transformer_via_modules(
            model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception:
        logger.exception("load_embedder: failed to load embedder via modules path")
        raise


@log_call(level=logging.DEBUG, include_result=False)
def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed a list of texts into a 2D numpy array."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    normalize = bool(getattr(model, "_normalize_embeddings", True))
    vecs = model.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if isinstance(vecs, np.ndarray):
        return vecs.astype(np.float32, copy=False)
    return np.asarray(vecs, dtype=np.float32)
