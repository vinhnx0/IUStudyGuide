# === stage3_ragkg/app/rag/indexing.py ===
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..logging_utils import get_logger, log_call
from .alias_normalizer import AliasNormalizer
from .bm25 import BM25Index
from .chunker import make_chunks
from .embedder import embed_texts, load_embedder
from .utils_text import normalize_vietnamese_text

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

_QDRANT_CLIENT: QdrantClient | None = None
_EMBED_DIM_CACHE: Dict[tuple, int] = {}  # (model_name, normalize, device) -> dim
logger = get_logger(__name__)


class IndexNotReadyError(RuntimeError):
    """Raised when index artifacts/collections are missing or empty."""


def _stable_hex_id(*parts: Any, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    for p in parts:
        if p is None:
            h.update(b"<none>")
        elif isinstance(p, (bytes, bytearray)):
            h.update(bytes(p))
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def _safe_qdrant_count(client: QdrantClient, collection_name: str) -> int:
    try:
        res = client.count(collection_name=collection_name, exact=True)
        return int(getattr(res, "count", 0) or 0)
    except Exception:
        return 0


def _load_kg_nodes_for_aliases(kg_json_path: Path) -> List[Dict[str, Any]]:
    """
    Load nodes from KG JSON in a schema-tolerant way.

    Supports:
      - {"nodes": [...]}  (your current KG)
      - {"data": {"nodes": [...]}}
      - or even a plain list of nodes
    """
    try:
        raw = json.loads(kg_json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        raise RuntimeError(f"Failed to read KG JSON at {kg_json_path}: {e}") from e

    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if isinstance(raw, dict):
        if isinstance(raw.get("nodes"), list):
            return [x for x in raw["nodes"] if isinstance(x, dict)]
        data = raw.get("data")
        if isinstance(data, dict) and isinstance(data.get("nodes"), list):
            return [x for x in data["nodes"] if isinstance(x, dict)]

    return []


def rebuild_aliases_json_from_kg(cfg: Dict[str, Any], aliases_path: Path) -> Dict[str, List[str]]:
    """
    Build aliases.json from KG nodes:
      { "IT135IU": ["IT135IU", "Khoa học dữ liệu", "Introduction to Data Science", ...], ... }

    This is intentionally simple and safe:
      - no fuzzy/semantic work here
      - only writes when we can extract course IDs from KG
    """
    paths = cfg.get("paths", {}) or {}
    kg_path = Path(paths.get("kg_json", "")) if paths.get("kg_json") else None
    if not kg_path or not kg_path.exists():
        raise RuntimeError(f"KG json not found. cfg.paths.kg_json={paths.get('kg_json')}")

    nodes = _load_kg_nodes_for_aliases(kg_path)
    if not nodes:
        raise RuntimeError(f"KG json loaded but no nodes found at {kg_path}")

    alias_map: Dict[str, List[str]] = {}

    for n in nodes:
        cid = n.get("id") or n.get("code") or n.get("course_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        cid = cid.strip()

        aliases: List[str] = [cid]

        # Common name fields: {"name": {"vi": "...", "en": "..."}} or {"name": "..."}
        name = n.get("name")
        if isinstance(name, dict):
            vi = name.get("vi")
            en = name.get("en")
            if isinstance(vi, str) and vi.strip():
                aliases.append(vi.strip())
            if isinstance(en, str) and en.strip():
                aliases.append(en.strip())
        elif isinstance(name, str) and name.strip():
            aliases.append(name.strip())

        # Optional: also include a "spaced" variant for code, e.g. IT135IU -> "IT 135 IU"
        # Keep it conservative to avoid noise.
        if len(cid) >= 6:
            # split letters+digits+letters tail (best-effort)
            import re

            m = re.match(r"^([A-Z]{2,4})(\d{2,3})([A-Z]{0,3})$", cid.upper())
            if m:
                a, b, c = m.groups()
                spaced = " ".join([x for x in [a, b, c] if x])
                if spaced and spaced != cid:
                    aliases.append(spaced)

        # dedupe while preserving order
        seen = set()
        uniq: List[str] = []
        for a in aliases:
            if a not in seen:
                seen.add(a)
                uniq.append(a)

        alias_map[cid] = uniq

    if not alias_map:
        raise RuntimeError(f"Built empty alias_map from KG at {kg_path}")

    aliases_path.parent.mkdir(parents=True, exist_ok=True)
    aliases_path.write_text(json.dumps(alias_map, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("rebuild_aliases_json_from_kg: wrote %s canon_ids=%d", aliases_path, len(alias_map))
    return alias_map


@log_call(level=logging.DEBUG, include_result=True)
def _get_embedding_dim(cfg: Dict[str, Any]) -> int:
    """
    Probe embedding dim ONCE and cache by (model_name, normalize, device).
    This avoids repeated dummy/probe embedding calls.
    """
    emb_cfg = cfg.get("embedding", {}) or {}
    model_name = emb_cfg.get("model_name")
    normalize_embeddings = emb_cfg.get("normalize_embeddings", True)
    device = emb_cfg.get("device")

    if not model_name:
        raise ValueError("config.embedding.model_name is missing")

    key = (str(model_name), bool(normalize_embeddings), str(device) if device is not None else None)
    cached = _EMBED_DIM_CACHE.get(key)
    if isinstance(cached, int) and cached > 0:
        return cached

    embedder = load_embedder(model_name, normalize_embeddings, device=device)
    probe = embed_texts(embedder, ["__dim_probe__"])
    if probe.shape[0] != 1 or probe.shape[1] <= 0:
        raise RuntimeError(f"Embedding dimension probe failed: shape={probe.shape}")
    dim = int(probe.shape[1])

    _EMBED_DIM_CACHE[key] = dim
    logger.info("Embedding dim probe (cached): model=%s dim=%d device=%s", model_name, dim, device)
    return dim


@log_call(level=logging.INFO, include_result=False)
def load_documents(data_dir: Path) -> List[Dict[str, Any]]:
    logger.info("load_documents: scanning data_dir=%s", data_dir)
    docs: List[Dict[str, Any]] = []
    total_files = 0
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        ext = path.suffix.lower()
        if ext not in {".pdf", ".txt", ".md"}:
            continue

        if ext == ".pdf":
            try:
                reader = pypdf.PdfReader(str(path))
            except Exception:
                logger.exception("load_documents: failed to read PDF %s", path)
                continue
            pages_text: List[str] = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            raw_text = "\n".join(pages_text)
            meta = {"source": path.name, "path": str(path), "pages": len(reader.pages)}
        else:
            try:
                raw_text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                logger.exception("load_documents: failed to read text file %s", path)
                continue
            meta = {"source": path.name, "path": str(path)}

        clean_text = normalize_vietnamese_text(raw_text)
        if not clean_text.strip():
            continue

        docs.append({"id": path.stem, "text": clean_text, "metadata": meta})

    logger.info("load_documents: scanned_files=%d loaded_docs=%d", total_files, len(docs))
    return docs


@log_call(level=logging.INFO, include_result=False)
def make_chunks_for_docs(
    docs: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    method: str = "word",
) -> List[Dict[str, Any]]:
    logger.info(
        "make_chunks_for_docs: docs=%d chunk_size=%d chunk_overlap=%d method=%s",
        len(docs),
        chunk_size,
        chunk_overlap,
        method,
    )
    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        pieces = make_chunks(text=doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap, method=method)
        for idx, txt in enumerate(pieces):
            if not txt.strip():
                continue
            chunk_id = f"{doc['id']}#chunk_{idx}"
            meta = {**doc["metadata"], "doc_id": doc["id"], "chunk_index": idx}
            chunks.append({"id": chunk_id, "text": txt, "metadata": meta})
    logger.info("make_chunks_for_docs: produced %d chunks", len(chunks))
    return chunks


@log_call(level=logging.DEBUG, include_result=False)
def enrich_chunks_with_aliases(chunks: List[Dict[str, Any]], alias_normalizer: AliasNormalizer) -> None:
    import re

    course_id_re = re.compile(r"[A-Z]{2,}\d{2,}")
    hit = 0
    for ch in chunks:
        text_upper = ch["text"].upper()
        raw_ids = set(course_id_re.findall(text_upper))
        canonical_ids: List[str] = []
        for rid in raw_ids:
            canonical = alias_normalizer.normalize(rid)
            if canonical and canonical not in canonical_ids:
                canonical_ids.append(canonical)
        if canonical_ids:
            ch["metadata"]["course_ids"] = canonical_ids
            hit += 1
    logger.info("enrich_chunks_with_aliases: chunks=%d with_course_ids=%d", len(chunks), hit)


def generate_hypothetical_questions_for_chunk(text: str, max_questions: int) -> List[str]:
    return []


@log_call(level=logging.INFO, include_result=False)
def enrich_chunks_with_hypothetical_questions(chunks: List[Dict[str, Any]], max_questions: int, enable: bool) -> None:
    if not enable or max_questions <= 0:
        logger.info("hypothetical_questions: disabled enable=%s max=%d", enable, max_questions)
        return
    total = 0
    for ch in chunks:
        questions = generate_hypothetical_questions_for_chunk(ch["text"], max_questions=max_questions)
        questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if questions:
            ch["metadata"]["questions"] = questions
            total += len(questions)
    logger.info("hypothetical_questions: chunks=%d total_questions=%d", len(chunks), total)


@log_call(level=logging.INFO, include_result=False)
def save_chunks_jsonl(chunks: List[Dict[str, Any]], path: Path) -> None:
    logger.info("save_chunks_jsonl: path=%s chunks=%d", path, len(chunks))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"id": ch["id"], "text": ch["text"], "metadata": ch["metadata"]}, ensure_ascii=False) + "\n")


@log_call(level=logging.INFO, include_result=False)
def build_bm25_index(chunks: List[Dict[str, Any]], bm25_path: Path) -> BM25Index:
    logger.info("build_bm25_index: chunks=%d bm25_path=%s", len(chunks), bm25_path)
    texts = [ch["text"] for ch in chunks]
    ids = [ch["id"] for ch in chunks]
    index = BM25Index(bm25_path)
    index.build(texts, doc_ids=ids)
    index.save()
    return index


@log_call(level=logging.DEBUG, include_result=False)
def _make_qdrant_client(cfg: Dict[str, Any]) -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    qcfg = cfg.get("qdrant", {}) or {}
    mode = str(qcfg.get("mode", "server")).lower()
    logger.info("Qdrant: initializing client mode=%s", mode)

    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(k, None)

    if mode == "embedded":
        location = cfg.get("paths", {}).get("qdrant_local_path", "storage/qdrant_embedded")
        Path(location).mkdir(parents=True, exist_ok=True)
        _QDRANT_CLIENT = QdrantClient(path=location)
        return _QDRANT_CLIENT

    try:
        host = qcfg.get("host", "localhost")
        port = qcfg.get("port", 6333)
        prefer_grpc = qcfg.get("prefer_grpc", False)
        _QDRANT_CLIENT = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc, timeout=2.0)
        return _QDRANT_CLIENT
    except Exception:
        logger.exception("Qdrant server connect failed; fallback to embedded")
        location = cfg.get("paths", {}).get("qdrant_local_path", "storage/qdrant_embedded")
        Path(location).mkdir(parents=True, exist_ok=True)
        _QDRANT_CLIENT = QdrantClient(path=location)
        return _QDRANT_CLIENT


@log_call(level=logging.INFO, include_result=False)
def ensure_collection_exists(
    client: QdrantClient,
    cfg: Dict[str, Any],
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    collection_name = cfg.get("paths", {}).get("qdrant_collection")
    if not collection_name:
        raise ValueError("config.paths.qdrant_collection is missing")

    if client.collection_exists(collection_name):
        logger.info("ensure_collection_exists: exists name=%s", collection_name)
        return

    logger.info("ensure_collection_exists: creating name=%s dim=%d distance=%s", collection_name, vector_size, distance)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )


@log_call(level=logging.INFO, include_result=True)
def ensure_vector_store_ready(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure vector store artifacts exist:
      - Qdrant collection exists
      - BM25 index exists
      - aliases.json exists (NEW: rebuild from KG if missing)
    Also caches embedding dim to avoid duplicate probes.
    """
    paths = cfg.get("paths", {}) or {}
    Path(paths.get("storage_dir", "storage")).mkdir(parents=True, exist_ok=True)

    # --- NEW: ensure aliases.json exists (lightweight, no re-embed) ---
    aliases_path = Path(paths.get("aliases_json", "storage/aliases.json"))
    aliases_ok = aliases_path.exists()
    if not aliases_ok:
        try:
            rebuild_aliases_json_from_kg(cfg, aliases_path)
            aliases_ok = aliases_path.exists()
        except Exception:
            logger.exception("ensure_vector_store_ready: failed to rebuild aliases_json at %s", aliases_path)

    client = _make_qdrant_client(cfg)

    # --- embedding dim probe (cached) ---
    embed_dim = _get_embedding_dim(cfg)
    ensure_collection_exists(client, cfg, vector_size=embed_dim)

    collection_name = paths.get("qdrant_collection")
    qdrant_points = _safe_qdrant_count(client, str(collection_name)) if collection_name else 0

    bm25_path = Path(paths.get("bm25_index", "storage/bm25.pkl"))
    bm25_ok = bm25_path.exists()

    needs_reindex = (qdrant_points <= 0) or (not bm25_ok)

    status = {
        "qdrant_ok": bool(collection_name) and client.collection_exists(str(collection_name)),
        "qdrant_points": int(qdrant_points),
        "bm25_ok": bool(bm25_ok),
        "aliases_ok": bool(aliases_ok),
        "needs_reindex": bool(needs_reindex),
        "collection_name": str(collection_name or ""),
        "vector_dim": int(embed_dim),
        "bm25_path": str(bm25_path),
        "aliases_path": str(aliases_path),
    }

    logger.info(
        "ensure_vector_store_ready: collection=%s points=%d bm25_ok=%s aliases_ok=%s needs_reindex=%s",
        status["collection_name"],
        status["qdrant_points"],
        status["bm25_ok"],
        status["aliases_ok"],
        status["needs_reindex"],
    )
    return status


@log_call(level=logging.INFO, include_result=False)
def _maybe_delete_collection(client: QdrantClient, collection_name: str) -> None:
    if client.collection_exists(collection_name):
        logger.info("Deleting Qdrant collection=%s", collection_name)
        client.delete_collection(collection_name=collection_name)


@log_call(level=logging.INFO, include_result=True)
def reindex_all(cfg: Dict[str, Any], force_recreate: bool = False) -> Dict[str, Any]:
    paths = cfg.get("paths", {}) or {}
    data_dir = Path(paths.get("data_dir", "../data"))
    chunks_path = Path(paths.get("chunks_jsonl", "storage/chunks.jsonl"))
    bm25_path = Path(paths.get("bm25_index", "storage/bm25.pkl"))
    collection_name = str(paths.get("qdrant_collection", ""))

    client = _make_qdrant_client(cfg)
    embed_dim = _get_embedding_dim(cfg)  # cached

    if not collection_name:
        raise ValueError("config.paths.qdrant_collection is missing")

    if force_recreate:
        _maybe_delete_collection(client, collection_name)
        ensure_collection_exists(client, cfg, vector_size=embed_dim)
        for p in [chunks_path, bm25_path]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                logger.exception("Failed to delete artifact %s", p)

    docs = load_documents(data_dir)

    chunk_cfg = cfg.get("chunking", {}) or {}
    chunks = make_chunks_for_docs(
        docs,
        chunk_size=int(chunk_cfg.get("chunk_size", 1000)),
        chunk_overlap=int(chunk_cfg.get("chunk_overlap", 120)),
        method=str(chunk_cfg.get("method", "word")),
    )

    alias_normalizer = AliasNormalizer(cfg)
    enrich_chunks_with_aliases(chunks, alias_normalizer)

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    enrich_chunks_with_hypothetical_questions(
        chunks,
        max_questions=int(retrieval_cfg.get("max_hypothetical_questions_per_chunk", 0)),
        enable=bool(retrieval_cfg.get("enable_hypothetical_questions", False)),
    )

    save_chunks_jsonl(chunks, chunks_path)
    build_bm25_index(chunks, bm25_path)

    qdrant_before = _safe_qdrant_count(client, collection_name)

    # pass cached dim so build_qdrant doesn't probe again
    build_qdrant_index_for_chunks_and_questions(cfg, chunks, force_recreate=False, vector_size=embed_dim)

    qdrant_after = _safe_qdrant_count(client, collection_name)

    return {
        "docs": int(len(docs)),
        "chunks": int(len(chunks)),
        "bm25_built": True,
        "bm25_path": str(bm25_path),
        "qdrant_collection": collection_name,
        "qdrant_points_before": int(qdrant_before),
        "qdrant_points_after": int(qdrant_after),
        "vector_dim": int(embed_dim),
        "force_recreate": bool(force_recreate),
    }


@log_call(level=logging.INFO, include_result=False)
def build_qdrant_index_for_chunks_and_questions(
    cfg: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    force_recreate: bool = False,
    vector_size: Optional[int] = None,  # NEW: allow caller to pass cached dim
) -> None:
    collection_name = cfg.get("paths", {}).get("qdrant_collection")
    if not collection_name:
        raise ValueError("config.paths.qdrant_collection is missing")

    client = _make_qdrant_client(cfg)

    embedding_cfg = cfg.get("embedding", {}) or {}
    model_name = embedding_cfg.get("model_name")
    normalize_embeddings = embedding_cfg.get("normalize_embeddings", True)
    device = embedding_cfg.get("device")
    batch_size = int(embedding_cfg.get("batch_size", 64))

    embedder = load_embedder(model_name, normalize_embeddings, device=device)

    # --- NEW: do NOT probe with "dummy" again; reuse cached dim ---
    if vector_size is None:
        vector_size = _get_embedding_dim(cfg)

    if force_recreate:
        _maybe_delete_collection(client, str(collection_name))

    ensure_collection_exists(client, cfg, vector_size=int(vector_size))

    texts_to_embed: List[str] = []
    payloads: List[Dict[str, Any]] = []
    point_ids: List[str] = []

    for ch in chunks:
        texts_to_embed.append(ch["text"])
        payloads.append({**ch["metadata"], "chunk_id": ch["id"], "text": ch["text"]})
        point_ids.append(_stable_hex_id("chunk", ch["id"]))

        questions = (ch.get("metadata") or {}).get("questions", [])
        for qi, q in enumerate(questions or []):
            texts_to_embed.append(q)
            payloads.append({**ch["metadata"], "chunk_id": ch["id"], "synthetic_question": q, "question_index": qi})
            point_ids.append(_stable_hex_id("question", ch["id"], qi, q))

    if not texts_to_embed:
        logger.warning("build_qdrant_index_for_chunks_and_questions: no texts to embed")
        return

    all_vectors: List[List[float]] = []
    for start in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[start : start + batch_size]
        vecs = embed_texts(embedder, batch)
        all_vectors.extend([v.tolist() for v in vecs])

    points: List[qmodels.PointStruct] = []
    for pid, vec, payload in zip(point_ids, all_vectors, payloads):
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))

    logger.info("Qdrant upsert: collection=%s points=%d dim=%d", collection_name, len(points), int(vector_size))
    client.upsert(collection_name=str(collection_name), wait=True, points=points)
