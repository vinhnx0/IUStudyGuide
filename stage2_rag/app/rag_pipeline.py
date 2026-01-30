from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO)
import time
from typing import Dict, Any, Optional, Tuple, List

from app.retriever.hybrid import HybridRetriever
from app.retriever.alias_normalizer import AliasNormalizer
from app.ingest.build_index import IndexBuilder

from app.llm import local_generate

logger = logging.getLogger(__name__)

from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "app\prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")

def _extract_text_source(item: Any) -> Tuple[str, str]:
    """
    Make Stage 2 resilient to different retriever output shapes.

    Supported shapes:
    - dict: {"text": "...", "source": "..."} (plus anything)
    - tuple/list: (text, meta) or (text, score, meta) or (score, text, meta)
      where meta may be dict containing "source"
    - anything else: fallback to str(item)
    """
    text = ""
    source = "unknown"

    # dict case
    if isinstance(item, dict):
        text = str(item.get("text", "") or "")
        source = str(item.get("source", item.get("path", item.get("doc_id", "unknown"))) or "unknown")
        return text, source

    # tuple/list case
    if isinstance(item, (list, tuple)):
        parts = list(item)

        meta = None
        # meta often is the last element if it's a dict
        if parts and isinstance(parts[-1], dict):
            meta = parts[-1]

        # try to find a string-like text among elements
        candidates = [p for p in parts if isinstance(p, str)]
        if candidates:
            # choose the longest string as "text" (often chunk content)
            text = max(candidates, key=len)
        else:
            # sometimes text could be under meta
            if isinstance(meta, dict):
                text = str(meta.get("text", "") or "")

        if isinstance(meta, dict):
            source = str(meta.get("source", meta.get("path", meta.get("doc_id", "unknown"))) or "unknown")

        return text, source

    # fallback
    return str(item), "unknown"


def pack_evidence(chunks, cfg: dict) -> str:
    """
    Pack evidence into a bounded string (char-budget based).
    Works with chunks being list[dict] or list[tuple/list] from legacy retrievers.
    """
    syn_cfg = cfg.get("synthesis", {})
    max_total_chars = int(syn_cfg.get("max_evidence_chars", 9000))
    max_chunk_chars = int(syn_cfg.get("max_chunk_chars", 1200))

    parts = []
    total = 0
    idx = 0

    for ch in chunks or []:
        text, source = _extract_text_source(ch)
        text = (text or "").strip()
        if not text:
            continue

        text = text[:max_chunk_chars]

        idx += 1
        block = f"[{idx}] SOURCE: {source}\nTEXT: {text}\n"
        if total + len(block) > max_total_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)


class RAGPipeline:
    """
    Stage 2 RAG pipeline (basic):
    - ingest: IndexBuilder.build(data_dir)  -> chunks.jsonl + faiss + (optional) bm25
    - retrieve: HybridRetriever (existing)
    - synthesize: LOCAL LLM (LM Studio) with prompt loaded from file
    - single-turn only
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.alias = AliasNormalizer(cfg)
        self.retriever = HybridRetriever(cfg)

        # Prompt file location: app/prompts/stage2_answer_prompt.txt
        self.answer_prompt = load_prompt("stage2_answer_prompt.txt")

    # ---------------- Ingest ---------------- #
    def ingest(self, data_path: Path, rebuild: bool = False) -> None:
        """
        Build / rebuild indexes from documents in data_path.
        This matches the existing connection in app/web.py:
            rag.ingest(Path(cfg["paths"]["data_dir"]), rebuild=True)

        After building, refresh retriever + alias so Stage 2 can answer immediately.
        """
        builder = IndexBuilder(self.cfg)
        builder.build(data_path, rebuild=rebuild)

        # refresh runtime resources
        self.retriever = HybridRetriever(self.cfg)
        self.alias = AliasNormalizer(self.cfg)

        logger.info("stage2_ingest_done rebuild=%s data_path=%s", rebuild, str(data_path))

    # ---------------- Ask ---------------- #
    def _build_queries(self, question: str) -> List[str]:
        """
        Minimal query expansion:
        - always include original question
        - if AliasNormalizer resolves a canonical term with good confidence, add it
        """
        queries = [question]
        try:
            hit = self.alias.resolve(question) or {}
            canonical = hit.get("canonical")
            conf = float(hit.get("confidence", 0.0) or 0.0)
            if canonical and canonical not in queries and conf >= 0.80:
                queries.append(canonical)
        except Exception:
            logger.exception("AliasNormalizer.resolve failed; using original question only")
        return queries

    def _to_citations(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep the output shape expected by app/web.py (it prints each citation item).
        """
        cits: List[Dict[str, Any]] = []
        for d in docs[:4]:
            m = d.get("metadata", {}) or {}
            cits.append(
                {
                    "source": m.get("source"),
                    "page": m.get("page"),
                    "url": m.get("url"),
                    "section": m.get("section_path"),
                }
            )
        return cits

    def answer(self, question: str) -> Dict[str, Any]:
        queries = self._build_queries(question)
        docs, dbg_retrieval = self.retriever.retrieve(queries)

        evidence = pack_evidence(docs, self.cfg)  # robust packer supports dict + metadata
        prompt = self.answer_prompt.format(question=question, evidence=evidence)
        text = local_generate(prompt, self.cfg)

        return {
            "answer": text,
            "citations": self._to_citations(docs),
            "aliases": self.alias.resolve(question),
            "_dbg_retrieval": dbg_retrieval,  # internal; only exposed when debug=True
            "_expanded_queries": queries,
        }

    def ask(self, query: str, lang: str = "auto", debug: bool = False) -> Dict[str, Any]:
        """
        Backward-compatible API for app/web.py.
        Also logs end-to-end runtime from receiving question -> producing answer.
        """
        t0 = time.perf_counter()
        try:
            resp = self.answer(query)

            # remove internal keys unless debug
            if not debug:
                resp.pop("_dbg_retrieval", None)
                resp.pop("_expanded_queries", None)
                return resp

            resp["debug"] = {
                "lang": lang,
                "expanded_queries": resp.pop("_expanded_queries", []),
                "retrieval": resp.pop("_dbg_retrieval", {}),
            }
            return resp
        finally:
            t1 = time.perf_counter()
            logger.info("stage2_runtime_seconds=%.3f query_len=%d", (t1 - t0), len(query or ""))