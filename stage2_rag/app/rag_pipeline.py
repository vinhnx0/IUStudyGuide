from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import yaml

from app.retriever.embedder import load_embedder, embed_texts
from app.retriever.hybrid import HybridRetriever
from app.retriever.alias_normalizer import AliasNormalizer
from app.ingest.build_index import IndexBuilder
from app.llm.gemini_backend import GeminiLLM
from app.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class RAGPipeline:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.storage_dir = Path(cfg["paths"]["storage_dir"])
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("RAGPipeline")
        logging.basicConfig(level=getattr(logging, cfg["logging"]["level"]))
        self.alias = AliasNormalizer(cfg)
        self.retriever = HybridRetriever(cfg)

        # LLMs
        self.backend = cfg["llm"]["default_backend"]
        self.gemini = GeminiLLM(cfg)

    # ---------------- Ingest ---------------- #
    def ingest(self, data_path: Path, rebuild: bool = False) -> None:
        builder = IndexBuilder(self.cfg)
        builder.build(data_path, rebuild=rebuild)
        # Refresh retriever resources
        self.retriever = HybridRetriever(self.cfg)
        # Refresh alias from storage
        self.alias = AliasNormalizer(self.cfg)

    # ---------------- Ask ---------------- #
    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        blocks = []
        for d in docs:
            meta = d.get("metadata", {})
            header = (
                f"[Source: {meta.get('source')} | section: {meta.get('section_path')} | page: {meta.get('page')}"
                + (f" | title: {meta.get('title')}" if meta.get('title') else "")
                + (f" | keywords: {', '.join(meta.get('keywords', []))}" if meta.get('keywords') else "")
                + "]"
            )
            blocks.append(header + "\n" + d["text"])
        return "\n\n---\n\n".join(blocks)

    def _to_citations(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cits = []
        for d in docs[:4]:
            m = d.get("metadata", {})
            cits.append(
                {
                    "source": m.get("source"),
                    "page": m.get("page"),
                    "url": m.get("url"),
                    "section": m.get("section_path"),
                }
            )
        return cits

    def _choose_backend(self):
        be = self.cfg["llm"]["default_backend"]
        if be == "gemini":
            return self.gemini
        return self.gemini


    def ask(self, query: str, lang: str = "auto", debug: bool = False) -> Dict[str, Any]:
        # Alias normalize attempt (extract likely course mentions)
        alias_info = self.alias.resolve(query)

        # Query expansion via alias/ontology (minimal stub)
        expanded_queries = [query]
        if alias_info and alias_info.get("canonical"):
            canonical = alias_info["canonical"]
            expanded_queries.append(canonical)

        docs, dbg_retrieval = self.retriever.retrieve(expanded_queries)

        ctx = self._format_context(docs)
        prompt = SYSTEM_PROMPT + "\n" + USER_PROMPT_TEMPLATE.format(ctx=ctx, query=query)

        llm = self._choose_backend()
        answer = llm.generate(prompt)

        result = {
            "answer": answer,
            "citations": self._to_citations(docs),
            "aliases": alias_info,
        }
        if debug:
            result["debug"] = {"retrieval": dbg_retrieval, "expanded_queries": expanded_queries}
        return result
