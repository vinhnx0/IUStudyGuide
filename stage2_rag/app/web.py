from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv
import yaml

from app.rag_pipeline import RAGPipeline


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    st.set_page_config(page_title="Stage2 RAG — Univ Q&A", layout="wide")
    st.title("🎓 Stage2 RAG — University Course Q&A")

    load_dotenv(find_dotenv())
    cfg = load_config()
    rag = RAGPipeline(cfg)

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-k", min_value=3, max_value=20, value=cfg["retrieval"]["top_k"])
        lang = st.selectbox("Language", ["auto", "vi", "en"], index=0)
        debug = st.checkbox("Show debug", value=False)
        if st.button("Reload indexes (re-ingest)"):
            rag.ingest(Path(cfg["paths"]["data_dir"]), rebuild=True)
            st.success("Re-ingest complete.")

    query = st.text_input("Ask a question", placeholder="Điều kiện tiên quyết của Giải tích 1?")
    if query:
        cfg["retrieval"]["top_k"] = top_k
        rag = RAGPipeline(cfg)
        resp = rag.ask(query, lang=lang, debug=debug)
        st.subheader("Answer")
        st.write(resp["answer"])

        if debug:
            st.subheader("Debug")
            st.json(resp["debug"])


if __name__ == "__main__":
    main()
