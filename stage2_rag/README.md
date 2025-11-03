=== FILE: stage2_rag/README.md ===

# Stage 2 — Basic RAG for University Course Q&A

> **Goal**: End-to-end RAG system (Python 3.10+) that answers questions about university curricula (course catalog, prerequisites, credits, VN/EN names), with CLI + REST API + optional simple Web UI.
> **Key focus**: robust course **alias normalization** (VN full/abbr/no-diacritics → canonical EN), consistent answers, citations, logs, quick eval.

---

## Architecture (Overview)

```
 ┌───────────────┐        ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
 │   Ingest      │  -->   │  Indexing     │  -->  │  Retrieval    │  -->  │  Generation   │
 │ (PDF/HTML/MD/ │        │ (FAISS + BM25)│       │  Hybrid+MMR   │       │  (LLM w/      │
 │  CSV)         │        │ + metadata    │       │ + alias map   │       │  citations)   │
 └───────────────┘        └───────────────┘       └───────────────┘       └───────────────┘
        ▲                         │                       │                       │
        │                         ▼                       ▼                       ▼
Preprocess (VN normalize;  chunk into 800-1200 words w/ overlap; section_path; save to storage/)
Alias Normalizer: auto + manual merge → aliases.json; exact → normalized → fuzzy → semantic

```

* **Hybrid retrieval**: Dense (Sentence Transformers) + **BM25** → union → **MMR re-rank** → prefer chunks containing resolved alias.
* **Citations**: source + page/url + section in every answer.
* **Two LLM backends**:

  * **Local** via Ollama (e.g., `llama3:8b`)
  * **OpenAI** (optional; `gpt-4o-mini`), key via `.env`

---

## Quickstart

```bash
# 0) Create a stage-local Python venv
# Recommended: create one `stage2_rag/.venv` per-stage to avoid conflicts
# On Windows (PowerShell):
python -m venv .\stage2_rag\.venv
.\stage2_rag\.venv\Scripts\Activate.ps1

# On macOS / Linux:
python -m venv stage2_rag/.venv && source stage2_rag/.venv/bin/activate

# 1) Install deps
cd stage2_rag
python -m pip install -r requirements.txt

# 2) Config & keys
# edit .env to set OPENAI_API_KEY (if using OpenAI), OLLAMA_HOST if needed
# edit config.yaml to adjust models/paths

# 3) Ingest (scrape-ready docs placed under data/; any subfolders are fine)
python -m app.cli ingest ..\data

# 6) Optional Simple Web UI (Streamlit)
cd stage2_rag
streamlit run app/web.py
```

---

## Data & Index

* **Vector store**: FAISS by default (`storage/faiss.index`), embeddings via `sentence-transformers/all-MiniLM-L6-v2` (configurable).
* **BM25**: rank_bm25 token-based index persisted to `storage/bm25.pkl`.
* **Chunks**: ~800–1200 words, 100–150 overlap, respect headings; metadata saved to `storage/chunks.jsonl`.
* **Aliases**: Auto-generated + manual merge into `storage/aliases.json` (updated after each ingest).

---

## Alias Normalization (Key Feature)

* Normalize VN names: **strip diacritics**, lowercase, condense spaces, **Roman ↔ Arabic** numerals, handle **course codes** (`MA101`, `MA-101`), remove department prefixes, fix punctuation.
* Matching pipeline:

  1. **Exact** match
  2. **Normalized exact**
  3. **Fuzzy** (RapidFuzz ratio ≥ 85 by default)
  4. **Semantic** (embedding cosine)
* Returns JSON like:

  ```json
  {"canonical":"Calculus I","matched":"Giai tich 1","method":"normalized_exact","confidence":0.97}
  ```

---

## Evaluation

```bash
python -m eval.run
```

* **Context hit-rate**: % queries whose retrieved contexts contain gold regex.
* **Alias resolution score**: % queries using VN/abbr names mapped to correct canonical EN.
* Report printed + saved to `eval/report.json`.

---

## Configuration

Edit `config.yaml`:

* Embedding model, chunk size/overlap, FAISS/Chroma, BM25 on/off, MMR params
* LLM backends (`ollama.model`, `openai.model`)
* Paths: data/, storage/

`.env`:

```
OPENAI_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434
HTTP_PROXY=
```

---

## Troubleshooting

* **PDF fonts** unreadable → try `pypdf` fallback or export to text; ensure `utf-8`.
* **Heading splits** odd → tweak `chunker.heading_patterns` in config.
* **Token/length errors** → reduce `chunk_size` in `config.yaml`.
* **Ollama not running** → install and `ollama serve`; set `OLLAMA_HOST`.
* **FAISS import error** (Apple Silicon): install `faiss-cpu` wheel; or switch to **Chroma** in config.
* **Slow retrieval** → disable BM25 or lower `k`; or pre-warm embeddings.

---

## Quality Bar (Acceptance)

* End-to-end run (ingest → ask) ✔
* With sample data, **context hit-rate ≥ 80%** ✔ (see `eval/`)
* Alias resolution active (logs show matched→canonical) ✔
* Citations included in all answers ✔
* `pytest -q` passes core tests ✔
* README lets a newcomer run within **<30 minutes** ✔

---

## Repo Hygiene

* Type hints, docstrings, `black`/`ruff`
* No hard-coded paths; use `config.yaml` + `.env`
* OSS only; paid keys in `.env`
* MIT License included

---

## Roadmap (Next Steps)

* Add Chroma/SQLite alternative store
* Add re-ranker (CrossEncoder) option
* Add small ontology expansion rules per department
* Add caching layers (embeddings; normalize; alias map)

---

## License

MIT — see `LICENSE`.

---

## Acknowledgments

* SentenceTransformers, FAISS, rank_bm25, FastAPI, Typer, Streamlit, RapidFuzz
