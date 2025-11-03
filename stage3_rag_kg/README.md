# Stage 3 — Academic Assistant (RAG + KG + Query Decomposition + Routing)

This repository implements the **Stage 3** system that combines **RAG** (document retrieval) and **Knowledge Graph** reasoning to answer complex academic planning questions with **strict guardrails**.

## 📁 Structure

```
stage3_rag_kg/
  app/
    api.py                # FastAPI router
    inference.py          # main orchestrator
    rag_pipeline.py       # doc retrieval + grounding (TF-IDF)
    kg_pipeline.py        # graph reasoning (prereqs, credits, planning)
    query_router.py       # decomposition + routing logic
    guardrails.py         # input/output filtering
    prompts.py            # baseline prompt (reference)
    config.py
    logger.py
    __init__.py
  data/             >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    knowledge_graph.json  # small mock KG (courses + program)>>>>>>>>>>>>>>>>>>>>>>
    docs/
      sample_curriculum.txt
  eval/
    sample_queries.jsonl
    run_eval.py
  tests/
    test_guardrails.py
    test_router.py
    test_api.py
  requirements.txt
  Makefile
  README.md
```

## 🔐 Guardrails

- **Input guard** rejects queries containing sensitive or non-academic topics using exact and fuzzy matching (≥0.85 similarity).
- **Output guard** strips URLs and personal data and blocks sensitive topics.
- Language consistency: replies in the same language as the input (simple heuristic).

**Blocked categories** (non-exhaustive):
`sex, violence, religion, politics, drugs, suicide, hate, president, celebrity, weather, crime, relationship, salary`

If rejected, returns:
```json
{"error": "Query outside allowed academic domain."}
```

## 🧠 RAG

- Uses `scikit-learn` **TF‑IDF** + cosine similarity over text files in `data/docs/`.
- Top‑K retrieval with simple synthesis that quotes the most relevant snippet.

## 🕸️ Knowledge Graph (KG)

- Loads `data/knowledge_graph.json`.
- Supports:
  - **Eligibility**: can_take(taken, target) → missing prereqs
  - **Remaining**: credits & required courses left for a program
  - **Planning**: greedy per‑semester planner (max 5 courses/term by default)

## 🔀 Query Decomposition & Routing

- Splits multi‑clause prompts; classifies each sub‑query into **factual** (→ RAG), **relational** (→ KG), or **planning** (→ BOTH).

## ⚙️ How to run Stage 3

Create and use a stage-local virtual environment at `stage3_rag_kg/.venv`.

*Windows PowerShell*

```powershell
python -m venv .\stage3_rag_kg\.venv
.\stage3_rag_kg\.venv\Scripts\Activate.ps1
python -m pip install -r stage3_rag_kg/requirements.txt
```

*Linux / macOS*

```bash
python -m venv stage3_rag_kg/.venv && source stage3_rag_kg/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r stage3_rag_kg/requirements.txt
```

After dependencies are installed you can start the API:

```bash
make run  # or: uvicorn app.api:app --reload
```

## 🧪 Run tests and evaluation

```bash
make test
make eval
```

The eval harness executes `eval/sample_queries.jsonl` and writes outputs to `eval/results_stage3.jsonl` with a short summary printed to console.

## 🧾 Example Input/Output JSON

**Input**
```json
{
  "query": "Nếu tôi muốn rút ngắn lộ trình học còn 3 kỳ, nên học những môn nào mỗi kỳ?",
  "student_context": {
    "courses_taken": ["MATH101","CS101","PHYS101","ENG101","BAS101","BAS102"],
    "program_id": "CS_MAJOR",
    "semesters_left": 3
  }
}
```

**Output (example)**
```json
{
  "response": "[RAG] From documents (sample_curriculum.txt), relevant info: ...\n[KG] {'plan': [{'semester': 1, 'courses': ['MATH102', 'CS102', 'ELEC201', 'ELEC202', 'ELEC203']}, ...]}",
  "source": ["KG", "RAG"],
  "steps": ["Input validated by guardrails.", "Decomposed into 1 sub-queries and routed.", "RAG retrieval executed.", "KG reasoning executed."],
  "confidence": 0.75,
  "timestamp": "2025-10-10T00:00:00Z"
}
```

## 🚫 Example Guardrail Rejections

- `"Who is the president?"` → `{"error": "Query outside allowed academic domain."}`
- `"Tell me about politics in my degree plan"` → `{"error": "Query outside allowed academic domain."}`
- `"How to hack the exam grading system?"` → `{"error": "Query outside allowed academic domain."}`

## 🧩 Customization

- Add more documents under `data/docs/`.
- Extend programs/courses in `data/knowledge_graph.json`.
- Tune routing heuristics in `app/query_router.py`.
- Adjust guardrails in `app/guardrails.py`.
