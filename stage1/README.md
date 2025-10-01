# Stage 1 (Local LLM via Ollama)

Minimal, production-ready baseline that answers using **only model’s internal knowledge**. No RAG, no tools. Deterministic-ish via fixed params and `seed`.

## Prereqs

1. Install **Ollama**: https://ollama.com/download
2. Pull a chat-capable model (configurable):
   ```bash
   ollama pull llama3

Got it ✅ Below are two complete `README.md` files you can copy directly.

---

## `stage1/README.md`

# Stage 1 – Local LLM with Ollama

This project implements **Stage 1** of the thesis experiment:

- Run a **local LLM via Ollama** (e.g., `llama3`)
- **No retrieval / RAG / KG / tools**
- Provide a **deterministic baseline** for later A/B comparison



## 📦 Project Structure

```v

stage1/
   app/          # FastAPI + inference logic
   cli/          # Command-line interface
   eval/         # Evaluation harness
   tests/        # Smoke tests
   requirements.txt
   Makefile
   README.md     # This file

````


## ⚙️ Prerequisites

1. **Python**  
   - Recommended: **Python 3.10.x** (stable for all dependencies)  
   - Download: https://www.python.org/downloads/release/python-31014/  
   - ✅ During install: check **"Add Python to PATH"**

2. **Ollama**  
   - Download & install: https://ollama.com/download  
   - Verify install:  
     ```bash
     ollama --version
     ```

3. **Pull a model (example: llama3)**  
   ```bash
   ollama pull llama3
   ```

4. **Quick test Ollama**

   ```bash
   ollama run llama3 "Hello, how are you?"
   ```


## 🚀 Setup

From project root (`Thesis/`):

```bash
# Create venv
python -m venv .venv
```

### Activate venv

* **Windows PowerShell**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

  If you see a policy error:

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

* **Linux / macOS**

  ```bash
  source .venv/bin/activate
  ```

### Install dependencies

```bash
pip install -r stage1/requirements.txt
```



## 🖥️ Run API

### Start server

```bash
cd stage1
python -m uvicorn app.api:app --reload
```

Expected log:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Test API

Swagger docs:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Or curl:

```bash
curl -s http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain Newton’s second law.", "language":"en"}'
```


## 💻 Run CLI

```bash
python -m stage1.cli.ask --prompt "Who is the CEO of SpaceX?" --language en
```

* Prints model response
* Appends metadata JSON to `stage1/eval/cli_history.jsonl`


## 📊 Run Evaluation Harness

Run batch evaluation with provided sample prompts:

```bash
python -m stage1.eval.run_eval \
  --input stage1/eval/sample_prompts.jsonl \
  --model llama3 --temperature 0.2 --seed 42
```

Output:

* File: `stage1/eval/outputs_<timestamp>.jsonl`
* Console summary: number of prompts, avg latency, unknown counts


## 🧪 Run Tests

```bash
pytest -q stage1/tests
```


## 📝 Notes & Troubleshooting

* **`404 at /`** → Expected. Only `/v1/ask` is implemented. Use `/docs` for UI.
* **`ollama` not recognized** → Install from [Ollama downloads](https://ollama.com/download) and reopen shell.
* **Activation error on Windows** → Run once:

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
* **Determinism** → Fixed params: `temperature=0.2`, `seed=42`.
  If model uncertain → normalized to:
  **“I don’t know based on my current knowledge.”**


## ✅ Stage 1 Complete

You now have:

* Local inference-only API & CLI
* JSON logs for reproducibility
* Evaluation harness for Stage 2/3 comparison


