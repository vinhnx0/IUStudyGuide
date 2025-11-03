# The Two Minds of a Chatbot: Quick Response and Deep Reasoning for Academic Guidance

## Root `Thesis/README.md`

# Thesis Project

This repository contains the source code, data, and experiments for the thesis.

## 📂 Project Layout

```
Thesis/
    data/                # Input PDFs and reference documents
    stage1/              # Stage 1 (Baseline) system with Ollama
    stage2_rag/          # Stage 2 (RAG) system
    stage3_rag_kg/       # Stage 3 (RAG + KG) system
    .env                 # Environment configuration (per-project)
```

Note: virtual environments are created per-stage (inside each `stageX/` directory) — there is no longer a single shared `.venv/` at the repository root. See each stage's README for exact commands.

## 🚀 Getting Started

Each stage is self-contained and has its own `requirements.txt` and local `.venv` directory. This avoids dependency conflicts between stages and allows you to run stages independently.

See the stage READMEs for detailed setup and run instructions:

- `stage1/README.md`
- `stage2_rag/README.md`
- `stage3_rag_kg/README.md`

If you previously had a root `.venv` and want to remove it (recommended to avoid confusion), you can delete it before creating stage-specific venvs. On Windows PowerShell:

```powershell
# Stop Python processes if they are running (optional, but useful on Windows to avoid file locks):
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Delete the root virtualenv (be careful — this permanently removes the env):
Remove-Item -Recurse -Force .\.venv
```

After that, follow the README inside the stage you want to run to create and activate that stage's `.venv`.
