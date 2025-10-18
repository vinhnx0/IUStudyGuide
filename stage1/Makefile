# Makefile
.PHONY: setup run api test pull

VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

setup:
	python3 -m venv $(VENV); \
	. $(VENV)/bin/activate; \
	$(PIP) install -r requirements.txt

pull:
	@echo "Ensure Ollama is installed: https://ollama.com/download"
	ollama pull qwen3:8b

api:
	$(PY) -m uvicorn app.api:app --reload

run:
	$(PY) -m cli.ask --prompt "Hello" --language en

test:
	$(PY) -m pytest -q
