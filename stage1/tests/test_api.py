# test_api.py
from fastapi.testclient import TestClient
from app.api import app
from app import api as api_module

class DummyEngine:
    def generate(self, prompt, language="auto", temperature=None, max_tokens=None, seed=None, model=None, timeout=None):
        return ("Hello baseline.", {
            "model": model or "llama3",
            "temperature": temperature if temperature is not None else 0.2,
            "seed": seed if seed is not None else 42,
            "max_tokens": max_tokens if max_tokens is not None else 512,
            "latency_ms": 12,
            "usage": {"tokens_prompt": 5, "tokens_output": 3, "total_tokens": 8}
        })

def test_api_schema_and_metadata(monkeypatch):
    # Patch the singleton engine used by the API
    monkeypatch.setattr(api_module, "engine", DummyEngine())
    client = TestClient(app)
    payload = {"prompt": "Ping?", "language": "en", "temperature": 0.2, "max_tokens": 64, "seed": 42, "model": "llama3"}
    r = client.post("/v1/ask", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "response" in data and "model" in data and "usage" in data and "latency_ms" in data and "timestamp" in data
    assert isinstance(data["usage"]["tokens_prompt"], int)
