import requests
import logging

logger = logging.getLogger(__name__)


def local_generate(prompt: str, cfg: dict) -> str:
    llm_cfg = cfg.get("llm", {})
    backend = llm_cfg.get("default_backend", "local")

    if backend != "local":
        raise ValueError(f"Unsupported local LLM backend: {backend}")

    base_url = llm_cfg.get("base_url", "http://localhost:1234/v1")
    model = llm_cfg.get("model")

    if not model:
        raise ValueError("local_llm.model must be set in config.yaml")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": llm_cfg.get("temperature", 0.3),
        "top_p": llm_cfg.get("top_p", 0.9),
        "max_tokens": llm_cfg.get("max_output_tokens", 512),
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.exception("LM Studio request failed")
        raise RuntimeError("Local LLM (LM Studio) call failed") from e

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
