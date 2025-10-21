# inference.py
import time, math, re, requests
from typing import Dict, Any, Tuple
from app.config import settings
from app.logger import get_logger
from app.prompts import BASELINE_SYSTEM_PROMPT, language_hint

log = get_logger("inference")

UNCERTAINTY_REGEX = re.compile(
    r"\b(i\s*(do\s*not|don't)\s*know|not\s*sure|uncertain|no\s*information\s*available|as\s+an\s+ai|cannot\s+access\s+the\s+internet)\b",
    re.IGNORECASE
)

UNKNOWN_STRING = "I don’t know based on my current knowledge."

def estimate_tokens(text: str) -> int:
    """
    Approximate token count: a crude, model-agnostic heuristic.
    """
    text = text.strip()
    if not text:
        return 0
    # Approx: 1 token ≈ 4 chars for English-like text; VN similar magnitude here
    return max(1, math.ceil(len(text) / 4))

def normalize_response(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return UNKNOWN_STRING
    if UNCERTAINTY_REGEX.search(t):
        return UNKNOWN_STRING
    return t

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, seed: int, timeout: int) -> Dict[str, Any]:
        """
        Calls Ollama's /api/chat with a single user message and system prompt.
        Returns a dict with keys: 'response', 'usage' (optional).
        """
        url = f"{self.base_url}/api/chat"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        json_body = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "seed": int(seed),
            },
            "stream": False
        }
        resp = requests.post(url, json=json_body, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
        data = resp.json()
        # Ollama returns { message: {content: ...}, eval_count, prompt_eval_count } for chat
        content = data.get("message", {}).get("content", "")
        usage = {
            "tokens_prompt": int(data.get("prompt_eval_count") or 0),
            "tokens_output": int(data.get("eval_count") or 0),
        }
        usage["total_tokens"] = usage["tokens_prompt"] + usage["tokens_output"]
        return {"response": content, "usage": usage}

class InferenceEngine:
    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(self, prompt: str, language: str = "auto",
                 temperature: float = None, max_tokens: int = None, seed: int = None,
                 model: str = None, timeout: int = None) -> Tuple[str, Dict[str, Any]]:
        start = time.time()
        model = model or settings.model
        temperature = float(settings.temperature if temperature is None else temperature)
        max_tokens = int(settings.max_tokens if max_tokens is None else max_tokens)
        seed = int(settings.seed if seed is None else seed)
        timeout = int(settings.request_timeout_s if timeout is None else timeout)

        sys_prompt = BASELINE_SYSTEM_PROMPT + "\n" + language_hint(language)
        try:
            result = self.client.chat(
                model=model,
                system_prompt=sys_prompt,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                timeout=timeout
            )
            raw = result["response"]
            response = normalize_response(raw)
            usage = result.get("usage") or {}
            if not usage:
                # Backfill usage crudely if Ollama returns nothing
                ptoks = estimate_tokens(sys_prompt + prompt)
                otoks = estimate_tokens(response)
                usage = {"tokens_prompt": ptoks, "tokens_output": otoks, "total_tokens": ptoks + otoks}
        except requests.exceptions.ReadTimeout:
            response = "Request to model timed out. The model may be busy or the response is taking too long."
            usage = {"tokens_prompt": 0, "tokens_output": 0, "total_tokens": 0}
        except requests.exceptions.ConnectionError:
            response = "Model host is not reachable at {}. Please ensure the model host (Ollama) is running and the model is pulled.".format(settings.ollama_host)
            usage = {"tokens_prompt": 0, "tokens_output": 0, "total_tokens": 0}
        except requests.exceptions.RequestException as e:
            # Catch-all for other request-related errors
            response = f"Request error when calling model: {e}"
            usage = {"tokens_prompt": 0, "tokens_output": 0, "total_tokens": 0}
        except RuntimeError as e:
            response = f"{e}"
            usage = {"tokens_prompt": 0, "tokens_output": 0, "total_tokens": 0}

        latency_ms = int((time.time() - start) * 1000)
        meta = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "latency_ms": latency_ms,
            "usage": usage
        }
        log.info("inference_complete", extra={"extra": {"prompt": prompt, "response": response, **meta}})
        return response, meta

# Singleton for API usage
client = OllamaClient(settings.ollama_host)
engine = InferenceEngine(client)
