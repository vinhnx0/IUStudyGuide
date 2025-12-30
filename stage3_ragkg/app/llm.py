from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.logging_utils import get_logger, log_call

logger = get_logger(__name__)

_ENV_LOADED = False


def _load_env_from_parent() -> None:
    """
    Keep .env loading for other optional backends (e.g., Gemini) if your project uses them.
    Groq is fully removed.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    app_dir = Path(__file__).resolve().parent
    root_dir = app_dir.parent.parent
    env_path = root_dir / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    _ENV_LOADED = True


def _backend(cfg: Dict[str, Any]) -> str:
    # Required key: cfg["llm"]["default_backend"], but be defensive.
    return str((cfg.get("llm", {}) or {}).get("default_backend", "")).lower().strip()


def _local_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return local backend config.

    Expected:
      llm:
        default_backend: local
        local:
          base_url: http://localhost:1234/v1
          model: ...

    Backward-compatible fallback:
      llm:
        base_url/model/... (flat)
    """
    llm_cfg = (cfg.get("llm", {}) or {})
    if isinstance(llm_cfg.get("local"), dict):
        return llm_cfg.get("local") or {}
    # fallback: treat llm section itself as local config
    return llm_cfg


# ======================================================
# PYDANTIC JSON OUTPUT MODELS
# ======================================================

class Decompose(BaseModel):
    entities: List[str] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list)
    kg_queries: List[Dict[str, Any]] = Field(default_factory=list)


class RouterDecision(BaseModel):
    route: Literal["FAST", "SLOW"]
    intent: str
    entities: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Constraints(BaseModel):
    """Structured extraction of curriculum-planning constraints from a user query."""

    min_credits_per_semester: Optional[int] = None
    max_credits_per_semester: Optional[int] = None
    terms_remaining: Optional[int] = None
    # IU program semester index for the *next* semester the student will register for (1..8)
    current_semester_index: Optional[int] = None
    # Optional explicit completed semester numbers found in the query (e.g., [1,2])
    completed_semesters: List[int] = Field(default_factory=list)

    completed_courses: List[str] = Field(default_factory=list)
    preferred_courses: List[str] = Field(default_factory=list)
    avoid_courses: List[str] = Field(default_factory=list)


MODEL_BY_STEP: Dict[str, Type[BaseModel]] = {
    "decompose": Decompose,
    "router": RouterDecision,
    "constraints": Constraints,
}


def _get_model_for_step(step: str) -> Type[BaseModel]:
    step = (step or "").strip().lower()
    if step not in MODEL_BY_STEP:
        raise ValueError(f"Unknown JSON step '{step}'. Expected one of: {sorted(MODEL_BY_STEP.keys())}")
    return MODEL_BY_STEP[step]


def _split_system_user(prompt: str) -> Tuple[str, str]:
    """Split a legacy single-string prompt into (system, user).

    Prompt templates in this codebase often start with a "SYSTEM:" block, and
    then include structured sections like KG_FINDINGS/EVIDENCE/QUESTION. LM Studio
    works best if we send system + user as separate messages.

    Heuristic:
      - If prompt starts with "SYSTEM:" and contains a later marker section,
        treat everything before the first marker as system; the rest as user.
      - Otherwise: empty system, entire prompt as user.
    """
    if not prompt:
        return "", ""

    markers = [
        "\n\nKG_FINDINGS",
        "\n\nPREREQ_EDGES",
        "\n\nEVIDENCE",
        "\n\nQUESTION",
    ]
    idx = -1
    for m in markers:
        pos = prompt.find(m)
        if pos != -1:
            idx = pos
            break

    if prompt.lstrip().startswith("SYSTEM:") and idx != -1:
        system_part = prompt[:idx].strip()
        user_part = prompt[idx:].strip()
        # Remove leading "SYSTEM:" tag to avoid leaking into instruction text.
        system_part = system_part.replace("SYSTEM:", "", 1).strip()
        return system_part, user_part

    # Fallback: no structured split.
    return "", prompt


def _lmstudio_chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout_s: int = 120,
) -> str:
    """Call a local OpenAI-compatible server (LM Studio).

    Expected base_url examples:
      - http://localhost:1234/v1
      - http://127.0.0.1:1234/v1
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ],
        "temperature": float(temperature),
        "stream": False,
    }

    # Some local servers reject max_tokens<=0 or -1; omit in that case.
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LM Studio error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def _safe_json_loads(raw: str) -> Any:
    """Parse JSON robustly from local-LLM outputs.

    Strategy:
      1) direct json.loads
      2) if fails, extract the largest {...} span and retry
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty JSON output")

    try:
        return json.loads(raw)
    except Exception:
        pass

    # Heuristic: take from first '{' to last '}'
    a = raw.find("{")
    b = raw.rfind("}")
    if a != -1 and b != -1 and b > a:
        candidate = raw[a : b + 1].strip()
        return json.loads(candidate)

    # Also support array outputs
    a = raw.find("[")
    b = raw.rfind("]")
    if a != -1 and b != -1 and b > a:
        candidate = raw[a : b + 1].strip()
        return json.loads(candidate)

    raise ValueError("Could not extract JSON from model output")


# ======================================================
# PUBLIC API
# ======================================================

def strip_think(text: str) -> str:
    if not text:
        return text
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@log_call(include_result=False)
def llm_generate_text(
    prompt: str,
    cfg: Dict[str, Any],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    backend = _backend(cfg)

    if backend != "local":
        raise RuntimeError(f"Unsupported backend: {backend}")

    lm_cfg = _local_cfg(cfg)
    base_url = str(lm_cfg.get("base_url", "http://localhost:1234/v1")).strip()
    text_model = str(lm_cfg.get("text_model", lm_cfg.get("model", "meta-llama-3.1-8b-instruct"))).strip()
    default_temp = float(lm_cfg.get("text_temperature", lm_cfg.get("temperature", 0.2)))
    default_max = int(lm_cfg.get("text_max_output_tokens", lm_cfg.get("max_output_tokens", 800)))

    sys_p, user_p = _split_system_user(prompt)
    logger.info(
        "LLM(local) text | base_url=%s model=%s temp=%.3f max_tokens=%s sys_len=%d user_len=%d",
        base_url,
        (model or text_model),
        (default_temp if temperature is None else float(temperature)),
        str(default_max if max_tokens is None else int(max_tokens)),
        len(sys_p or ""),
        len(user_p or ""),
    )

    return _lmstudio_chat_completion(
        system_prompt=sys_p,
        user_prompt=user_p,
        base_url=base_url,
        model=(model or text_model),
        temperature=(default_temp if temperature is None else float(temperature)),
        max_tokens=(default_max if max_tokens is None else int(max_tokens)),
        timeout_s=int(lm_cfg.get("timeout_s", 120)),
    )


@log_call(include_result=False)
def llm_generate_json(
    prompt: str,
    cfg: Dict[str, Any],
    *,
    step: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    backend = _backend(cfg)

    if backend != "local":
        raise RuntimeError(f"Unsupported backend for JSON: {backend}")

    lm_cfg = _local_cfg(cfg)
    base_url = str(lm_cfg.get("base_url", "http://localhost:1234/v1")).strip()
    json_model = str(lm_cfg.get("json_model", lm_cfg.get("model", "meta-llama-3.1-8b-instruct"))).strip()
    timeout_s = int(lm_cfg.get("timeout_s", 120))

    out_model = _get_model_for_step(step)
    schema = out_model.model_json_schema()

    # Prompt-only JSON contract (no Structured Output toggle required).
    schema_str = json.dumps(schema, ensure_ascii=False)
    json_contract = (
        "You are a JSON generator. Output ONLY valid JSON. No markdown. No extra text.\n"
        "The JSON must strictly follow this JSON Schema:\n"
        f"{schema_str}\n\n"
        "If a field is unknown, use a sensible default (empty string, empty array, or null)."
    )
    system_prompt = json_contract
    user_prompt = f"TASK INPUT:\n{prompt}\n"

    def _try_once(extra_fix: Optional[str] = None) -> Dict[str, Any]:
        sys_p = system_prompt if not extra_fix else (system_prompt + "\n\nFIX:\n" + extra_fix)
        raw = _lmstudio_chat_completion(
            system_prompt=sys_p,
            user_prompt=user_prompt,
            base_url=base_url,
            model=(model or json_model),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout_s=timeout_s,
        )
        raw = strip_think(raw)
        data = _safe_json_loads(raw)
        validated = out_model.model_validate(data)
        return validated.model_dump()

    logger.info(
        "LLM(local) json | step=%s base_url=%s model=%s temp=%.3f max_tokens=%d",
        step,
        base_url,
        (model or json_model),
        float(temperature),
        int(max_tokens),
    )

    # 1st attempt
    try:
        return _try_once()
    except Exception as e1:
        logger.warning("llm_generate_json: first parse/validate failed: %s", e1)

    # 2nd attempt with stronger fix instruction
    try:
        return _try_once(
            "Return ONLY the JSON object/array. No leading text. No trailing text. Do not wrap in markdown."
        )
    except Exception as e2:
        logger.error("llm_generate_json: second parse/validate failed: %s", e2)
        raise
