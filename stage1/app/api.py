# api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
from datetime import datetime, timezone
from app.config import settings
from app.inference import engine
from app.logger import get_logger

log = get_logger("api")

class AskRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    language: Literal["auto", "vi", "en"] = Field(default="auto")
    temperature: float = Field(default=None)
    max_tokens: int = Field(default=None)
    seed: int = Field(default=None)
    model: str = Field(default=None)

class Usage(BaseModel):
    tokens_prompt: int
    tokens_output: int
    total_tokens: int

class AskResponse(BaseModel):
    response: str
    model: str
    usage: Usage
    latency_ms: int
    timestamp: str

app = FastAPI(title=settings.api_title, version=settings.api_version)

@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest):
    response, meta = engine.generate(
        prompt=req.prompt,
        language=req.language,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        seed=req.seed,
        model=req.model
    )
    ts = datetime.now(timezone.utc).isoformat()
    body: Dict[str, Any] = {
        "response": response,
        "model": meta["model"],
        "usage": meta["usage"],
        "latency_ms": meta["latency_ms"],
        "timestamp": ts
    }
    log.info("api_request", extra={"extra": {"request": req.model_dump(), **body}})
    return body
