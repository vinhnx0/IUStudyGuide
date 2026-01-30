from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import time
import math
import random
from pathlib import Path
import yaml

class RateLimiter:
    """Simple RPM-based rate limiter (single-threaded).
    Ensures at most `rpm` requests per minute. If rpm is None or 0, it does nothing.
    """
    def __init__(self, rpm: int | None):
        self.rpm = int(rpm) if rpm else 0
        self._min_interval = 60.0 / self.rpm if self.rpm > 0 else 0.0
        self._next_allowed = 0.0

    def wait(self):
        if self._min_interval <= 0:
            return
        now = time.time()
        if now < self._next_allowed:
            time.sleep(self._next_allowed - now)
        self._next_allowed = max(self._next_allowed + self._min_interval, time.time())

def exponential_backoff_sleep(attempt: int,
                              initial: float,
                              max_wait: float,
                              jitter: bool) -> float:
    """Return seconds to sleep for the given attempt (1-based)."""
    # attempt 1 -> initial, attempt 2 -> 2*initial, capped at max_wait
    delay = min(max_wait, initial * (2 ** (attempt - 1)))
    if jitter:
        # Randomize in [0.5, 1.5]x to avoid thundering herd
        delay *= random.uniform(0.5, 1.5)
    time.sleep(delay)
    return delay


import yaml
from dotenv import load_dotenv

from app.rag_pipeline import RAGPipeline


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_eval(questions: list[str], ask_fn, cfg: dict):
    """
    Run evaluation for a list of questions with retry, delay, and rate-limiting.

    questions: list of question strings
    ask_fn: callable(prompt:str) -> dict  (usually RAGPipeline.ask)
    cfg: full config dict to read evaluation and retry parameters
    """
    ecfg = cfg.get("evaluation", cfg.get("eval", {}))  # backward-compatible key names
    rpm = int(ecfg.get("requests_per_minute", 1) or 0)
    fixed_delay = float(ecfg.get("per_query_delay_seconds", ecfg.get("fixed_delay_seconds", 0)) or 0)
    max_retries = int(ecfg.get("max_retries", 6))
    initial_backoff = float(ecfg.get("initial_backoff_seconds", 2))
    max_backoff = float(ecfg.get("max_backoff_seconds", 60))
    use_jitter = bool(ecfg.get("jitter", True))

    limiter = RateLimiter(rpm if rpm > 0 else None)
    results = []

    for i, q in enumerate(questions, 1):
        # --- RATE LIMIT ---
        # Make sure we don't exceed the requests_per_minute limit
        limiter.wait()

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                # Call the model using the provided function
                resp = ask_fn(q)
                results.append({"idx": i, "query": q, "resp": resp})
                break
            except Exception as e:
                msg = str(e).lower()
                # Detect retryable errors (quota, rate-limit, timeouts)
                retryable = any(tok in msg for tok in ["429", "quota", "rate limit", "temporarily", "timeout", "unavailable"])
                retry_after = getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                        continue
                    except Exception:
                        pass
                # Exponential backoff for retryable errors
                if retryable and attempt < max_retries:
                    exponential_backoff_sleep(attempt, initial_backoff, max_backoff, use_jitter)
                    last_err = e
                    continue
                last_err = e
                break

        # Log final error if retries failed
        if last_err and (not results or results[-1].get("idx") != i):
            results.append({"idx": i, "query": q, "error": repr(last_err)})

        # --- FIXED DELAY BETWEEN QUERIES ---
        if fixed_delay > 0:
            time.sleep(fixed_delay + (random.uniform(-1.0, 1.0) if use_jitter else 0))

    return results

def save_only(
    cfg: dict,
    qs_path: Path,
    out_path: Path,
    include_citations: bool = False,
    include_debug: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Run queries via run_eval (with rate limit/backoff) and save all results to JSONL."""
    load_dotenv()
    rag = RAGPipeline(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load all questions from the JSONL file ---
    questions: List[str] = []
    with open(qs_path, "r", encoding="utf-8") as f_in:
        for i, line in enumerate(f_in):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("query", "")
            if q:
                questions.append(q)

    # --- Define the model call function ---
    def ask_fn(prompt: str) -> Dict[str, Any]:
        # Call the RAG pipeline to get a full response
        return rag.ask(prompt, debug=bool(include_debug))

    # --- Run evaluation with retry/backoff ---
    results = run_eval(questions, ask_fn, cfg)

    # --- Save all responses back to file ---
    with open(out_path, "w", encoding="utf-8") as f_out:
        for item in results:
            entry: Dict[str, Any] = {"query": item["query"]}
            if "resp" in item:
                resp = item["resp"] or {}
                entry["answer"] = resp.get("answer", "")
                entry["aliases"] = resp.get("aliases", {})
                if include_citations:
                    entry["citations"] = resp.get("citations", [])
                if include_debug:
                    entry["debug"] = resp.get("debug", {})
            else:
                entry["answer"] = ""
                entry["aliases"] = {}
                entry["error"] = item.get("error", "unknown error")
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Save answers from RAG pipeline for a batch of queries")
    parser.add_argument("--qs-file", type=str, default=None, help="Path to qs.jsonl (overrides config)")
    parser.add_argument("--out", type=str, default=None, help="Output path for answers JSONL")
    parser.add_argument("--include-citations", action="store_true", help="Include citations in saved answers")
    parser.add_argument("--include-debug", action="store_true", help="Include debug info in saved answers")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N queries (useful for testing)")
    args = parser.parse_args(argv)

    cfg = load_config()
    qs_path = Path(args.qs_file) if args.qs_file else Path(cfg.get("evaluation", {}).get("qs_file", "eval/qs.jsonl"))
    out_path = Path(args.out) if args.out else Path("eval/answers.jsonl")

    save_only(
        cfg,
        qs_path,
        out_path,
        include_citations=args.include_citations,
        include_debug=args.include_debug,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
