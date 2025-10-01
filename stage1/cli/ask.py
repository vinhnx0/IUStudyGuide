# ask.py
import argparse, json, os, sys
from datetime import datetime, timezone
from app.inference import engine
from app.logger import get_logger
from pathlib import Path

log = get_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="Stage1 CLI (Ollama)")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--language", default="auto", choices=["auto", "vi", "en"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    resp, meta = engine.generate(
        prompt=args.prompt, language=args.language,
        temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed, model=args.model
    )

    print(resp)

    record = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "prompt": args.prompt,
        "response": resp,
        "model": meta["model"],
        "temperature": meta["temperature"],
        "seed": meta["seed"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": meta["latency_ms"],
        "tokens_prompt": meta["usage"]["tokens_prompt"],
        "tokens_output": meta["usage"]["tokens_output"],
        "total_tokens": meta["usage"]["total_tokens"],
    }

    out_dir = Path("eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cli_history.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("cli_ask_complete", extra={"extra": record})

if __name__ == "__main__":
    main()
