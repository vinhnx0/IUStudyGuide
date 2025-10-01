# run_eval.py
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.inference import engine

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run Stage1 baseline evaluation")
    parser.add_argument("--input", default="eval/sample_prompts.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    data = read_jsonl(args.input)
    outputs: List[Dict[str, Any]] = []

    for item in data:
        pid = item.get("id")
        prompt = item.get("prompt")
        language = item.get("lang", "auto")
        start = time.time()
        resp, meta = engine.generate(
            prompt=prompt, language=language,
            temperature=args.temperature, max_tokens=args.max_tokens,
            seed=args.seed, model=args.model
        )
        latency_ms = int((time.time() - start) * 1000)
        row = {
            "id": pid,
            "prompt": prompt,
            "response": resp,
            "model": meta["model"],
            "temperature": meta["temperature"],
            "seed": meta["seed"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "tokens_prompt": meta["usage"]["tokens_prompt"],
            "tokens_output": meta["usage"]["tokens_output"],
            "total_tokens": meta["usage"]["total_tokens"]
        }
        outputs.append(row)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or f"eval/outputs_{ts}.jsonl"
    Path("eval").mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, outputs)

    # Summary
    n = len(outputs)
    avg_latency = sum(o["latency_ms"] for o in outputs) / max(1, n)
    unknowns = sum(1 for o in outputs if o["response"].strip() == "I don’t know based on my current knowledge.")
    print(f"Eval complete: count={n}, avg_latency_ms={avg_latency:.1f}, unknown_count={unknowns}")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
