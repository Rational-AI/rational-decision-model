"""Accuracy by JevBench tier and latency from a JevBench results.jsonl.

    uv run python scripts/summarize.py runs/<run>/results.jsonl
"""
import json
import statistics
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
tier = lambda r: r["task_id"].split("-")[0] if r["task_id"].split("-")[0] in ("easy", "hard") else "standard"
correct = sum(r["correct"] for r in rows)
print(f"correct {correct}/{len(rows)} ({correct / len(rows):.1%}), valid {sum(r['valid'] for r in rows)}/{len(rows)}")
print(" · ".join(f"{t} {sum(r['correct'] for r in rows if tier(r) == t)}/{sum(tier(r) == t for r in rows)}"
                 for t in ("easy", "standard", "hard")))
lat = sorted(r["latency_s"] for r in rows if tier(r) == "standard")
print(f"standard-tier latency p50 {statistics.median(lat):.3f}s p95 {lat[int(0.95 * (len(lat) - 1))]:.3f}s, "
      f"mean input tokens {statistics.mean(r['usage']['prompt_tokens'] for r in rows):.0f}")
