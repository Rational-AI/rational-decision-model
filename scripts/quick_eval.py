"""Two-minute sanity check: accuracy and chosen-letter histogram on JevBench's short tiers (easy + standard, 120 items).

    HF_HUB_OFFLINE=1 uv run python scripts/quick_eval.py JEVBENCH_DIR [MODEL REVISION]

A readout bug shows up here as a collapse onto one letter (the causal-mask bug put 106 of 120 answers on A).
Reference: Cygnet's 12B bf16 on vLLM gets 118/120.
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 2:
    os.environ["RMDM_MODEL"], os.environ["RMDM_REVISION"] = sys.argv[2], sys.argv[3]

from rmdm.engine import decision_text, load_engine, render_state  # noqa: E402
from rmdm.readout import LABELS, distribution  # noqa: E402

data = Path(sys.argv[1]) / "datasets" / "public"
items = [json.loads(line) for f in ("easy.jsonl", "original.jsonl") for line in (data / f).open() if line.strip()]
engine = load_engine()
correct, chosen, started = 0, collections.Counter(), time.time()
for item in items:
    q = item["question"]
    criteria = q["criteria"]
    labels = list(criteria) if isinstance(criteria, dict) else [str(i) for i in range(len(criteria))]
    descriptions = [str(criteria[k]) for k in labels] if isinstance(criteria, dict) else [str(c) for c in criteria]
    instructions = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
    text = decision_text(render_state(item["state"]), instructions, descriptions)
    p = distribution(engine.scores(text, [], len(labels))[0])
    best = max(range(len(p)), key=p.__getitem__)
    chosen[LABELS[best]] += 1
    expected = str(item["expected"]).lower()
    if q["type"] == "noul":
        expected = "true" if expected in ("true", "yes") else "false"
    correct += labels[best].lower() == expected
print(f"{engine.model_id}: {correct}/{len(items)}, letters {dict(sorted(chosen.items()))}, {time.time() - started:.0f}s")
