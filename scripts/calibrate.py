"""Fit a calibration temperature on held-out decision items (never on the set you report on).

    uv run python scripts/calibrate.py ITEMS.jsonl READS.jsonl

ITEMS is JevBench-format `choice` items (id, state, question.instructions, question.criteria, expected), for example
Cygnet's own generated set, cygnet-recipe/calibration/items.jsonl. The model's raw option scores for each item are
cached in READS, so refitting does not rerun the model. Prints the NLL-optimal T over 0.5..5.0 and, as a check, the
ECE on one half with T fitted on the other.
"""
import json
import math
import sys
from pathlib import Path

import torch

from rmdm.engine import decision_text, load_engine, render_state
from rmdm.readout import distribution


def read_all(items_path: Path, reads_path: Path) -> list[tuple[list[float], int]]:
    items = [json.loads(line) for line in items_path.open() if line.strip()]
    cached = {}
    if reads_path.exists():
        cached = {r["id"]: r for r in map(json.loads, reads_path.open())}
    missing = [it for it in items if it["id"] not in cached]
    if missing:
        engine = load_engine()
        with reads_path.open("a") as out:
            for n, it in enumerate(missing, 1):
                labels = list(it["question"]["criteria"])
                text = decision_text(render_state(it["state"]), it["question"]["instructions"],
                                     [str(it["question"]["criteria"][k]) for k in labels])
                scores, _ = engine.scores(text, [], len(labels))
                r = {"id": it["id"], "labels": labels, "scores": scores.tolist(), "model": engine.model_id,
                     "revision": engine.revision}
                out.write(json.dumps(r) + "\n")
                out.flush()
                cached[it["id"]] = r
                print(f"\r{n}/{len(missing)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    return [(cached[it["id"]]["scores"], cached[it["id"]]["labels"].index(it["expected"])) for it in items]


def nll(data, t):
    return -sum(math.log(max(distribution(torch.tensor(s), t)[g], 1e-12)) for s, g in data) / len(data)


def ece(data, t, bins=10):
    rows = []
    for s, g in data:
        p = distribution(torch.tensor(s), t)
        best = max(range(len(p)), key=p.__getitem__)
        rows.append((p[best], best == g))
    total = 0.0
    for b in range(bins):
        inside = [(c, ok) for c, ok in rows if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if inside:
            total += len(inside) / len(rows) * abs(sum(c for c, _ in inside) / len(inside)
                                                   - sum(ok for _, ok in inside) / len(inside))
    return total


if __name__ == "__main__":
    data = read_all(Path(sys.argv[1]), Path(sys.argv[2]))
    grid = [round(0.5 + 0.1 * i, 2) for i in range(46)]
    acc = sum(max(range(len(s)), key=s.__getitem__) == g for s, g in data) / len(data)
    best = min(grid, key=lambda t: nll(data, t))
    print(f"{len(data)} items, accuracy {acc:.3f}; T = {best}: NLL {nll(data, 1.0):.3f} -> {nll(data, best):.3f}")
    fit, held = data[0::2], data[1::2]
    t_half = min(grid, key=lambda t: nll(fit, t))
    print(f"fit on {len(fit)} (T = {t_half}), held-out {len(held)}: ECE {ece(held, 1.0):.3f} -> {ece(held, t_half):.3f}")
