"""One decision on local images, from the command line (no server).

    HF_HUB_OFFLINE=1 uv run python scripts/decide.py PHOTO [PHOTO ...] \
      --instructions "What should we do with this component?" \
      --option ok="No visible defect: keep in service." \
      --option inspect="Visible wear: schedule an inspection." \
      --option replace="Clear damage: replace it now." \
      --abstain [--state '{"machine": "A17"}'] [--profile raw]
"""
import argparse
import base64
import json
from pathlib import Path

from rmdm.engine import load_engine
from rmdm.server import decide

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", type=Path)
parser.add_argument("--instructions", required=True)
parser.add_argument("--option", action="append", required=True, metavar="ID=DESCRIPTION")
parser.add_argument("--abstain", action="store_true", help="add a human_review option")
parser.add_argument("--state", help="JSON object passed as structured state")
parser.add_argument("--profile", help="calibration profile (default: the model's)")
args = parser.parse_args()
for missing in (p for p in args.images if not p.is_file()):
    parser.error(f"no such image: {missing}")

options = [dict(zip(("id", "description"), o.split("=", 1))) for o in args.option]
parts = [{"type": "image", "label": f"Image {i}: {p.name}" if len(args.images) > 1 else None,
          "image": base64.b64encode(p.read_bytes()).decode()} for i, p in enumerate(args.images, 1)]
if args.state:
    parts.append({"type": "json", "value": json.loads(args.state)})
body = {"input": parts, "decision": {"id": "cli", "instructions": args.instructions, "options": options,
                                     "abstain": args.abstain}}
if args.profile:
    body["calibration_profile"] = args.profile
result = decide(load_engine(), body)
print(json.dumps({k: result[k] for k in ("decision", "probabilities", "raw_probabilities", "latency_ms")}, indent=1))
