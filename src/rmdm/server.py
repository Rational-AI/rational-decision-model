"""Local HTTP server. No outbound network: the model, the images and the calibration profiles are all local.

POST /v1/decisions   Rational's multimodal decision API (README.md)
POST /v1/systemone   JevBench's `typesafe` wire format, text only, for the benchmark baseline

    uv run python -m rmdm.server            # RMDM_PORT (8009), RMDM_PROFILE (default), RMDM_SYSTEMONE_TEMPERATURE (1.0)

One request at a time: there is one model on one GPU, and the server is single-threaded on purpose.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .box import locate
from .engine import PROMPT_VERSION, Engine, decision_text, load_engine, render_state
from .readout import distribution

PROFILES = json.loads((Path(__file__).parent / "calibration.json").read_text())
DEFAULT_PROFILE = os.environ.get("RMDM_PROFILE", "gemma4-12b-4bit-mlx")
SYSTEMONE_TEMPERATURE = float(os.environ.get("RMDM_SYSTEMONE_TEMPERATURE", "1.0"))
ABSTAIN = {"id": "human_review",
           "description": "The evidence is insufficient, contradictory, or none of the other options applies: "
                          "request human review."}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class BadRequest(ValueError):
    pass


def decode_image(data) -> Image.Image:
    """Base64 or a data: URI only. No URLs (no network) and no server-side paths (no file reads for the caller)."""
    if not isinstance(data, str):
        raise BadRequest("image must be a base64 string")
    if data.startswith("data:"):
        data = data.partition(",")[2]
    if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise BadRequest("image too large")
    try:
        return Image.open(io.BytesIO(base64.b64decode(data, validate=True))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError) as e:
        raise BadRequest(f"unreadable image: {e}") from e


class Images:
    """Turns asset references in request strings into image markers, collecting the images in text order.

    As in TypeSafe's API, every image is listed once in the request's `assets` map under a reference string, and
    the reference may appear anywhere text goes: the input, a question's instructions, an option's description.
    """

    def __init__(self, engine: Engine, assets):
        if not isinstance(assets, dict):
            raise BadRequest("'assets' must be an object mapping reference strings to base64 images")
        self.marker = engine.image_marker
        self.images = {ref: decode_image(data) for ref, data in assets.items()}
        if any(not ref.strip() for ref in self.images):
            raise BadRequest("an asset reference cannot be empty")
        self.pattern = re.compile("|".join(map(re.escape, sorted(self.images, key=len, reverse=True))) or "(?!)")
        self.used = set()

    def resolve(self, text: str, out: list) -> str:
        def swap(match):
            self.used.add(match.group(0))
            out.append(self.images[match.group(0)])
            return self.marker
        # caller text cannot smuggle in image tokens of its own
        return self.pattern.sub(swap, text.replace(self.marker, ""))


def as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def decision_options(d: dict) -> list[dict]:
    """choice and score take `options` (score: ordered levels, lowest first); noul takes optional `criteria`."""
    kind = d.get("type", "choice")
    if kind == "noul":
        criteria = d.get("criteria") or {}
        if not isinstance(criteria, dict):
            raise BadRequest("noul criteria must be an object with 'true' and/or 'false'")
        options = [{"id": "true", "description": criteria.get("true", "Yes.")},
                   {"id": "false", "description": criteria.get("false", "No.")}]
    elif kind in ("choice", "score"):
        options = list(d.get("options") or [])
    else:
        raise BadRequest(f"unsupported decision type {kind!r} (choice, noul, score, box)")
    if not all(isinstance(o, dict) and isinstance(o.get("id"), str) and "description" in o for o in options):
        raise BadRequest("every option needs a string 'id' and a 'description'")
    if d.get("abstain"):
        options.append(ABSTAIN)
    ids = [o["id"] for o in options]
    if len(set(ids)) != len(ids):
        raise BadRequest("option ids must be unique")
    return options


def decide(engine: Engine, body: dict) -> dict:
    """One decision (`decision`) or several over the same input (`decisions`, a map of id to decision)."""
    parts, single, many = body.get("input"), body.get("decision"), body.get("decisions")
    if not isinstance(parts, list) or (single is None) == (many is None):
        raise BadRequest("expected 'input' (a list) and either 'decision' or 'decisions'")
    decisions = {(single.get("id") if isinstance(single, dict) else None): single} if single else many
    if not isinstance(decisions, dict) or not all(isinstance(d, dict) for d in decisions.values()):
        raise BadRequest("'decisions' must map ids to decision objects")
    profile_name = body.get("calibration_profile") or DEFAULT_PROFILE
    if profile_name not in PROFILES:
        raise BadRequest(f"unknown calibration profile {profile_name!r}")
    assets = dict(body.get("assets") or {})

    # The state: text and JSON parts in order, and inline images as their own lines (a label, when given, sits
    # right before its image). Inline images become assets under private references.
    lines, state_images = [], []
    for n, part in enumerate(parts):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "image":
            ref = f"\x00inline-image-{n}\x00"
            assets[ref] = part.get("image")
            lines.append(f"{part['label']}:\n{ref}" if part.get("label") else ref)
        elif kind == "text" and isinstance(part.get("text"), str):
            lines.append(part["text"])
        elif kind == "json" and "value" in part:
            lines.append(render_state(part["value"]))
        else:
            raise BadRequest(f"unsupported input part: {kind!r}")
    images = Images(engine, assets)
    state_text = images.resolve("\n\n".join(lines), state_images)

    prepared = []
    for decision_id, d in decisions.items():
        if d.get("type") == "box":
            ref = d.get("image")
            if ref is None and len(images.images) == 1:
                ref = next(iter(images.images))
            if ref not in images.images or not isinstance(d.get("instructions"), str):
                raise BadRequest("a box needs text 'instructions' naming the target and an 'image' reference "
                                 "(optional when the request holds exactly one image)")
            images.used.add(ref)
            prepared.append((decision_id, d, None, None, [images.images[ref]]))
            continue
        options = decision_options(d)
        question_images = []  # resolved in text order: instructions, then each option
        instructions = images.resolve(as_text(d.get("instructions") or ""), question_images)
        descriptions = [images.resolve(as_text(o["description"]), question_images) for o in options]
        prepared.append((decision_id, d, options, decision_text(state_text, instructions, descriptions),
                         state_images + question_images))
    unused = set(images.images) - images.used
    if unused:
        raise BadRequest(f"assets not referenced anywhere: {sorted(unused)}")

    temperature = PROFILES[profile_name]["temperature"]
    input_sha256 = hashlib.sha256(json.dumps({"input": parts, "assets": body.get("assets") or {}},
                                             sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    answers, started_all = {}, time.perf_counter()
    for decision_id, d, options, text, decision_images in prepared:
        started = time.perf_counter()
        if options is None:  # a box: several readouts over a gridded, then zoomed, picture
            answer = {"type": "box", **locate(engine, decision_images[0], d["instructions"], temperature),
                      "calibration": {"method": "temperature", "temperature": temperature, "profile": profile_name},
                      "latency_ms": round((time.perf_counter() - started) * 1000),
                      "trace": {"backend": engine.backend, "model_id": engine.model_id,
                                "model_revision": engine.revision, "prompt_version": PROMPT_VERSION,
                                "input_sha256": input_sha256, "timestamp": datetime.now(timezone.utc).isoformat()}}
            answers[decision_id] = answer
            continue
        try:
            # everything up to the question is shared by further decisions on the same input
            scores, prompt_tokens = engine.scores(text, decision_images, len(options),
                                                  shared_chars=len(state_text.rstrip()) + 2)
        except ValueError as e:
            raise BadRequest(str(e)) from e
        ids = [o["id"] for o in options]
        calibrated = dict(zip(ids, distribution(scores, temperature)))
        answer = {
            "type": d.get("type", "choice"),
            "decision": max(calibrated, key=calibrated.get),
            "probabilities": calibrated,
            "raw_probabilities": dict(zip(ids, distribution(scores))),
        }
        if answer["type"] == "noul":
            answer["noul"] = calibrated["true"]
        elif answer["type"] == "score":
            levels = [o["id"] for o in options if o is not ABSTAIN]
            mass = sum(calibrated[i] for i in levels)
            answer["expected_level"] = sum(n * calibrated[i] for n, i in enumerate(levels)) / mass
        answer.update({
            "calibration": {"method": "temperature", "temperature": temperature, "profile": profile_name},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "usage": {"input_tokens": prompt_tokens, "output_positions": 1},
            # What governance needs to replay the decision.
            "trace": {"backend": engine.backend, "model_id": engine.model_id, "model_revision": engine.revision,
                      "prompt_version": PROMPT_VERSION, "options": ids, "input_sha256": input_sha256,
                      "timestamp": datetime.now(timezone.utc).isoformat()},
        })
        answers[decision_id] = answer
    model = body.get("model") or engine.model_id
    if single is not None:
        return {"model": model, "decision_id": single.get("id"), **next(iter(answers.values()))}
    return {"model": model, "answers": answers, "latency_ms": round((time.perf_counter() - started_all) * 1000)}


def systemone(engine: Engine, body: dict) -> dict:
    """JevBench `typesafe`: choice ({label: criteria}), score ([level, ...]) and noul ({"true": .., "false": ..})."""
    q = (body.get("questions") or {}).get("decision")
    if not isinstance(q, dict):
        raise BadRequest("expected questions.decision")
    kind, criteria = q.get("type"), q.get("criteria")
    if kind == "score" and isinstance(criteria, list):
        labeled = [(str(i), c) for i, c in enumerate(criteria)]
    elif kind in ("choice", "noul") and isinstance(criteria, dict):
        labeled = list(criteria.items())
    else:
        raise BadRequest(f"unsupported question {kind!r} / criteria")
    instructions = q.get("instructions") or ""
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    text = decision_text(render_state(body.get("state") or ""), instructions, [str(c) for _, c in labeled])
    try:
        scores, prompt_tokens = engine.scores(text, [], len(labeled))
    except ValueError as e:
        raise BadRequest(str(e)) from e
    probs = dict(zip((label for label, _ in labeled), distribution(scores, SYSTEMONE_TEMPERATURE)))
    if kind == "noul":
        true_key = next((k for k in probs if str(k).lower() == "true"), None)
        if true_key is None:
            raise BadRequest("noul criteria have no 'true' key")
        answer = {"type": "noul", "noul": probs[true_key]}
    else:
        answer = {"type": kind, "choice": max(probs, key=probs.get), "probabilities": probs}
    return {"model": engine.model_id, "answers": {"decision": answer},
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}}


ROUTES = {"/v1/decisions": decide, "/v1/systemone": systemone}


def serve(engine: Engine, port: int):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            payload = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            route = ROUTES.get(self.path)
            if route is None:
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("the body must be a JSON object")
                self._send(200, route(engine, body))
            except (BadRequest, json.JSONDecodeError) as e:
                # JevBench scores a 422 as one wrong answer and carries on; other failures count toward its stop rule.
                self._send(422, {"error": str(e)})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    engine = load_engine()
    port = int(os.environ.get("RMDM_PORT", "8009"))
    print(f"rmdm: {engine.backend} {engine.model_id}@{engine.revision[:8]}, http://127.0.0.1:{port}",
          file=sys.stderr, flush=True)
    serve(engine, port)
