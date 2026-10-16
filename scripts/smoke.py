"""Multimodal smoke check: the image, not the text, must decide.

Each case runs the same question twice with two different images. The text is identical, so if the decision
changes, the image drove it. Images are drawn locally (no downloads).

    uv run python scripts/smoke.py
"""
import base64
import io
import json

from PIL import Image, ImageDraw, ImageFont

from rmdm.engine import load_engine
from rmdm.server import decide


def b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gauge(value: float) -> Image.Image:
    """A pressure gauge whose needle points at `value` on a 0-10 scale with a red zone above 8."""
    import math
    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((24, 24, 424, 424), outline="black", width=6)
    font = ImageFont.load_default(size=28)
    for v in range(11):
        a = math.radians(225 - 27 * v)
        x, y = 224 + 170 * math.cos(a), 224 - 170 * math.sin(a)
        d.text((x - 10, y - 14), str(v), fill="red" if v >= 8 else "black", font=font)
    a = math.radians(225 - 27 * value)
    d.line((224, 224, 224 + 150 * math.cos(a), 224 - 150 * math.sin(a)), fill="black", width=8)
    d.text((170, 300), "bar", fill="black", font=font)
    return img


def label(text: str) -> Image.Image:
    img = Image.new("RGB", (640, 240), "white")
    ImageDraw.Draw(img).multiline_text((24, 40), text, fill="black", font=ImageFont.load_default(size=34), spacing=16)
    return img


CASES = [
    ("gauge", "Read the pressure gauge in the photo. Operating limit is 8 bar.",
     [("continue", "Continue normal operation."), ("stop", "Stop the pump immediately: pressure is over the limit.")],
     gauge(3), gauge(9.3)),
    ("ocr-invoice", "Route this scanned invoice.",
     [("pay", "Schedule for payment: the invoice is within the approval limit."),
      ("escalate", "Escalate to the CFO: the total exceeds EUR 10,000.")],
     label("INVOICE 2026-118\nACME S.p.A.\nTOTAL: EUR 740.00"), label("INVOICE 2026-119\nACME S.p.A.\nTOTAL: EUR 48,900.00")),
    ("before-after", "Compare the maintenance photos.",
     [("improved", "The second image shows fewer defects."), ("degraded", "The second image shows more defects.")],
     None, None),
]


def request(instructions, options, images, abstain=False):
    return {"input": [*({"type": "image", "label": lab, "image": b64(img)} for lab, img in images),
                      {"type": "json", "value": {"machine": "P-17", "operating_hours": 2300}}],
            "decision": {"id": "smoke", "instructions": instructions, "abstain": abstain,
                         "options": [{"id": i, "description": d} for i, d in options]}}


def spots(n: int) -> Image.Image:
    img = Image.new("RGB", (448, 448), (200, 200, 205))
    d = ImageDraw.Draw(img)
    for k in range(n):
        x, y = 40 + (k * 97) % 360, 40 + (k * 61) % 360
        d.ellipse((x, y, x + 40, y + 40), fill=(120, 60, 20))
    return img


if __name__ == "__main__":
    engine = load_engine()
    for name, instructions, options, a, b in CASES:
        if name == "before-after":
            runs = [[("Image 1: previous inspection", spots(9)), ("Image 2: current inspection", spots(1))],
                    [("Image 1: previous inspection", spots(1)), ("Image 2: current inspection", spots(9))]]
        else:
            runs = [[("Photo", a)], [("Photo", b)]]
        for abstain in (False, True):
            got = [decide(engine, request(instructions, options, imgs, abstain)) for imgs in runs]
            print(json.dumps({"case": name, "abstain": abstain,
                              "runs": [{"decision": g["decision"], "raw": {k: round(v, 3) for k, v in g["raw_probabilities"].items()},
                                        "latency_ms": g["latency_ms"], "input_tokens": g["usage"]["input_tokens"]}
                                       for g in got]}))
