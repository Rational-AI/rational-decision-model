"""A bounding box from the same one-token readout: ask for each edge on a gridded picture, then zoom in and ask again.

Level by level, the picture (or the region found so far) gets a grid of numbered columns and lettered rows drawn
in a margin around it, and four lettered-option questions ask which column holds the target's leftmost and
rightmost parts and which row its top and bottom. The four questions share the gridded picture, so after the first
one the prefix cache makes them cheap. A first question asks whether the target is there at all, which a box
alone cannot say.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .engine import Engine, decision_text
from .readout import LABELS, distribution

GRID = 8
LEVELS = 2
MIN_SIDE = 896  # small crops are scaled up so the model sees the zoomed region in detail
MARGIN = 40


def gridded(image: Image.Image, n: int = GRID) -> Image.Image:
    """The image inside a white margin, with n columns numbered 1..n along the top and n rows lettered A.. on the
    left, and thin grid lines across the image."""
    w, h = image.size
    out = Image.new("RGB", (w + MARGIN, h + MARGIN), "white")
    out.paste(image, (MARGIN, MARGIN))
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=MARGIN // 2)
    for i in range(n):
        x0, y0 = MARGIN + w * i / n, MARGIN + h * i / n
        d.text((x0 + w / n / 2, MARGIN / 2), str(i + 1), fill="black", font=font, anchor="mm")
        d.text((MARGIN / 2, y0 + h / n / 2), LABELS[i], fill="black", font=font, anchor="mm")
        if i:
            d.line((x0, MARGIN, x0, MARGIN + h), fill=(255, 0, 255), width=2)
            d.line((MARGIN, y0, MARGIN + w, y0), fill=(255, 0, 255), width=2)
    return out


def locate(engine: Engine, image: Image.Image, target: str, temperature: float) -> dict:
    """{"found": P(visible), "box": {left, top, right, bottom} as fractions of the image, "confidence", "steps"}."""
    region = (0.0, 0.0, 1.0, 1.0)  # left, top, right, bottom, as fractions of the full image
    confidence, steps, found = 1.0, 0, None
    W, H = image.size
    for level in range(LEVELS):
        left, top, right, bottom = region
        crop = image.crop((round(left * W), round(top * H), round(right * W), round(bottom * H)))
        scale = max(1.0, MIN_SIDE / max(crop.size))
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        picture = gridded(crop)
        state = (f"{engine.image_marker}\nThe picture has a grid: {GRID} numbered columns (1 is the leftmost) and "
                 f"{GRID} lettered rows (A is the top).")

        def ask(question, descriptions):
            nonlocal steps
            steps += 1
            scores, _ = engine.scores(decision_text(state, question, descriptions), [picture], len(descriptions),
                                      shared_chars=len(state) + 2)
            p = distribution(scores, temperature)
            best = max(range(len(p)), key=p.__getitem__)
            return best, p[best]

        if level == 0:
            yes, p = ask(f"Is {target} visible in the picture?", ["Yes.", "No."])
            found = p if yes == 0 else 1 - p
        columns = [f"Column {i + 1}." for i in range(GRID)]
        rows = [f"Row {LABELS[i]}." for i in range(GRID)]
        edges = [ask(f"Which column holds the leftmost part of {target}?", columns),
                 ask(f"Which column holds the rightmost part of {target}?", columns),
                 ask(f"Which row holds the topmost part of {target}?", rows),
                 ask(f"Which row holds the bottommost part of {target}?", rows)]
        (c0, _), (c1, _), (r0, _), (r1, _) = edges
        c0, c1 = sorted((c0, c1))
        r0, r1 = sorted((r0, r1))
        confidence = min(confidence, *(p for _, p in edges))
        cw, rh = (right - left) / GRID, (bottom - top) / GRID
        if level == LEVELS - 1:
            # an edge lies somewhere in its cell: the cell's middle is the unbiased estimate
            region = (left + (c0 + 0.5) * cw, top + (r0 + 0.5) * rh, left + (c1 + 0.5) * cw, top + (r1 + 0.5) * rh)
        else:
            # zoom on the cells found plus one cell all round, so an edge in a neighbouring cell is not lost
            region = (max(0.0, left + (c0 - 1) * cw), max(0.0, top + (r0 - 1) * rh),
                      min(1.0, left + (c1 + 2) * cw), min(1.0, top + (r1 + 2) * rh))
    return {"found": found, "box": dict(zip(("left", "top", "right", "bottom"), region)),
            "confidence": confidence, "steps": steps}
