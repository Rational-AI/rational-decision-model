import base64
import io

import pytest
from PIL import Image

from rmdm.server import ABSTAIN, BadRequest, Images, decision_options


class FakeEngine:
    image_marker = "<|image|>"


def png(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_references_become_markers_in_text_order():
    images = Images(FakeEngine(), {"cat": png("red"), "category-photo": png("blue")})
    out = []
    text = images.resolve("A category-photo then a cat.", out)
    assert text == "A <|image|> then a <|image|>."  # the longer reference wins, matches never overlap
    assert [i.getpixel((0, 0)) for i in out] == [(0, 0, 255), (255, 0, 0)]
    assert images.used == {"cat", "category-photo"}


def test_caller_text_cannot_smuggle_image_markers():
    out = []
    assert Images(FakeEngine(), {}).resolve("look <|image|> here", out) == "look  here" and out == []


def test_bad_assets_are_rejected():
    with pytest.raises(BadRequest):
        Images(FakeEngine(), {"x": "not base64!"})
    with pytest.raises(BadRequest):
        Images(FakeEngine(), {" ": png("red")})


def test_noul_and_score_options():
    assert [o["id"] for o in decision_options({"type": "noul"})] == ["true", "false"]
    noul = decision_options({"type": "noul", "criteria": {"true": "Assembled correctly."}, "abstain": True})
    assert noul[0]["description"] == "Assembled correctly." and noul[1]["description"] == "No." and noul[2] is ABSTAIN
    levels = [{"id": i, "description": i} for i in ("low", "high")]
    assert decision_options({"type": "score", "options": levels}) == levels
    with pytest.raises(BadRequest):
        decision_options({"type": "box"})
