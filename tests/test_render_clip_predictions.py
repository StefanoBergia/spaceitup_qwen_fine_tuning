"""Layout helpers of scripts/render_clip_predictions.py: the reasoning column is wrapped by
measured pixel width and the canvas is sized once per clip, because ffmpeg is fed raw
frames and every frame of a clip must have the same dimensions."""

import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _mod():
    spec = importlib.util.spec_from_file_location(
        "render_clip_predictions",
        Path(__file__).resolve().parent.parent / "scripts" / "render_clip_predictions.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["render_clip_predictions"] = m
    spec.loader.exec_module(m)
    return m


def test_wrap_trace_respects_pixel_width_and_paragraphs():
    rv = _mod()
    font = ImageFont.load_default(size=11)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = rv.wrap_trace("one two three four five six seven eight nine ten", font, 90, d)
    assert len(lines) > 1
    assert all(d.textlength(l, font=font) <= 90 for l in lines)
    assert " ".join(lines) == "one two three four five six seven eight nine ten"
    assert rv.wrap_trace("a\nb", font, 500, d) == ["a", "b"]
    assert rv.wrap_trace("", font, 500, d) == ["(no reasoning)"]


def test_trace_lines_are_capped():
    rv = _mod()
    font = ImageFont.load_default(size=11)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = rv.trace_lines("word " * 5000, font, d)
    assert len(lines) == rv.MAX_TRACE_LINES + 1 and lines[-1] == "..."


def test_compose_honours_a_fixed_height_and_even_dims(tmp_path):
    rv = _mod()
    img = tmp_path / "f.png"
    Image.new("RGB", (640, 360), (60, 60, 60)).save(img)
    gt = {"path": [[0.5, 1.0, 1], [0.5, 0.5, 1]], "goal": [0.5, 0.5, 1]}
    pred = {"path": [[0.5, 1.0, 1], [0.6, 0.5, 1]], "goal": [0.6, 0.5, 1]}
    fonts = tuple(ImageFont.load_default(size=s) for s in (13, 12, 11))
    tags, colours = ["a", "b"], {"a": (255, 0, 0), "b": (0, 0, 255)}
    rec = {"id": "x", "image": str(img), "gt": gt}
    short = rv.compose(rec, {"a": pred, "b": pred}, tags, colours, ["one line"], fonts,
                       {"a": 0.1, "b": None})
    tall = rv.compose(rec, {"a": pred, "b": None}, tags, colours, ["l"] * 40, fonts,
                      {"a": 0.1, "b": None}, height=short.height + 300)
    assert short.width % 2 == 0 and short.height % 2 == 0
    assert tall.width == short.width and tall.height == short.height + 300
    # no reasoning column when there is no trace
    assert rv.compose(rec, {"a": pred}, ["a"], colours, None, fonts, {"a": 0.1}).width < short.width
