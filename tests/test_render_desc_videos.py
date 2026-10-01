"""render_desc_videos: frame picking and panel layout (ffmpeg itself is not exercised)."""
import importlib.util
from pathlib import Path

from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "render_desc_videos", Path(__file__).resolve().parent.parent / "scripts" / "render_desc_videos.py")
rdv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rdv)


def _row(rid, pair, img):
    return {"id": rid, "parsed": {"path": [[0.2, 1.0, 1], [0.5, 0.6, 1]]}, "gt": {"path": [[0.2, 1.0, 1], [0.4, 0.6, 1]]},
            "habitat_meta": {"pair_id": pair, "instruction": "Go left of the sofa and stop.", "image": str(img)}}


def test_pick_frames_groups_pairs_first(tmp_path):
    rows = [_row("s", None, "x"), _row("p_L", "p", "x"), _row("p_R", "p", "x"), _row("t", None, "x")]
    frames = rdv.pick_frames(rows, 2)
    assert [[r["id"] for r in f] for f in frames] == [["p_L", "p_R"], ["s"]]


def test_panel_has_one_tile_per_row(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (512, 512), (90, 90, 90)).save(p)
    one = rdv.panel(p, [_row("s", None, p)], px=200)
    two = rdv.panel(p, [_row("a", "p", p), _row("b", "p", p)], px=200)
    assert two.width == 2 * one.width and one.height == two.height and one.height > 200
