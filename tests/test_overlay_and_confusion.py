import base64

from PIL import Image, ImageDraw

from rover_vlm.eval import goal_visibility_confusion
from rover_vlm.overlay import (
    OBS_RED,
    VIS_GREEN,
    draw_goal,
    draw_path,
    draw_polyline,
    embed_jpeg,
    side_by_side,
)


# --- goal-visibility confusion ------------------------------------------------------


def _rec(gt_v, pred_v):
    return {"gt": {"goal": [0.5, 0.5, gt_v]}, "parsed": {"goal": [0.5, 0.5, pred_v]}}


def test_confusion_perfect_predictions():
    c = goal_visibility_confusion([_rec(1, 1), _rec(0, 0)])
    assert c["recall_visible"] == 1.0 and c["recall_obstructed"] == 1.0
    assert c["balanced_accuracy"] == 1.0


def test_majority_class_collapse_scores_half():
    """The whole point of balanced accuracy: always answering the majority class
    scores its base rate on raw accuracy but exactly 0.5 balanced."""
    recs = [_rec(0, 0)] * 75 + [_rec(1, 0)] * 25     # always says "obstructed"
    c = goal_visibility_confusion(recs)
    assert c["recall_obstructed"] == 1.0
    assert c["recall_visible"] == 0.0
    assert c["balanced_accuracy"] == 0.5
    assert (c["n_visible"], c["n_obstructed"]) == (25, 75)


def test_opposite_collapse_also_scores_half():
    c = goal_visibility_confusion([_rec(0, 1)] * 75 + [_rec(1, 1)] * 25)
    assert c["balanced_accuracy"] == 0.5


def test_confusion_skips_unparseable():
    recs = [_rec(1, 1), {"gt": {"goal": [0.5, 0.5, 0]}, "parsed": None}]
    c = goal_visibility_confusion(recs)
    assert c["n_obstructed"] == 0 and c["n_visible"] == 1


def test_confusion_handles_empty_input():
    c = goal_visibility_confusion([])
    assert c["balanced_accuracy"] == 0.0


def test_confusion_absent_class_does_not_crash():
    c = goal_visibility_confusion([_rec(0, 0)] * 5)     # no visible goals at all
    assert c["recall_obstructed"] == 1.0
    assert c["recall_visible"] == 0.0
    assert c["n_visible"] == 0


# --- overlay drawing ----------------------------------------------------------------


def _canvas(w=60, h=60, color=(0, 0, 0)):
    img = Image.new("RGB", (w, h), color)
    return img, ImageDraw.Draw(img)


def test_draw_polyline_marks_pixels_and_returns_pixel_coords():
    img, d = _canvas()
    xy = draw_polyline(d, [(0.1, 0.5), (0.9, 0.5)], (255, 0, 0), 60, 60)
    assert xy == [(6.0, 30.0), (54.0, 30.0)]
    assert img.getpixel((30, 30)) == (255, 0, 0)


def test_draw_polyline_lays_white_underlay_around_the_stroke():
    img, d = _canvas()
    draw_polyline(d, [(0.1, 0.5), (0.9, 0.5)], (255, 0, 0), 60, 60, width=4)
    # centre is the colour, the halo just outside it is white
    assert img.getpixel((30, 30)) == (255, 0, 0)
    assert img.getpixel((30, 33)) == (255, 255, 255)


def test_draw_polyline_ignores_single_point():
    img, d = _canvas()
    draw_polyline(d, [(0.5, 0.5)], (255, 0, 0), 60, 60)
    assert img.getpixel((30, 30)) == (0, 0, 0)


def test_visible_waypoint_is_filled_and_obstructed_is_hollow():
    # space the markers well beyond 2r so neither one's outline lands on the other
    vis_img, vd = _canvas(120, 60)
    draw_path(vd, [(0.2, 0.5, 1), (0.8, 0.5, 1)], (255, 0, 0), 120, 60, r=6)
    obs_img, od = _canvas(120, 60)
    draw_path(od, [(0.2, 0.5, 0), (0.8, 0.5, 0)], (255, 0, 0), 120, 60, r=6)
    # the marker centre carries the encoding: filled = visible, white = obstructed
    assert vis_img.getpixel((24, 30)) == (255, 0, 0)
    assert obs_img.getpixel((24, 30)) == (255, 255, 255)
    # ...and the hollow marker still has a coloured ring, so it is not a blank gap
    assert obs_img.getpixel((24 - 6, 30)) == (255, 0, 0)


def test_goal_marker_colour_follows_predicted_visibility():
    a, da = _canvas(); draw_goal(da, [0.5, 0.5, 1], 60, 60)
    b, db = _canvas(); draw_goal(db, [0.5, 0.5, 0], 60, 60)
    assert a.getpixel((30, 30)) == VIS_GREEN
    assert b.getpixel((30, 30)) == OBS_RED


def test_side_by_side_widths_add_with_the_gutter():
    out = side_by_side(Image.new("RGB", (40, 30)), Image.new("RGB", (50, 30)), gap=8)
    assert out.size == (98, 30)


def test_embed_jpeg_is_a_self_contained_data_uri():
    uri = embed_jpeg(Image.new("RGB", (900, 900), (10, 20, 30)), max_px=64)
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"                       # JPEG magic
    assert max(Image.open(__import__("io").BytesIO(raw)).size) == 64
