"""Route events (prompt d1): geometry on synthetic occupancy grids + real-frame regressions.

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (LEFT below).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rover_vlm.landmarks import FREE, OCC
from rover_vlm.route_events import (
    NEAR_CAMERA_M, candidates, first_blocker, load_frame, route_geometry, visible_cells,
)

LEFT = np.array([1.0, 0.0])


def _free():
    return np.full((100, 100), FREE, np.uint8)


def write_sample(d, state, routes, *, pose=(2.5, 0.2), heading=(0.0, 1.0), depth=None):
    """A synthetic sample dir: state.npz, meta.json (one candidate per route), optional depth."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    meta = {"pose_xz": list(pose), "heading_xz": list(heading), "flavor": "detour",
            "chord": {"crossing_xz": None}, "candidates": cands}
    (d / "meta.json").write_text(json.dumps(meta))
    if depth is not None:
        Image.fromarray(depth.astype(np.uint8), "L").save(d / "depth.png")
    return d


# S-route: bows to +x (left) past box A, then to -x (right) past box B
S_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (1.8, 2.4), (1.8, 3.0), (2.5, 3.6)]


def _s_state():
    s = _free()
    s[20:30, 52:58] = OCC        # box A: x 2.6..2.9, z 1.0..1.5 (on the route's right there)
    s[50:58, 46:52] = OCC        # box B: x 2.3..2.6, z 2.5..2.9 (on the route's left there; >= TOUCH_M off the route)
    return s


def test_route_geometry_right_vector_points_right_of_travel():
    rg = route_geometry([(0.0, 0.0), (0.0, 2.0)])
    assert np.allclose(rg["right"][5], [-1.0, 0.0])      # travel +z -> right is -x (Camera.right)
    assert rg["arc"][-1] == pytest.approx(2.0)
    assert rg["offset"].max() == pytest.approx(0.0)


def test_each_obstacle_gets_its_own_side_on_an_s_route(tmp_path):
    _, frame = load_frame(write_sample(tmp_path / "s", _s_state(), [S_ROUTE]))
    cs = candidates(frame, np.array(S_ROUTE))
    sides = [c["side"] for c in cs]
    assert sides == ["left", "right"]                    # route order: A then B
    assert all(c["pass_m"] <= 0.7 for c in cs)
    assert cs[0]["outline"]["area"] >= 0.01 and len(cs[0]["outline"]["polygon"]) >= 3
    assert cs[0]["score"] > 0


def test_obstacle_beyond_pass_distance_is_not_a_candidate(tmp_path):
    s = _free()
    s[20:30, 52:58] = OCC                                # box A as above
    route = [(2.5, 0.2), (1.6, 0.6), (1.6, 3.6)]   # stays >= 0.78 m from box A after the 0.3 m start skip
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    assert candidates(frame, np.array(route)) == []


def test_occupied_cells_at_the_camera_do_not_hide_what_is_behind(tmp_path):
    # 00372 (2026-09-30): a strip 0.31 m ahead of the rover, bare floor in the image, was
    # the first occupied cell on every ray and hid the armchair.
    s = _s_state()
    s[9:11, 46:54] = OCC                                 # strip z 0.45..0.55, 0.25-0.35 m ahead
    _, frame = load_frame(write_sample(tmp_path / "s", s, [S_ROUTE]))
    target = frame.grid.xz([[25, 55]])[0]                # a cell of box A
    assert first_blocker(frame, target) == int(frame.lab[25, 55])
    cs = candidates(frame, np.array(S_ROUTE))
    assert [c["side"] for c in cs] == ["left", "right"]
    strip = int(frame.lab[10, 50])
    assert strip not in {c["blob"] for c in cs}


def test_hidden_cells_are_not_visible(tmp_path):
    s = _free()
    s[20:22, 40:60] = OCC                                # screen x 2.0..3.0 at z 1.0..1.1
    s[40:44, 48:52] = OCC                                # post behind it, x 2.4..2.6, z 2.0..2.2
    _, frame = load_frame(write_sample(tmp_path / "s", s, [S_ROUTE]))
    post = np.argwhere(frame.lab == frame.lab[41, 49])
    assert len(visible_cells(frame, post, int(frame.lab[41, 49]))) == 0


def test_near_camera_constant():
    assert NEAR_CAMERA_M == pytest.approx(0.5)


from PIL import ImageDraw  # noqa: E402

from rover_vlm.route_events import calibrate_depth, column_outline, object_outline  # noqa: E402

LO, HI = 0.5, 6.0     # the synthetic frame's min-max range (the generator does not store it)


def _depth_image(cam, obj_poly_uv, obj_fwd, H=128, W=128):
    """Planar depth of a level camera over a flat floor, far wall at HI above the horizon,
    one object region at obj_fwd; min-max normalised to 0..255 like the generator."""
    v = (np.arange(H) + 0.5) / H
    floor = np.where(v > 0.5, 0.5 * 0.8 / np.maximum(v - 0.5, 1e-6), HI)
    m = np.repeat(np.minimum(floor, HI)[:, None], W, 1)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(u * W, q * H) for u, q in obj_poly_uv], fill=1)
    m[np.asarray(mask, bool)] = obj_fwd
    return np.round((m - LO) / (HI - LO) * 255)


def _box_frame(tmp_path, with_depth=True):
    s = _free()
    s[28:34, 48:54] = OCC                        # box x 2.4..2.7, z 1.4..1.7 (1.2-1.5 m ahead)
    route = [(2.5, 0.2), (3.2, 1.2), (3.2, 1.9), (2.5, 3.0)]
    d = write_sample(tmp_path / "noimg", s, [route])
    _, f0 = load_frame(d)
    vis = visible_cells(f0, np.argwhere(f0.lab == f0.lab[30, 50]), int(f0.lab[30, 50]))
    depth = None
    if with_depth:
        half = column_outline(f0, vis, top=0.5)["polygon"]   # the object is 0.5 m tall
        depth = _depth_image(f0.cam, half, 1.3)
    d = write_sample(tmp_path / ("d" if with_depth else "n"), s, [route], depth=depth)
    return load_frame(d)[1], vis


def test_calibration_recovers_the_frame_range(tmp_path):
    frame, _ = _box_frame(tmp_path)
    scale, offset, resid = frame.depth_fit
    assert scale == pytest.approx((HI - LO) / 255, rel=0.05)
    assert offset == pytest.approx(LO, abs=0.1)
    assert resid < 0.1


def test_trimmed_outline_drops_the_background_above_a_low_object(tmp_path):
    frame, vis = _box_frame(tmp_path)
    col = column_outline(frame, vis)
    trim = object_outline(frame, vis)
    assert trim["depth_fallback"] is False
    assert trim["bbox"][1] > col["bbox"][1] + 0.02        # top edge moved down: the far wall is gone
    assert trim["area"] < col["area"]


def test_outline_keeps_only_the_depth_piece_holding_the_object(tmp_path):
    """Two things at one depth, apart in the image; the blob's cells hull both (a long wall) but
    only the piece holding the cells' own pixels is the object."""
    s = _free()
    s[28:34, 48:54] = OCC                        # the object: x 2.4..2.7
    route = [(2.5, 0.2), (3.2, 1.2), (3.2, 1.9), (2.5, 3.0)]
    f0 = load_frame(write_sample(tmp_path / "noimg", s, [route]))[1]
    b1 = int(f0.lab[30, 50])
    vis1 = visible_cells(f0, np.argwhere(f0.lab == b1), b1)
    far = np.array([[1.5, 1.45]])                # a cell of the same blob, ~0.4 widths to the side
    vis = np.vstack([vis1, far])
    u_far = f0.cam.uv(far)[0][0, 0]
    box = column_outline(f0, vis1, top=0.5)["polygon"]
    block = [[u_far - 0.1, 0.55], [u_far - 0.04, 0.55], [u_far - 0.04, 0.66], [u_far - 0.1, 0.66]]
    cover = [[0.56, 0.66], [1, 0.66], [1, 1], [0.56, 1]]   # nearer stuff: no floor strip joins the pieces
    H = W = 128
    depth = _depth_image(f0.cam, cover, 0.7)
    for poly in (box, block):
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon([(u * W, v * H) for u, v in poly], fill=1)
        depth = np.where(np.asarray(m, bool), _depth_image(f0.cam, [[0, 0], [1, 0], [1, 1], [0, 1]], 1.3), depth)
    frame = load_frame(write_sample(tmp_path / "d", s, [route], depth=depth))[1]
    assert column_outline(frame, vis)["bbox"][2] > u_far - 0.01        # the column hull spans both
    trim = object_outline(frame, vis)
    assert trim["depth_fallback"] is False
    assert trim["bbox"][2] < u_far - 0.1                                # ... the trimmed one stops short of the block


def _two_things_frame(tmp_path, floor_between):
    """The box plus a second thing at the same depth, 0.4 widths aside, inside the hull of a
    blob that also owns a far cell whose own pixels hit nothing. floor_between=False paints a nearer
    region over the floor strip joining them (the pre-floor-removal workaround)."""
    s = _free()
    s[28:34, 48:54] = OCC
    route = [(2.5, 0.2), (3.2, 1.2), (3.2, 1.9), (2.5, 3.0)]
    f0 = load_frame(write_sample(tmp_path / "noimg", s, [route]))[1]
    b1 = int(f0.lab[30, 50])
    vis1 = visible_cells(f0, np.argwhere(f0.lab == b1), b1)
    far = np.array([[1.5, 1.45]])
    u_far = f0.cam.uv(far)[0][0, 0]
    box = column_outline(f0, vis1, top=0.5)["polygon"]
    block = [[u_far - 0.1, 0.55], [u_far - 0.04, 0.55], [u_far - 0.04, 0.66], [u_far - 0.1, 0.66]]
    depth = _depth_image(f0.cam, [[0.56, 0.66], [1, 0.66], [1, 1], [0.56, 1]] if not floor_between else box, 0.7 if not floor_between else 1.3)  # noqa: E501
    for poly in (box, block):
        m = Image.new("L", (128, 128), 0)
        ImageDraw.Draw(m).polygon([(u * 128, v * 128) for u, v in poly], fill=1)
        depth = np.where(np.asarray(m, bool), _depth_image(f0.cam, [[0, 0], [1, 0], [1, 1], [0, 1]], 1.3), depth)
    frame = load_frame(write_sample(tmp_path / "d", s, [route], depth=depth))[1]
    return frame, np.vstack([vis1, far]), u_far


def test_outline_excludes_a_second_object_joined_only_by_floor(tmp_path):
    frame, vis, u_far = _two_things_frame(tmp_path, floor_between=True)
    assert column_outline(frame, vis)["bbox"][2] > u_far - 0.01
    trim = object_outline(frame, vis)
    assert trim["depth_fallback"] is False
    assert trim["bbox"][2] < u_far - 0.1


def test_outline_with_no_seeded_piece_hulls_the_floor_free_band(tmp_path):
    frame, vis, _ = _two_things_frame(tmp_path, floor_between=True)
    # a very low object: its pixels hold none of the seeds (heights min(top, 0.25), top/2, top)
    low = column_outline(frame, vis, top=0.1)["polygon"]
    m = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(m).polygon([(u * 128, v * 128) for u, v in low], fill=1)
    depth = _depth_image(frame.cam, [[0, 0], [1, 0], [1, 1], [0, 1]], 1.3)
    far_floor = _depth_image(frame.cam, [[0, 0], [0, 0], [0, 0]], 1.3)
    frame.depth = np.where(np.asarray(m, bool), depth, far_floor)
    frame.depth_fit = calibrate_depth(frame.depth, frame.cam, frame.grid) or frame.depth_fit
    o = object_outline(frame, vis)
    # final review I2: not the column outline (the group-B shape) but the band with the floor cut
    assert o["depth_fallback"] is False
    assert o["area"] < column_outline(frame, vis)["area"]


def test_outline_with_only_floor_in_the_band_falls_back_to_the_column(tmp_path):
    frame, vis = _box_frame(tmp_path)
    frame.depth = _depth_image(frame.cam, [[0, 0], [0, 0], [0, 0]], 1.3)   # bare floor + far wall
    frame.depth_fit = calibrate_depth(frame.depth, frame.cam, frame.grid)
    o = object_outline(frame, vis)
    assert o["depth_fallback"] is True and o["polygon"] == column_outline(frame, vis)["polygon"]


def test_low_object_below_half_the_outline_top_gets_a_trimmed_outline(tmp_path):
    # final review I2: a 0.3 m tall object ~1.5 m ahead -- the top/2 and top seeds land on the far wall
    s = _free()
    s[30:36, 48:54] = OCC                        # box x 2.4..2.7, z 1.5..1.8 (1.3-1.6 m ahead)
    route = [(2.5, 0.2), (3.2, 1.2), (3.2, 1.9), (2.5, 3.0)]
    f0 = load_frame(write_sample(tmp_path / "noimg", s, [route]))[1]
    b = int(f0.lab[32, 50])
    vis = visible_cells(f0, np.argwhere(f0.lab == b), b)
    shape = column_outline(f0, vis, top=0.3)["polygon"]
    frame = load_frame(write_sample(tmp_path / "d", s, [route], depth=_depth_image(f0.cam, shape, 1.5)))[1]
    col, o = column_outline(frame, vis), object_outline(frame, vis)
    assert o["depth_fallback"] is False
    assert o["bbox"][1] > col["bbox"][1] + 0.05 and o["area"] < 0.6 * col["area"]


def test_no_depth_png_falls_back_to_the_column_outline(tmp_path):
    frame, vis = _box_frame(tmp_path, with_depth=False)
    assert frame.depth is None and frame.depth_fit is None
    o = object_outline(frame, vis)
    assert o["depth_fallback"] is True and o["polygon"] == column_outline(frame, vis)["polygon"]


def test_poor_fit_falls_back(tmp_path):
    frame, vis = _box_frame(tmp_path)
    frame.depth = np.random.default_rng(0).uniform(0, 255, frame.depth.shape)
    frame.depth_fit = calibrate_depth(frame.depth, frame.cam, frame.grid)
    assert frame.depth_fit is None or frame.depth_fit[2] > 0.3
    assert object_outline(frame, vis)["depth_fallback"] is True


from rover_vlm.route_events import (  # noqa: E402
    MAX_EVENTS, build_events, collect_objects, describe_route, describe_xz, detect_turn, stop_phrase,
)


def test_opposite_sides_close_along_the_route_merge_into_between(tmp_path):
    s = _free()
    s[40:46, 62:68] = OCC        # x 3.1..3.4, z 2.0..2.3 : route's left there (0.625 m)
    s[40:46, 32:38] = OCC        # x 1.6..1.9, z 2.0..2.3 : route's right there (0.625 m)
    route = [(2.5, 0.2), (2.5, 1.5), (2.5, 2.8), (2.6, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert [e["type"] for e in ev] == ["between"] and ev[0]["side"] is None
    assert len(ev[0]["members"]) == 2


def _doorway_state(gap_cols):
    """Wall z 2.0..2.1 with a doorway; side/back walls join the halves into one blob."""
    s = _free()
    g0, g1 = gap_cols
    s[40:42, 10:g0] = OCC
    s[40:42, g1:90] = OCC
    s[10:42, 8:10] = OCC         # side walls joining both halves into one blob
    s[10:42, 90:92] = OCC
    s[8:10, 8:92] = OCC          # back wall z 0.4..0.5
    return s


def test_doorway_in_one_wall_is_a_through_event(tmp_path):
    # doorway x 2.2..2.8 (cell centres 2.175 / 2.825: 0.65 m, wider than MIN_GAP_M)
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    d = write_sample(tmp_path / "s", _doorway_state((44, 56)), [route], pose=(2.5, 0.8))
    _, frame = load_frame(d)
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert [e["type"] for e in ev] == ["through"]
    p, q = ev[0]["opening"]["gap_xz"]
    assert abs(p[0] - q[0]) == pytest.approx(0.65, abs=0.1)


def test_gap_narrower_than_the_rover_is_not_a_doorway(tmp_path):
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    d = write_sample(tmp_path / "s", _doorway_state((47, 53)), [route], pose=(2.5, 0.8))   # 0.35 m
    _, frame = load_frame(d)
    assert "through" not in [e["type"] for e in build_events(frame, candidates(frame, np.array(route)))]


def test_wall_straight_ahead_that_the_route_stops_short_of_is_not_a_doorway(tmp_path):
    s = _free()
    s[60:62, 5:95] = OCC                                 # wall z 3.0..3.1 across the map
    route = [(2.5, 0.2), (2.5, 2.5)]                     # ends 0.5 m before it
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert "through" not in [e["type"] for e in ev]


def test_long_wall_the_route_grazes_alongside_is_not_a_doorway(tmp_path):
    s = _free()
    s[10:90, 54:56] = OCC                                # wall x 2.7..2.8 along z, 0.25 m off the route
    route = [(2.5, 0.2), (2.5, 4.5)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert "through" not in [e["type"] for e in ev]


def test_two_doorways_of_one_wall_get_different_object_ids(tmp_path):
    s = _doorway_state((44, 56))                         # doorway A at x 2.2..2.8
    s[40:42, 62:74] = FREE                               # doorway B at x 3.1..3.7
    s[42:80, 58:60] = OCC                                # spine x 2.9..3.0 + arch z 3.9..4.0 join the middle piece to the rest
    s[78:80, 58:92] = OCC
    s[42:78, 90:92] = OCC
    r1 = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    r2 = [(2.5, 0.8), (3.4, 1.5), (3.4, 2.05), (3.4, 3.6)]
    d = write_sample(tmp_path / "s00002", s, [r1, r2], pose=(2.5, 0.8))
    ids = []
    for k in (0, 1):
        out = describe_route(d, k)
        assert out["reason"] is None, out
        th = [e for e in out["events"] if e["type"] == "through"]
        assert len(th) == 1, out["events"]
        ids.append(th[0]["objects"][0])
    assert ids[0] != ids[1] and all(i.startswith("s00002:gap") for i in ids)


def test_top_three_by_score_then_route_order(tmp_path):
    s = _free()
    for z0 in (16, 30, 44, 58, 72):                      # five posts, alternating sides
        x0 = 56 if (z0 // 14) % 2 else 40
        s[z0:z0 + 4, x0:x0 + 4] = OCC
    route = [(2.5, 0.2), (2.5, 4.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert len(ev) <= MAX_EVENTS
    assert [e["arc_m"] for e in ev] == sorted(e["arc_m"] for e in ev)


def _stop_state():
    s = _free()
    s[20:30, 52:58] = OCC                                # box A (passed on the left)
    s[70:80, 36:60] = OCC                                # cabinet x 1.8..3.0, z 3.5..4.0 (>= 1% of the image)
    return s


STOP_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (2.5, 2.9)]   # ends 0.6 m before the cabinet


def test_stop_near_an_object_ahead_is_in_front_of(tmp_path):
    route = STOP_ROUTE
    _, frame = load_frame(write_sample(tmp_path / "s", _stop_state(), [route]))
    st = stop_phrase(frame, np.array(route), [])
    assert st["type"] == "near" and st["relation"] == "in front of"


def test_stop_just_after_the_last_event(tmp_path):
    s = _free()
    s[30:40, 52:58] = OCC                             # box A: x 2.6..2.9, z 1.5..2.0
    route = [(2.5, 0.2), (3.3, 1.0), (3.7, 2.7)]      # goal 1.15 m from box A, 1.35 m of route after its closest approach
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert stop_phrase(frame, np.array(route), ev)["type"] == "after_last"


def test_stop_at_a_rough_distance_before_an_object_ahead(tmp_path):
    s = _free()
    s[90:94, 30:70] = OCC                                # sideboard x 1.5..3.5, z 4.5..4.7
    route = [(2.5, 0.2), (2.5, 2.5)]                     # goal 2.0 m short of it, nothing passed
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    st = stop_phrase(frame, np.array(route), [])
    assert st["type"] == "distance" and st["relation"] == "before" and st["distance_m"] == 2.0


def test_open_floor_goal_with_nothing_around_has_no_stop(tmp_path):
    route = [(2.5, 0.2), (2.5, 2.5)]
    _, frame = load_frame(write_sample(tmp_path / "s", _free(), [route]))
    assert stop_phrase(frame, np.array(route), []) is None


def test_describe_route_rows_and_object_ids(tmp_path):
    route = STOP_ROUTE
    d = write_sample(tmp_path / "s00001", _stop_state(), [route])
    out = describe_route(d, 0)
    assert out["reason"] is None
    # Addendum A: box A (passed on the left) and the cabinet (passed on the right) are adjacent,
    # opposite and < MERGE_ARC_M apart -> one "between" of both (was pass-left + pass)
    assert out["events"][0]["type"] == "between" and out["events"][0]["side"] is None
    assert len(out["events"][0]["objects"]) == 2
    assert out["stop"]["amount"] is None and out["stop"]["relation"] == "in front of"
    oid = out["events"][0]["objects"][0]
    assert oid.startswith("s00001:b") and out["objects"][oid]["kind_hint"] == "object"
    assert out["stop"]["object"] in out["objects"]
    assert 4 <= len(out["objects"][oid]["cells_xz"]) <= 40
    assert describe_route(d, 5)["reason"] == "bad_candidate"
    (d / "state.npz").unlink()
    assert describe_route(d, 0)["reason"] == "no_state"


def test_collect_objects_dedupes_shared_objects():
    o = {"object_id": "f:b1", "polygon": [], "kind_hint": "object"}
    rows = [{"id": "f_L", "image": "img.png", "objects": {"f:b1": o}},
            {"id": "f_R", "image": "img.png", "objects": {"f:b1": dict(o, polygon=[[0, 0]])}}]
    kept, objects = collect_objects(rows)
    assert [r.get("objects") for r in kept] == [None, None]
    assert len(objects) == 1 and objects[0]["polygon"] == [] and objects[0]["image"] == "img.png"


# --- real frames (NFS); skipped where the dataset is not mounted --------------------------
AUG = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated_aug/train")
ORIG = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/train")
REAL = [
    # (sample dir, candidate, world xz of the obstacle a person names, audit group)
    (AUG / "00372-GMwtBqNLGBs/samples/00372-GMwtBqNLGBs_c030_y1", 0, (-0.91, 8.51), "C: armchair hidden by a strip at the camera"),
    (ORIG / "00206-uhkqDVMtEnn/samples/00206-uhkqDVMtEnn_c040", 4, (6.82, -1.21), "A: pillar, not the grazed wall"),
    (AUG / "00475-g7hUFVNac26/samples/00475-g7hUFVNac26_c013_y1", 2, (5.34, 1.97), "A: post on the line, not the sofa"),
]


@pytest.mark.parametrize("sd,cand,xz,why", REAL, ids=[r[3] for r in REAL])
def test_real_frame_names_the_obstacle_a_person_would(sd, cand, xz, why):
    if not (sd / "state.npz").exists():
        pytest.skip("dataset not mounted")
    out = describe_route(sd, cand)
    assert out["reason"] is None, out
    event_objects = [o for e in out["events"] for o in e["objects"]]
    near = [o for o in event_objects
            if np.linalg.norm(np.array(out["objects"][o]["cells_xz"]) - xz, axis=1).min() <= 0.4]
    assert near, f"{why}: events {out['events']}"
    if why.startswith("A:"):                             # a pillar / a post: objects, not walls
        assert any(out["objects"][o]["kind_hint"] == "object" for o in near), out["objects"]


import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
prepare_instr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_instr)


def test_event_row_carries_record_fields(tmp_path):
    route = STOP_ROUTE
    d = write_sample(tmp_path / "s1", _stop_state(), [route])
    rec = {"id": "s1_L", "image": [str(d / "fpv_enhanced.png")],
           "habitat_meta": {"candidate_index": 0, "kind": "instructed", "pair_id": None, "sample_id": "s1"}}
    row = prepare_instr.event_row(rec)
    assert row["id"] == "s1_L" and row["reason"] is None and row["objects"]
    assert row["image"].endswith("fpv_enhanced.png") and row["sample_id"] == "s1"
    rec["habitat_meta"]["candidate_index"] = 9
    assert prepare_instr.event_row(rec)["reason"] == "bad_candidate"


def test_stop_falls_back_to_distance_past_the_last_event(tmp_path):
    """Rough distance past the last thing passed, when nothing visible is near the goal."""
    s = _free()
    s[18:28, 52:58] = OCC                             # box A: x 2.6..2.9, z 0.9..1.4 (clear of the route's first leg)
    route = [(2.5, 0.2), (3.3, 0.8), (3.3, 2.0), (3.3, 3.0), (4.6, 3.0)]   # ends heading sideways, off the box's line
    d = write_sample(tmp_path / "s00002", s, [route])
    _, frame = load_frame(d)
    cands = candidates(frame, np.array(route))
    ev = build_events(frame, cands)
    st = stop_phrase(frame, np.array(route), ev, cands)
    assert st["type"] == "distance" and st["relation"] == "past"
    rem = route_geometry(np.array(route))["arc"][-1] - ev[-1]["arc_m"]
    # Addendum A: a coarse amount + the raw metres (info only), no longer the nearest metre step
    assert st["amount"] == ("a little" if rem < 3.0 else "well")
    assert st["distance_m"] == pytest.approx(rem, abs=0.051)
    out = describe_route(d, 0)
    assert out["reason"] is None and out["stop"]["type"] == "distance"
    assert out["stop"]["object"] == out["events"][-1]["objects"][0]
    assert len(out["objects"]) == 1


# --- Addendum A (Task 14): walls, between/pass merging, coarse stop amounts ---------------
from rover_vlm.route_events import (  # noqa: E402
    ALONG_MIN_M, AMOUNT_STEPS, MERGE_ARC_M, WALL_EXTENT_M, WALL_KEEP_M,
)


def test_addendum_constants():
    assert (ALONG_MIN_M, WALL_KEEP_M, MERGE_ARC_M) == (1.0, 0.8, 3.0)
    assert AMOUNT_STEPS == ((1.5, "just"), (3.0, "a little"), (float("inf"), "well"))


def _wall_left_state():
    s = _free()
    s[10:90, 68:70] = OCC                                # wall x 3.4..3.5, z 0.5..4.5
    return s


def test_long_wall_alongside_is_a_wall_event_on_the_rovers_side(tmp_path):
    route = [(2.5, 0.2), (2.9, 0.8), (2.9, 2.6)]          # runs alongside, ends well before the wall's end
    _, frame = load_frame(write_sample(tmp_path / "s", _wall_left_state(), [route]))
    cs = candidates(frame, np.array(route))
    assert len(cs) == 1 and cs[0]["wall"] is True and cs[0]["along_m"] >= ALONG_MIN_M
    assert cs[0]["side"] == "right"                      # the route passes it on its right ...
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["wall"] and ev[0]["side"] == "left"   # ... the wall is on the rover's left
    b = cs[0]["blob"]
    whole = column_outline(frame, visible_cells(frame, np.argwhere(frame.lab == b), b))
    o = cs[0]["outline"]
    assert o["bbox"][2] - o["bbox"][0] < whole["bbox"][2] - whole["bbox"][0]
    route_pts = route_geometry(route)["pts"]
    assert np.linalg.norm(cs[0]["vis_xz"][:, None] - route_pts[None], axis=2).min(1).max() <= WALL_KEEP_M + 1e-6


def test_s_route_opposite_passes_merge_into_one_between(tmp_path):
    _, frame = load_frame(write_sample(tmp_path / "s", _s_state(), [S_ROUTE]))
    cs = candidates(frame, np.array(S_ROUTE))
    assert abs(cs[1]["arc_m"] - cs[0]["arc_m"]) > 0.5     # far beyond the old 0.5 m pairing
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["between"] and ev[0]["side"] is None
    assert sorted(ev[0]["members"]) == [0, 1]
    assert ev[0]["score"] == pytest.approx(cs[0]["score"] + cs[1]["score"])


def _same_side_state():
    s = _free()
    s[30:36, 55:61] = OCC                                # box x 2.75..3.05, z 1.5..1.8
    s[52:58, 55:61] = OCC                                # box x 2.75..3.05, z 2.6..2.9 (~1.1 m on)
    return s


def test_two_objects_on_the_same_side_merge_into_one_pass(tmp_path):
    route = [(2.5, 0.2), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", _same_side_state(), [route]))
    cs = candidates(frame, np.array(route))
    assert [c["side"] for c in cs] == ["right", "right"] and not any(c["wall"] for c in cs)
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["pass"] and ev[0]["side"] == "right"
    assert sorted(ev[0]["members"]) == [0, 1]


def test_passes_farther_apart_than_merge_arc_stay_separate(tmp_path):
    s = _free()
    s[16:22, 55:61] = OCC                                # left of the route, z 0.8..1.1
    s[84:92, 32:44] = OCC                                # right of the route, x 1.6..2.2, z 4.2..4.6 (~3.4 m on)
    route = [(2.5, 0.2), (2.5, 4.8)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    assert len(cs) == 2 and cs[1]["arc_m"] - cs[0]["arc_m"] > MERGE_ARC_M
    assert [e["type"] for e in build_events(frame, cs)] == ["pass", "pass"]


def _wall_and_box_state():
    s = _wall_left_state()
    s[44:50, 38:44] = OCC                                # box x 1.9..2.2, z 2.2..2.5 (route's right)
    return s


WALL_BOX_ROUTE = [(2.5, 0.2), (2.8, 0.8), (2.8, 3.2)]


def test_wall_and_object_on_opposite_sides_are_between(tmp_path):
    d = write_sample(tmp_path / "s00003", _wall_and_box_state(), [WALL_BOX_ROUTE])
    _, frame = load_frame(d)
    cs = candidates(frame, np.array(WALL_BOX_ROUTE))
    assert sorted(c["wall"] for c in cs) == [False, True]
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["between"]
    out = describe_route(d, 0)
    assert out["reason"] is None, out
    (row,) = out["events"]
    assert row["type"] == "between" and row["side"] is None and len(row["objects"]) == 2
    assert sorted(out["objects"][o]["kind_hint"] for o in row["objects"]) == ["object", "wall"]


def test_doorway_wall_the_route_runs_alongside_stays_through(tmp_path):
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    d = write_sample(tmp_path / "s", _doorway_state((44, 56)), [route], pose=(2.5, 0.8))
    _, frame = load_frame(d)
    cs = candidates(frame, np.array(route))
    assert any(c["gap"] is not None and c["wall"] for c in cs)   # a wall by the wall rule ...
    assert [e["type"] for e in build_events(frame, cs)] == ["through"]   # ... but the gap wins


def test_two_separate_wall_halves_with_an_opening_are_through(tmp_path):
    s = _free()
    s[40:42, 10:44] = OCC                                # two unjoined halves, opening x 2.2..2.8
    s[40:42, 56:90] = OCC
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route], pose=(2.5, 0.8)))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert [e["type"] for e in ev] == ["through"]


def test_merging_never_joins_a_through_event(tmp_path):
    s = _doorway_state((44, 56))
    s[50:56, 56:62] = OCC                                # box x 2.8..3.1, z 2.5..2.8, just past the door
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route], pose=(2.5, 0.8)))
    ev = build_events(frame, candidates(frame, np.array(route)))
    types = [e["type"] for e in ev]
    assert "through" in types and "between" not in types
    assert all(len(e["members"]) == 1 for e in ev if e["type"] == "through")


def test_near_stop_with_a_dominating_lateral_offset_is_not_next_to(tmp_path):
    s = _free()
    s[20:30, 52:58] = OCC                                # box A x 2.6..2.9, z 1.0..1.5 (passed)
    s[46:50, 72:78] = OCC                                # box C x 3.6..3.9, z 2.3..2.5: beside the goal
    route = [(2.5, 0.2), (3.3, 1.0), (3.3, 2.4)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    st = stop_phrase(frame, np.array(route), build_events(frame, cs), cs)
    assert st is not None and st["type"] != "near" and st["relation"] != "next to"


def _long_state(nz=200):
    s = np.full((nz, 100), FREE, np.uint8)
    s[18:22, 54:58] = OCC                                # post x 2.7..2.9, z 0.9..1.1
    return s


@pytest.mark.parametrize("goal_z,amount", [(7.8, "well"), (3.2, "a little")])
def test_distance_past_the_last_event_uses_coarse_amounts(tmp_path, goal_z, amount):
    route = [(2.5, 0.2), (2.5, goal_z)]
    d = write_sample(tmp_path / "s00004", _long_state(), [route])
    _, frame = load_frame(d)
    cs = candidates(frame, np.array(route))
    ev = build_events(frame, cs)
    st = stop_phrase(frame, np.array(route), ev, cs)
    rem = route_geometry(np.array(route))["arc"][-1] - ev[-1]["arc_m"]
    assert st["type"] == "distance" and st["relation"] == "past" and st["amount"] == amount
    assert st["distance_m"] == pytest.approx(rem, abs=0.051)     # raw metres, rounded to 0.1 (info only)
    row = describe_route(d, 0)["stop"]
    assert set(row) == {"type", "relation", "amount", "distance_m", "object"}
    assert row["amount"] == amount and row["object"] is not None


def test_short_remaining_is_after_last_with_no_amount(tmp_path):
    route = [(2.5, 0.2), (2.5, 2.1)]                     # ~1.1 m after the post
    d = write_sample(tmp_path / "s00005", _long_state(100), [route])
    row = describe_route(d, 0)["stop"]
    assert row["type"] == "after_last" and row["amount"] is None and row["distance_m"] is None


def test_wall_like_anchor_just_past_the_goal_is_not_a_near_stop(tmp_path):
    s = _free()
    s[10:44, 40:42] = OCC                                # wall x 2.0..2.1, z 0.5..2.2 (1.7 m)
    route = [(2.5, 0.2), (2.5, 2.3), (1.6, 2.6)]         # rounds its end; the goal is just past it
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["wall"]
    # by design nothing fits: the wall is the only blob -- not "just past" it (near), not "just after"
    # it (the goal is still within PASS_M of it), not "past" it (the route never leaves it), and the
    # line-of-approach rule never stops a wall-like blob "past"
    assert stop_phrase(frame, np.array(route), ev, cs) is None


def test_wall_ahead_of_the_goal_is_in_front_of(tmp_path):
    s = _free()
    s[60:62, 5:95] = OCC                                 # wall z 3.0..3.1 across the map
    route = [(2.5, 0.2), (2.5, 2.5)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    st = stop_phrase(frame, np.array(route), [])
    assert st["type"] == "near" and st["relation"] == "in front of"
    assert np.hypot(*np.ptp(frame.grid.xz(np.argwhere(frame.lab == st["anchor"]["blob"])), axis=0)) >= WALL_EXTENT_M
    # the blob is also a pass member (the route comes within PASS_M of it, but not alongside):
    # its event role wins -> "object" (fix round 2 ruling)
    out = describe_route(write_sample(tmp_path / "s00007", s, [route]), 0)
    assert out["stop"]["relation"] == "in front of"
    assert [e["type"] for e in out["events"]] == ["pass"] and out["events"][0]["objects"] == [out["stop"]["object"]]
    assert out["objects"][out["stop"]["object"]]["kind_hint"] == "object"


def test_wall_like_stop_anchor_in_no_event_is_a_wall(tmp_path):
    s = _free()
    s[64:66, 5:95] = OCC                                 # wall z 3.2..3.3 across the map
    route = [(2.5, 0.2), (2.5, 2.45)]                    # ends 0.75 m before it: beyond PASS_M, within STOP_NEAR_M
    out = describe_route(write_sample(tmp_path / "s00008", s, [route]), 0)
    assert out["reason"] is None and out["events"] == []
    assert out["stop"]["type"] == "near" and out["stop"]["relation"] == "in front of"
    assert out["objects"][out["stop"]["object"]]["kind_hint"] == "wall"


# --- controller ruling on Task 14 concern 1: a wall anchors "past" only once the route has left it ---
def _short_wall_state(nz=200):
    s = np.full((nz, 100), FREE, np.uint8)
    s[10:50, 68:70] = OCC                                # wall x 3.4..3.5, z 0.5..2.5
    return s


@pytest.mark.parametrize("beyond,amount", [(2.2, "a little"), (4.0, "well")])
def test_wall_anchors_a_past_stop_once_the_route_has_left_it(tmp_path, beyond, amount):
    # the route stays within PASS_M of the wall up to z ~2.94, then runs on `beyond` metres
    route = [(2.5, 0.2), (2.9, 0.8), (2.9, 2.94 + beyond)]
    d = write_sample(tmp_path / "s00006", _short_wall_state(), [route])
    _, frame = load_frame(d)
    cs = candidates(frame, np.array(route))
    assert [c["wall"] for c in cs] == [True]
    rg = route_geometry(np.array(route))
    assert cs[0]["along_end_arc_m"] == pytest.approx(2.86, abs=0.1)
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["wall"]
    st = stop_phrase(frame, np.array(route), ev, cs)
    assert st["type"] == "distance" and st["relation"] == "past" and st["amount"] == amount
    assert st["anchor"]["blob"] == cs[0]["blob"]
    assert st["distance_m"] == pytest.approx(rg["arc"][-1] - cs[0]["along_end_arc_m"], abs=0.051)
    out = describe_route(d, 0)
    assert out["reason"] is None and out["stop"]["object"] == out["events"][0]["objects"][0]
    assert out["objects"][out["stop"]["object"]]["kind_hint"] == "wall"


def test_goal_still_alongside_the_wall_gets_no_wall_past_stop(tmp_path):
    route = [(2.5, 0.2), (2.9, 0.8), (2.9, 3.6)]         # the wall runs on to z 4.5: still beside the goal
    _, frame = load_frame(write_sample(tmp_path / "s", _wall_left_state(), [route]))
    cs = candidates(frame, np.array(route))
    assert [c["wall"] for c in cs] == [True]
    assert route_geometry(np.array(route))["arc"][-1] - cs[0]["along_end_arc_m"] <= 1.5
    st = stop_phrase(frame, np.array(route), build_events(frame, cs), cs)
    assert st is None or st["anchor"] is None or st["anchor"]["blob"] != cs[0]["blob"], st


@pytest.mark.parametrize("goal_z,amount", [(3.3, "just"), (2.5, "a little")])
def test_line_of_approach_stop_uses_coarse_amounts(tmp_path, goal_z, amount):
    s = _free()
    s[90:94, 40:60] = OCC                                # cabinet x 2.0..3.0, z 4.5..4.7 (not wall-like)
    route = [(2.5, 0.2), (2.5, goal_z)]                  # 1.2 m / 2.0 m short of it, nothing passed
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    st = stop_phrase(frame, np.array(route), [])
    assert st["type"] == "distance" and st["relation"] == "before" and st["amount"] == amount


# --- fix round 1: a wall's event ends where the route leaves it -----------------------------
@pytest.mark.parametrize("goal_z,after_last", [(2.0, False), (3.5, True)])
def test_after_last_measures_a_wall_from_where_the_route_leaves_it(tmp_path, goal_z, after_last):
    # wall z 0.5..2.5: goal 0.5 m before its end (still beside it) / 1 m after its end
    route = [(2.5, 0.2), (2.9, 0.8), (2.9, goal_z)]
    _, frame = load_frame(write_sample(tmp_path / "s", _short_wall_state(), [route]))
    cs = candidates(frame, np.array(route))
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["wall"] and ev[0]["end_arc_m"] == cs[0]["along_end_arc_m"]
    st = stop_phrase(frame, np.array(route), ev, cs)
    assert (st is not None and st["type"] == "after_last") is after_last, st and (st["type"], st["relation"])


def test_object_across_from_the_far_end_of_a_long_wall_merges_into_between(tmp_path):
    s = np.full((200, 100), FREE, np.uint8)
    s[10:110, 68:70] = OCC                               # wall x 3.4..3.5, z 0.5..5.5 (route's left)
    s[100:110, 34:46] = OCC                              # box x 1.7..2.3, z 5.0..5.5 (route's right)
    route = [(2.5, 0.2), (2.9, 0.8), (2.9, 7.5)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    assert [c["wall"] for c in cs] == [True, False]
    assert cs[1]["arc_m"] - cs[0]["arc_m"] > MERGE_ARC_M  # far from the wall's closest approach ...
    assert cs[1]["arc_m"] <= cs[0]["along_end_arc_m"]      # ... but still beside the wall
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["between"] and sorted(ev[0]["members"]) == [0, 1]


# --- final review I1: a through event ends at its crossing, never at a joined wall's far end ---
def test_through_with_a_joined_wall_beyond_the_door_is_measured_from_the_crossing(tmp_path):
    s = np.full((160, 100), FREE, np.uint8)
    s[:100] = _doorway_state((44, 56))
    s[42:88, 62:64] = OCC                                # corridor wall x 3.1..3.2, z 2.1..4.4, joined to the door wall
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 5.6)]        # crosses the door ~1.3 m in, goal 4.8 m in
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route], pose=(2.5, 0.8)))
    cs = candidates(frame, np.array(route))
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["through"]
    assert cs[ev[0]["members"][0]]["along_end_arc_m"] > 3.0          # the joined wall runs on past the door
    assert ev[0]["end_arc_m"] == pytest.approx(ev[0]["arc_m"]) and ev[0]["arc_m"] < 1.6
    st = stop_phrase(frame, np.array(route), ev, cs)
    assert st is not None and st["type"] == "distance" and st["relation"] == "past", st
    assert st["anchor"]["kind"] == "opening"
    assert st["distance_m"] == pytest.approx(4.8 - ev[0]["arc_m"], abs=0.11)


# --- final review M-stamp: consumers refuse events files from before Addendum A --------------
from rover_vlm.route_events import EVENTS_SCHEMA, check_events_schema  # noqa: E402


def test_events_schema_check_accepts_stamped_meta(tmp_path):
    (tmp_path / "events_meta.json").write_text(json.dumps({"schema": EVENTS_SCHEMA, "train": {}}))
    assert EVENTS_SCHEMA == "d2-endings"
    check_events_schema(tmp_path / "events_train.jsonl")          # no raise


@pytest.mark.parametrize("meta", [None, {"train": {"records": 1}}, {"schema": "d1"}])
def test_events_schema_check_refuses_old_or_missing_meta(tmp_path, meta):
    if meta is not None:
        (tmp_path / "events_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="rerun prepare_habitat_instr.py events"):
        check_events_schema(tmp_path / "events_eval.jsonl")


# --- Addendum A2 (Gate 1 r2): a blob the route runs through is not a candidate ---------------
from rover_vlm.route_events import TOUCH_M  # noqa: E402


def test_touch_constant():
    assert TOUCH_M == 0.2


def test_long_wall_the_route_clips_the_end_of_is_not_a_candidate(tmp_path):
    s = _free()
    s[40:42, 48:78] = OCC            # wall x 2.4..3.9, z 2.0..2.1: two cells right of the route, the rest left
    route = [(2.5, 0.2), (2.5, 2.05), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    assert [c["touch"] for c in cs] == [True]
    assert build_events(frame, cs) == []


def test_small_object_the_route_runs_over_is_not_a_candidate(tmp_path):
    s = _free()
    s[40:46, 48:54] = OCC            # box x 2.4..2.7, z 2.0..2.3, on the route
    route = [(2.5, 0.2), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    assert [c["touch"] for c in cs] == [True]
    assert build_events(frame, cs) == []


def test_obstacle_at_0_3_m_is_still_a_candidate(tmp_path):
    s = _free()
    s[40:46, 56:62] = OCC            # box x 2.8..3.1: nearest cell centre ~0.32 m from the route
    route = [(2.5, 0.2), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    assert len(cs) == 1 and 0.2 <= cs[0]["pass_m"] <= 0.4 and not cs[0]["touch"]
    assert [e["type"] for e in build_events(frame, cs)] == ["pass"]


def test_doorway_with_jambs_closer_than_touch_m_is_still_through(tmp_path):
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    d = write_sample(tmp_path / "s", _doorway_state((47, 60)), [route], pose=(2.5, 0.8))   # jamb 0.175 m off the route
    _, frame = load_frame(d)
    cs = candidates(frame, np.array(route))
    assert min(c["pass_m"] for c in cs) < TOUCH_M
    assert [e["type"] for e in build_events(frame, cs)] == ["through"]


# --- zero-event routes must be straight (Addendum A2) -------------------------------------------
from rover_vlm.route_events import REASONS, STRAIGHT_MAX_M  # noqa: E402

BENT_ROUTE = [(2.5, 0.2), (3.3, 1.0), (2.5, 2.5)]    # bows 0.8 m off its entry->goal line


def test_straight_max_constant():
    assert STRAIGHT_MAX_M == 0.3


def test_zero_event_bent_route_is_dropped(tmp_path):
    s = _free()
    s[90:94, 30:70] = OCC                                # sideboard ahead: a stop exists, no event
    d = write_sample(tmp_path / "s", s, [BENT_ROUTE])
    out = describe_route(d, 0)
    assert out["reason"] == "no_events_bent"
    assert "no_events_bent" in REASONS


def test_zero_event_straight_route_is_still_described(tmp_path):
    s = _free()
    s[90:94, 30:70] = OCC
    out = describe_route(write_sample(tmp_path / "s", s, [[(2.5, 0.2), (2.5, 2.5)]]), 0)
    assert out["reason"] is None and out["events"] == []
    assert out["stop"]["type"] == "distance"


def test_bent_route_with_an_event_is_described(tmp_path):
    out = describe_route(write_sample(tmp_path / "s", _stop_state(), [STOP_ROUTE]), 0)
    assert out["reason"] is None and len(out["events"]) >= 1
    assert route_geometry(STOP_ROUTE)["offset"].max() > STRAIGHT_MAX_M


def _halves_state(left_end_col, right_start_col):
    s = _free()
    s[40:42, 2:left_end_col] = OCC           # two unjoined wall halves, opening between the columns
    s[40:42, right_start_col:98] = OCC
    return s


def test_two_wall_halves_with_a_jamb_closer_than_touch_m_are_still_through(tmp_path):
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    s = _halves_state(47, 60)                # jamb cell centre x 2.325 (0.175 m off the route), other 3.025
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route], pose=(2.5, 0.8)))
    cs = candidates(frame, np.array(route))
    assert any(c["touch"] for c in cs)
    assert [e["type"] for e in build_events(frame, cs)] == ["through"]


def test_a_touching_blob_is_never_in_a_non_through_event(tmp_path):
    s = _free()
    s[40:42, 48:78] = OCC                    # wall the route clips (touching)
    s[10:16, 56:62] = OCC                    # a box ahead-left, clear of the route (z 0.5..0.8)
    route = [(2.5, 0.2), (2.5, 2.05), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    for e in build_events(frame, cs):
        assert e["type"] == "through" or not any(cs[m]["touch"] for m in e["members"])


def test_touching_blob_between_two_clear_neighbours_does_not_stop_them_merging(tmp_path):
    s = _free()
    s[10:16, 56:62] = OCC                    # box A on the left, z 0.5..0.8
    s[36:42, 48:54] = OCC                    # box on the route (touching), z 1.8..2.1
    s[60:66, 56:62] = OCC                    # box C on the left, z 3.0..3.3
    route = [(2.5, 0.2), (2.5, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    cs = candidates(frame, np.array(route))
    assert sum(c["touch"] for c in cs) == 1 and len(cs) == 3
    ev = build_events(frame, cs)
    assert [e["type"] for e in ev] == ["pass"] and len(ev[0]["members"]) == 2


# --- finalize-descriptions --desc-suffix (A5): _mix reads desc_{split}_mix_d1.jsonl ----------
@pytest.mark.parametrize("argv,stem", [([], "desc_{}_d1.jsonl"), (["--desc-suffix", "_mix"], "desc_{}_mix_d1.jsonl")])
def test_finalize_descriptions_desc_suffix(monkeypatch, tmp_path, argv, stem):
    seen = {}
    monkeypatch.setattr(prepare_instr, "stage_finalize_descriptions", lambda a: seen.update(vars(a)))
    monkeypatch.setattr("sys.argv", ["prep", "finalize-descriptions", *argv])
    prepare_instr.main()
    assert prepare_instr.desc_path(seen["labels"], "eval", seen["desc_suffix"]).name == stem.format("eval")


def test_describe_xz_matches_describe_route(tmp_path):
    sd = write_sample(tmp_path / "s", _stop_state(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    assert describe_xz(meta, frame, np.array(STOP_ROUTE), turn=detect_turn(STOP_ROUTE)) == describe_route(sd, 0)
