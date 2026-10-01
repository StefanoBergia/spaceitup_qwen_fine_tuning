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
    s[46:56, 44:54] = OCC        # box B: x 2.2..2.7, z 2.3..2.8 (on the route's left there)
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
    MAX_EVENTS, build_events, collect_objects, describe_route, stop_phrase,
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
    assert out["events"][0]["type"] == "pass" and out["events"][0]["side"] == "left"
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
