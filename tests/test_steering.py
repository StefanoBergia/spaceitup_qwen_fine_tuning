"""Steering eval pairs: planner, pair builders, naming, selection, finalize (synthetic maps).

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (as in test_route_events.py). The image's bottom
edge is 0.8 m ahead of the camera (z = 1.0).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm import steering as st
from rover_vlm.landmarks import FREE, OCC, UNKNOWN, Camera, Grid
from rover_vlm.route_events import load_frame

CAM = Camera([2.5, 0.2], [0.0, 1.0])


def _state():
    return np.full((100, 100), FREE, np.uint8)


def _box(s, rows, cols):
    s[rows[0]:rows[1], cols[0]:cols[1]] = OCC
    return s


def _grid(s):
    return Grid(s, 0.05, 0.0, 0.0)


def cell(x, z):
    return int(z / 0.05), int(x / 0.05)


def test_traversable_keeps_clearance_unknown_and_frame():
    s = _box(_state(), (50, 58), (46, 54))      # box: cell centres x 2.325..2.675, z 2.525..2.875
    s[:, 90:] = UNKNOWN                          # x >= 4.5 never seen
    m = st.traversable(_grid(s), CAM)
    assert not m[cell(2.5, 2.2)]                 # centre 0.30 m from the box
    assert m[cell(2.5, 2.0)]                     # centre 0.50 m from the box
    assert not m[cell(4.6, 4.0)]                 # unknown
    assert not m[cell(2.5, 0.6)]                 # below the image (0.43 m ahead < 0.8 m)
    assert not m[cell(0.2, 1.5)]                 # outside the 90 degree wedge
    assert m[cell(3.0, 1.5)]


def test_astar_goes_round_an_obstacle_and_fails_when_cut_off():
    s = _box(_state(), (50, 58), (46, 54))
    m = st.traversable(_grid(s), CAM)
    a, b = cell(2.5, 1.2), cell(2.5, 3.6)
    path = st.astar(m, a, b)
    assert path[0] == a and path[-1] == b and all(m[p] for p in path)
    wall = _box(_state(), (50, 52), (0, 100))    # wall across the room at z 2.5
    assert st.astar(st.traversable(_grid(wall), CAM), a, b) is None


def test_string_pull_keeps_ends_and_line_of_sight():
    m = st.traversable(_grid(_box(_state(), (50, 58), (46, 54))), CAM)
    path = st.astar(m, cell(2.5, 1.2), cell(2.5, 3.6))
    pulled = st.string_pull(m, path)
    assert pulled[0] == path[0] and pulled[-1] == path[-1] and len(pulled) < len(path)
    assert all(st.line_clear(m, p, q) for p, q in zip(pulled[:-1], pulled[1:]))


def test_plan_route_keeps_its_via_point_and_needs_a_snap():
    g = _grid(_box(_state(), (50, 58), (46, 54)))
    m = st.traversable(g, CAM)
    route = st.plan_route(g, m, [(2.5, 1.0), (3.25, 2.7), (2.5, 3.4)])
    assert route is not None and route.shape[1] == 2
    assert np.min(np.linalg.norm(route - [3.25, 2.7], axis=1)) < 0.1     # via kept as a corner
    assert route[:, 0].max() > 2.9                                        # went round on +x
    assert st.plan_route(g, m, [(2.5, 1.0), (2.5, 2.7)]) is None         # inside the box: no cell within 0.3 m


def test_route_length():
    assert st.route_length([(0, 0), (3, 4), (3, 5)]) == pytest.approx(6.0)
    assert st.route_length([(1, 1)]) == 0.0


def _desc(side, goal=(2.5, 3.0)):
    """A hand-made describe_xz result: one pass of a post at x 2.5..2.55, z 2.0."""
    return {"reason": None, "pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0], "goal_xz": list(goal),
            "events": [{"type": "pass", "side": side, "objects": ["f:b1"], "arc_m": 1.0}],
            "stop": {"type": "after_last", "relation": "just after", "amount": None, "distance_m": None,
                     "object": None},
            "objects": {"f:b1": {"cells_xz": [[2.5, 2.0], [2.55, 2.0]]}}}


L_XZ = [(2.5, 1.0), (3.3, 2.0), (2.5, 3.0)]        # bows to +x: the rover's left
R_XZ = [(2.5, 1.0), (1.7, 2.0), (2.5, 3.0)]


def test_entry_floor_inverts_the_camera():
    meta = {"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0]}
    assert st.entry_floor(meta, [0.5, 1.0]) == pytest.approx([2.5, 1.0])
    assert st.entry_floor(meta, [0.5, 0.4]) is None             # at or above the horizon


def test_answer_path_projects_a_fully_visible_floor_route():
    path = st.answer_path(CAM, L_XZ)
    assert path[0][:2] == pytest.approx([0.5, 1.0], abs=1e-3)
    assert all(w[2] == 1 for w in path) and 3 <= len(path) <= 12
    assert path[1][0] < 0.5                                      # +x is image-left


def test_separation():
    a, b = st.answer_path(CAM, L_XZ), st.answer_path(CAM, R_XZ)
    assert st.separation(a, a) == 0.0
    assert st.separation(a, b) > 0.1


def test_exclusive_needs_each_route_to_fail_the_other_text():
    rl = {"path": st.answer_path(CAM, L_XZ), "desc": _desc("left")}
    rr = {"path": st.answer_path(CAM, R_XZ), "desc": _desc("right")}
    assert st.exclusive(rl, rr)
    assert not st.exclusive(rl, {**rr, "desc": _desc("left")})     # R fails its own text
    assert not st.exclusive(rl, {**rl, "desc": _desc("left")})     # one route satisfies both


def _sample(d, state):
    """A synthetic sample dir: state.npz (with floor_height) and meta.json, no candidates."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0, floor_height=0.1)
    (d / "meta.json").write_text(json.dumps({"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0],
                                             "flavor": "detour", "chord": {"crossing_xz": None},
                                             "candidates": []}))
    return d


START = np.array([2.5, 1.0])
BOX = ((50, 58), (46, 54))                      # 0.4 m box 1.5-1.9 m ahead of the entry


def _frame(tmp_path, state, name="f"):
    meta, frame = load_frame(_sample(tmp_path / name, state))
    return meta, frame, st.traversable(frame.grid, frame.cam)


def test_side_pair_rounds_both_sides_one_word_apart(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    pair, why = st.side_pair(frame, meta, mask, START, blob)
    assert why is None and pair["slice"] == "side" and pair["anchor"] == f"f:b{blob}"
    l, r = pair["routes"]
    assert (l["tag"], l["side"], r["tag"], r["side"]) == ("L", "left", "R", "right")
    assert l["route_xz"][0] == r["route_xz"][0] and l["path"][0] == r["path"][0]     # shared entry
    i = l["anchor_event"]
    assert i == r["anchor_event"]
    assert l["desc"]["events"][i]["side"] == "left" and r["desc"]["events"][i]["side"] == "right"
    assert st.stop_key(l["desc"]["stop"]) == st.stop_key(r["desc"]["stop"])
    assert pair["separation"] >= st.MIN_SEP_UV


def test_side_pair_rejects_a_second_obstacle_between_the_routes(tmp_path):
    s = _box(_state(), *BOX)
    s[62:64, 49:51] = OCC                        # post 0.25 m behind the box, between both routes
    meta, frame, mask = _frame(tmp_path, s)
    pair, why = st.side_pair(frame, meta, mask, START, int(frame.lab[54, 50]))
    assert pair is None and why == "not_two_sided"


def test_two_routes_round_the_same_side_are_not_two_sided(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    a = st.plan_route(frame.grid, mask, [START, (3.25, 2.7), (2.5, 3.4)])
    b = st.plan_route(frame.grid, mask, [START, (3.7, 2.7), (2.5, 3.4)])
    assert not st.two_sided(frame, a, b, blob)
    c = st.plan_route(frame.grid, mask, [START, (1.75, 2.7), (2.5, 3.4)])
    assert st.two_sided(frame, a, c, blob)


def test_anchor_too_close_to_the_entry(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), (26, 30), (48, 52)))   # 0.3-0.5 m ahead
    pair, why = st.side_pair(frame, meta, mask, START, int(frame.lab[28, 50]))
    assert pair is None and why == "too_near"


def _ev(t, side, objs):
    return {"type": t, "side": side, "objects": objs, "arc_m": 1.0}


def test_side_difference():
    o = "f:b1"
    wall_l, wall_r = _ev("wall", "left", ["f:b9"]), _ev("wall", "left", ["f:b9"])
    assert st.side_difference([wall_l, _ev("pass", "left", [o])], [wall_r, _ev("pass", "right", [o])], o) == (1, None)
    assert st.side_difference([_ev("pass", "left", [o])], [_ev("pass", "left", [o])], o) == (None, "anchor_missing")
    assert st.side_difference([_ev("pass", "left", [o, "f:b2"])], [_ev("pass", "right", [o])], o) == (None, "anchor_missing")
    assert st.side_difference([wall_l, _ev("pass", "left", [o])], [_ev("pass", "right", [o])], o) == (None, "events_differ")
    assert st.side_difference([_ev("pass", "left", [o]), _ev("pass", "left", ["f:b2"])],
                              [_ev("pass", "right", [o]), _ev("pass", "right", ["f:b2"])], o) == (None, "events_differ")


def _dest_state():
    s = _state()
    s[66:74, 66:74] = OCC        # box A: x 3.3..3.7, z 3.3..3.7 (image-left)
    s[66:74, 26:34] = OCC        # box B: x 1.3..1.7 (image-right)
    return s


def test_dest_route_stops_in_front_of_its_object(tmp_path):
    meta, frame, mask = _frame(tmp_path, _dest_state())
    blob = int(frame.lab[70, 70])
    r, why = st.dest_route(frame, meta, mask, START, blob)
    assert why is None and r["target"] == f"f:b{blob}"
    stop = r["desc"]["stop"]
    assert (stop["type"], stop["relation"], stop["object"]) == ("near", "in front of", f"f:b{blob}")
    assert f"f:b{blob}" not in {o for e in r["desc"]["events"] for o in e["objects"]}   # not also a pass
    assert r["path"][-1][0] < 0.5


def test_dest_pairs_tags_by_image_x_and_drops_close_endpoints(tmp_path):
    meta, frame, mask = _frame(tmp_path, _dest_state())
    ra, _ = st.dest_route(frame, meta, mask, START, int(frame.lab[70, 70]))
    rb, _ = st.dest_route(frame, meta, mask, START, int(frame.lab[70, 30]))
    pairs, reasons = st.dest_pairs([rb, ra])
    assert len(pairs) == 1 and not reasons
    p = pairs[0]
    assert [r["tag"] for r in p["routes"]] == ["A", "B"]
    assert p["routes"][0]["path"][-1][0] < p["routes"][1]["path"][-1][0]
    assert p["targets"] == [ra["target"], rb["target"]] and p["slice"] == "dest"
    assert st.exclusive(*p["routes"])
    near_twin = {**ra, "path": [w[:] for w in ra["path"]]}
    near_twin["path"][-1][0] += 0.1                                  # endpoints 0.1 apart in x
    pairs, reasons = st.dest_pairs([ra, near_twin])
    assert pairs == [] and reasons["too_close"] == 1


def test_dest_route_too_short_and_wrong_stop(tmp_path, monkeypatch):
    s = _state()
    s[46:50, 48:52] = OCC                        # small box 1.3-1.5 m ahead: target 0.5 m from the entry
    meta, frame, mask = _frame(tmp_path, s)
    r, why = st.dest_route(frame, meta, mask, START, int(frame.lab[48, 50]))
    assert r is None and why == "too_short"
    meta, frame, mask = _frame(tmp_path, _dest_state(), name="g")
    monkeypatch.setattr(st, "describe_xz", lambda m, f, route: {
        "reason": None, "events": [], "stop": {"type": "near", "relation": "in front of", "object": "g:b999"}})
    r, why = st.dest_route(frame, meta, mask, START, int(frame.lab[70, 70]))
    assert r is None and why == "no_stop"
