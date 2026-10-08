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
    m = st.traversable(_grid(s), CAM, clear_m=0.45)
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
    pair, why = st.side_pair(frame, meta, START, blob, st.map_planner())
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
    pair, why = st.side_pair(frame, meta, START, int(frame.lab[54, 50]), st.map_planner())
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
    pair, why = st.side_pair(frame, meta, START, int(frame.lab[28, 50]), st.map_planner())
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
    r, why = st.dest_route(frame, meta, START, blob, st.map_planner())
    assert why is None and r["target"] == f"f:b{blob}"
    stop = r["desc"]["stop"]
    assert (stop["type"], stop["relation"], stop["object"]) == ("near", "in front of", f"f:b{blob}")
    assert f"f:b{blob}" not in {o for e in r["desc"]["events"] for o in e["objects"]}   # not also a pass
    assert r["path"][-1][0] < 0.5


def test_dest_pairs_tags_by_image_x_and_drops_close_endpoints(tmp_path):
    meta, frame, mask = _frame(tmp_path, _dest_state())
    ra, _ = st.dest_route(frame, meta, START, int(frame.lab[70, 70]), st.map_planner())
    rb, _ = st.dest_route(frame, meta, START, int(frame.lab[70, 30]), st.map_planner())
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
    r, why = st.dest_route(frame, meta, START, int(frame.lab[48, 50]), st.map_planner())
    assert r is None and why == "too_short"
    meta, frame, mask = _frame(tmp_path, _dest_state(), name="g")
    monkeypatch.setattr(st, "describe_xz", lambda m, f, route: {
        "reason": None, "events": [], "stop": {"type": "near", "relation": "in front of", "object": "g:b999"}})
    r, why = st.dest_route(frame, meta, START, int(frame.lab[70, 70]), st.map_planner())
    assert r is None and why == "no_stop"


def test_dest_route_rejects_target_that_is_also_a_pass_event(tmp_path, monkeypatch):
    meta, frame, mask = _frame(tmp_path, _dest_state(), name="h")
    blob = int(frame.lab[70, 70])
    oid = f"h:b{blob}"
    monkeypatch.setattr(st, "describe_xz", lambda m, f, route: {
        "reason": None, "events": [{"type": "pass", "side": "left", "objects": [oid]}],
        "stop": {"type": "near", "relation": "in front of", "object": oid}})
    r, why = st.dest_route(frame, meta, START, blob, st.map_planner())
    assert r is None and why == "target_passed"


def test_frame_clash_is_within_one_frame():
    names = {"f:b1": "grey sofa", "f:b2": "leather sofa", "g:b1": "sofa", "f:b3": "lamp"}
    assert st.frame_clash("f:b1", names)
    assert not st.frame_clash("f:b3", names)
    assert not st.frame_clash("g:b1", names)


def test_load_names_splits_files_and_keeps_kinds():
    prim = [{"id": "f:b1", "passed": True, "name": "sofa", "need": "object"},
            {"id": "f:b2", "passed": False, "name": None, "need": "wall"}]
    fb = [{"id": "f:b2", "passed": True, "name": "white wall", "need": "wall"}]
    names, fallback, kinds = st.load_names(prim, fb)
    assert names == {"f:b1": "sofa"} and fallback == {"f:b2": "white wall"}
    assert kinds == {"f:b1": "object", "f:b2": "wall"}


def test_side_pair_named_gets_one_word_apart_templates(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    pair, _ = st.side_pair(frame, meta, START, blob, st.map_planner())
    assert st.name_pair(pair, {f"f:b{blob}": "armchair"}, None, {}) is None
    l, r = pair["routes"]
    assert l["namer"] == "primary" and "left of the armchair" in l["template"]
    assert l["template"].replace("left", "right") == r["template"]
    assert st.name_pair(pair, {}, None, {}) == "unnamed_object"


def test_anchor_blobs_and_frame_pairs_on_named_frames(tmp_path):
    # Two separate scenes: with the side box and the two destination boxes in ONE room, the
    # side routes pass within PASS_M of a destination box (an extra, one-sided event) and the
    # destination targets have the side box nearer than their own object (wrong stop anchor).
    sd = _sample(tmp_path / "f", _box(_state(), *BOX))
    _, frame = load_frame(sd)
    o = int(frame.lab[54, 50])
    names = {f"f:b{o}": "ottoman"}
    assert st.anchor_blobs(frame, names, None, {}) == [o]
    assert st.anchor_blobs(frame, names, None, {f"f:b{o}": "wall"}) == []
    assert st.anchor_blobs(frame, {}, names, {}) == [o]                     # fallback names count too
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], names, None, {}, st.map_planner())
    assert [p["pair_id"] for p in pairs] == [f"steer_side_f_b{o}"]
    assert reasons["side:kept"] == 1 and reasons["dest:too_short"] == 1  # target 0.7 m from the entry
    sd = _sample(tmp_path / "g", _dest_state())
    _, frame = load_frame(sd)
    a, b = int(frame.lab[70, 70]), int(frame.lab[70, 30])                  # a = image-left box
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], {f"g:b{a}": "armchair", f"g:b{b}": "sofa"}, None, {}, st.map_planner())
    dest = [p for p in pairs if p["slice"] == "dest"]
    assert [p["pair_id"] for p in dest] == [f"steer_dest_g_b{a}_b{b}"] and reasons["dest:kept"] == 1
    json.dumps(pairs)                                                        # JSON-ready


def test_frame_pairs_counts_no_start_and_no_names(tmp_path):
    sd = _sample(tmp_path / "f", _dest_state())
    _, reasons = st.frame_pairs(sd, [0.5, 0.4], {"f:b1": "x"}, None, {}, st.map_planner())
    assert reasons == {"no_start": 1}
    _, reasons = st.frame_pairs(sd, [0.5, 1.0], {}, None, {}, st.map_planner())
    assert reasons == {"no_named_object": 1}
    _, reasons = st.frame_pairs(tmp_path / "missing", [0.5, 1.0], {}, None, {}, st.map_planner())
    assert reasons == {"no_state": 1}


def test_steer_frames_one_item_per_frame_gt_record_first():
    def rec(rid, sid, kind, src="original", parent=None, start=(0.5, 1.0)):
        return {"id": rid, "image": [f"/d/{sid}/fpv_enhanced.png"],
                "habitat_meta": {"sample_id": sid, "kind": kind, "source": src, "parent_sample_id": parent,
                                 "start_uv_norm": list(start), "scene_id": "sc"}}
    items = st.steer_frames([rec("a_R", "a", "opposite", start=(0.4, 1.0)), rec("a_L", "a", "instructed"),
                             rec("b_y0_L", "b_y0", "instructed", "augmented", "b")])
    assert [i["sample_id"] for i in items] == ["a", "b_y0"]
    assert items[0]["start_uv"] == [0.5, 1.0] and items[0]["viewpoint"] == "a" and not items[0]["is_aug"]
    assert items[1]["viewpoint"] == "b" and items[1]["is_aug"] and items[1]["sample_dir"] == "/d/b_y0"


def test_floor_height(tmp_path):
    assert st.floor_height(_sample(tmp_path / "f", _state())) == pytest.approx(0.1)
    assert st.floor_height(tmp_path / "missing") is None


def _prep():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return prep


def test_steer_frame_turns_an_exception_into_a_reason(monkeypatch):
    prep = _prep()

    def boom(*a, **k):
        raise ValueError("bad map")
    monkeypatch.setattr(prep, "frame_pairs", boom)
    prep._steer_init({}, None, {}, None)
    pairs, reasons, err, reqs = prep.steer_frame({"sample_id": "x", "sample_dir": "/nope", "start_uv": [0.5, 1.0]})
    assert pairs == [] and reasons == {"error": 1} and "ValueError: bad map" in err and reqs == {}


def test_steer_frame_records_then_looks_up(tmp_path):
    prep = _prep()
    sd = _sample(tmp_path / "g", _dest_state())
    _, frame = load_frame(sd)
    a, b = int(frame.lab[70, 70]), int(frame.lab[70, 30])
    names = {f"g:b{a}": "armchair", f"g:b{b}": "sofa"}
    item = {"sample_id": "g", "sample_dir": str(sd), "start_uv": [0.5, 1.0], "viewpoint": "g", "is_aug": False,
            "scene_id": "sc", "image": "/img.png"}
    prep._steer_init(names, None, {}, None)
    pairs, _, err, reqs = prep.steer_frame(item)
    assert err is None and pairs == [] and reqs
    direct = st.map_planner()
    routes = {k: (lambda r: None if r is None else r.tolist())(direct(frame, w)) for k, w in reqs.items()}
    prep._steer_init(names, None, {}, routes)
    pairs, _, err, reqs = prep.steer_frame(item)
    assert err is None and reqs == {} and "dest" in [p["slice"] for p in pairs]
    assert pairs[0]["floor_y"] == pytest.approx(0.1) and pairs[0]["viewpoint"] == "g"


def test_answer_path_marks_floor_behind_an_obstacle_hidden(tmp_path):
    _, frame, _ = _frame(tmp_path, _box(_state(), *BOX))
    path = st.answer_path(frame.cam, [(3.4, 1.2), (2.5, 3.6)], frame)    # from beside the box to straight behind it
    assert path[0][2] == 1                                   # near the entry: seen
    assert path[-1][2] == 0                                  # 0.7 m behind the box, straight behind it
    assert st.answer_path(frame.cam, [(2.5, 0.3), (2.5, 2.0)], frame) is None   # starts 0.1 m from the camera


def test_planners_record_look_up_and_agree(tmp_path):
    meta, frame, _ = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    reqs = {}
    pair, why = st.side_pair(frame, meta, START, blob, st.recording_planner(reqs))
    assert pair is None and why == "plan_failed" and len(reqs) == 2 * len(st.STOP_BEYOND_M)
    assert all(k.startswith("f|") for k in reqs)
    direct = st.map_planner()
    routes = {k: (lambda r: None if r is None else r.tolist())(direct(frame, w)) for k, w in reqs.items()}
    looked, _ = st.side_pair(frame, meta, START, blob, st.lookup_planner(routes))
    ref, _ = st.side_pair(frame, meta, START, blob, st.map_planner())
    assert looked["routes"][0]["route_xz"] == ref["routes"][0]["route_xz"]
    assert st.request_key("f", [(1.0, 2.0)]) == "f|1.000,2.000"


def test_frame_pairs_with_recorded_then_looked_up_routes(tmp_path):
    sd = _sample(tmp_path / "g", _dest_state())
    _, frame = load_frame(sd)
    a, b = int(frame.lab[70, 70]), int(frame.lab[70, 30])
    names = {f"g:b{a}": "armchair", f"g:b{b}": "sofa"}
    reqs = {}
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], names, None, {}, st.recording_planner(reqs))
    assert pairs == [] and reqs
    direct = st.map_planner()
    routes = {k: (lambda r: None if r is None else r.tolist())(direct(frame, w)) for k, w in reqs.items()}
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], names, None, {}, st.lookup_planner(routes))
    assert [p["pair_id"] for p in pairs if p["slice"] == "dest"] == [f"steer_dest_g_b{a}_b{b}"]


def test_name_pair_clash_with_an_object_named_only_in_the_fallback(tmp_path):
    meta, frame, _ = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    pair, _ = st.side_pair(frame, meta, START, blob, st.map_planner())
    oid = f"f:b{blob}"
    assert st.name_pair(pair, {oid: "sofa"}, {}, {}) != "name_clash"
    pair, _ = st.side_pair(frame, meta, START, blob, st.map_planner())
    assert st.name_pair(pair, {oid: "sofa"}, {"f:b999": "leather sofa"}, {}) == "name_clash"


def test_lookup_planner_distinguishes_no_route_from_never_requested(tmp_path):
    _, frame, _ = _frame(tmp_path, _box(_state(), *BOX))
    key = st.request_key("f", [(1.0, 2.0)])
    plan = st.lookup_planner({key: None, st.request_key("f", [(3.0, 4.0)]): [[0, 0], [1, 1]]})
    assert plan(frame, [(1.0, 2.0)]) is None
    assert plan(frame, [(3.0, 4.0)]).shape == (2, 2)
    with pytest.raises(KeyError):
        plan(frame, [(9.0, 9.0)])


def test_object_rows_cover_visible_non_wall_objects(tmp_path):
    s = _dest_state()
    s[90:94, 0:100] = OCC                       # a wall across the far end (longer than WALL_EXTENT_M)
    sd = _sample(tmp_path / "g", s)
    _, frame = load_frame(sd)
    a, b, w = int(frame.lab[70, 70]), int(frame.lab[70, 30]), int(frame.lab[92, 50])
    rows = st.object_rows(sd, "/img.png")
    ids = {r["object_id"] for r in rows}
    assert ids == {f"g:b{a}", f"g:b{b}"}                         # the wall is left out
    assert all(r["kind_hint"] == "object" and r["image"] == "/img.png" and len(r["polygon"]) >= 3 for r in rows)
    assert {r["object_id"] for r in st.object_rows(sd, "/img.png", skip_ids={f"g:b{a}"})} == {f"g:b{b}"}
    assert st.object_rows(tmp_path / "missing", "/img.png") == []


def _label_desc():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "label_descriptions", Path(__file__).resolve().parent.parent / "scripts" / "label_descriptions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_label_descriptions_all_objects_needs_no_events(tmp_path, monkeypatch, capsys):
    ld = _label_desc()
    objs = tmp_path / "objects.jsonl"
    objs.write_text(json.dumps({"object_id": "g:b1", "kind_hint": "object", "image": "x", "polygon": [],
                                "crop_box": None, "bbox": None}) + "\n")
    out = tmp_path / "names.jsonl"
    out.write_text(json.dumps({"id": "g:b1", "passed": True, "name": "sofa"}) + "\n")   # already done
    monkeypatch.setattr("sys.argv", ["x", "--stage", "name", "--all-objects", "--objects", str(objs),
                                     "--out", str(out), "--resume"])
    ld.main()                                                    # nothing to label: no endpoint needed
    assert "1 items, 0 to label" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["x", "--stage", "compose", "--all-objects", "--out", str(out)])
    with pytest.raises(SystemExit):
        ld.main()
