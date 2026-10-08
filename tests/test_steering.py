"""Steering eval stop pairs: cuts, answer paths, pair building, naming, selection (synthetic maps).

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (as in test_route_events.py). The image's bottom
edge is 0.8 m ahead of the camera (z = 1.0).
"""
import base64
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm import steering as st
from rover_vlm.landmarks import FREE, OCC, Camera
from rover_vlm.route_events import load_frame

CAM = Camera([2.5, 0.2], [0.0, 1.0])


def _state():
    return np.full((100, 100), FREE, np.uint8)


def _box(s, rows, cols):
    s[rows[0]:rows[1], cols[0]:cols[1]] = OCC
    return s


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


def _sample(d, state, routes=()):
    """A synthetic sample dir: state.npz (with floor_height) and meta.json (one candidate per route)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0, floor_height=0.1)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    (d / "meta.json").write_text(json.dumps({"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0],
                                             "flavor": "detour", "chord": {"crossing_xz": None},
                                             "candidates": cands}))
    return d


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


def test_steer_frames_one_item_per_frame_gt_record_first():
    def rec(rid, sid, kind, src="original", parent=None, start=(0.5, 1.0)):
        return {"id": rid, "image": [f"/d/{sid}/fpv_enhanced.png"],
                "habitat_meta": {"sample_id": sid, "kind": kind, "source": src, "parent_sample_id": parent,
                                 "start_uv_norm": list(start), "scene_id": "sc",
                                 "candidate_index": 3 if kind == "instructed" else 1}}
    items = st.steer_frames([rec("a_R", "a", "opposite", start=(0.4, 1.0)), rec("a_L", "a", "instructed"),
                             rec("b_y0_L", "b_y0", "instructed", "augmented", "b")])
    assert [i["sample_id"] for i in items] == ["a", "b_y0"]
    assert "start_uv" not in items[0] and items[0]["viewpoint"] == "a" and not items[0]["is_aug"]
    assert items[0]["candidate_index"] == 3 and items[1]["candidate_index"] == 3
    assert items[1]["viewpoint"] == "b" and items[1]["is_aug"] and items[1]["sample_dir"] == "/d/b_y0"


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
    prep._steer_init({}, None, {})
    pairs, reasons, err = prep.steer_frame({"sample_id": "x", "sample_dir": "/nope", "candidate_index": 0})
    assert pairs == [] and reasons == {"error": 1} and "ValueError: bad map" in err


def test_answer_path_marks_floor_behind_an_obstacle_hidden(tmp_path):
    _, frame = load_frame(_sample(tmp_path / "f", _box(_state(), (50, 58), (46, 54))))     # 0.4 m box 1.5-1.9 m ahead
    path = st.answer_path(frame.cam, [(3.4, 1.2), (2.5, 3.6)], frame)    # from beside the box to straight behind it
    assert path[0][2] == 1                                   # near the entry: seen
    assert path[-1][2] == 0                                  # 0.7 m behind the box, straight behind it
    assert st.answer_path(frame.cam, [(2.5, 0.3), (2.5, 2.0)], frame) is None   # starts 0.1 m from the camera


def _two_boxes():
    s = _state()
    s[66:74, 66:74] = OCC        # box A: x 3.3..3.7, z 3.3..3.7
    s[66:74, 26:34] = OCC        # box B: x 1.3..1.7
    return s


def test_object_rows_cover_visible_non_wall_objects(tmp_path):
    s = _two_boxes()
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



def _stop_scene():
    s = _state()
    s[20:30, 52:58] = OCC          # box A: x 2.6..2.9, z 1.0..1.5 (passed on the left)
    s[70:80, 36:60] = OCC          # cabinet: x 1.8..3.0, z 3.5..4.0
    return s


STOP_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (2.5, 2.9)]    # ends 0.6 m before the cabinet (route_events tests)


def test_in_view_starts_at_the_bottom_edge():
    v = st.in_view(CAM, [(2.5, 0.2), (2.5, 3.0)])
    assert v[0][1] == pytest.approx(1.0, abs=0.06) and v[-1][1] == pytest.approx(3.0)
    assert len(st.in_view(CAM, [(2.5, 0.2), (2.5, 0.6)])) == 0          # never reaches the image


def test_in_view_keeps_the_last_point_before_the_edge():
    v = st.in_view(CAM, [(2.5, 0.2), (2.5, 3.0)])
    _, fwd = CAM.uv(v)
    assert fwd[0] < st.CAM_HEIGHT_M <= fwd[1]            # clip_polyline_unit then puts the entry on the edge


def test_stop_pair_answers_enter_on_an_image_edge(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    pairs, _ = st.stop_pairs(frame, meta, STOP_ROUTE)
    assert pairs
    for p in pairs:
        for r in p["routes"]:
            x, y = r["path"][0][0], r["path"][0][1]
            assert min(x, 1 - x) < 1e-6 or abs(y - 1) < 1e-6, r["path"][0]


def test_cuts_keep_at_least_min_route_m():
    fr = [f for f, _ in st.cuts(STOP_ROUTE)]
    assert fr == [0.5, 0.65, 0.8, 1.0]                                   # 0.35 x 3.26 m < 1.5 m
    f, pts = st.cuts(STOP_ROUTE)[-1]
    assert st.route_length(pts) == pytest.approx(st.route_length(STOP_ROUTE), abs=0.06)


def test_stop_pairs_on_a_route_past_a_box_to_a_cabinet(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    pairs, reasons = st.stop_pairs(frame, meta, STOP_ROUTE)
    assert pairs, dict(reasons)
    for p in pairs:
        a, b = p["routes"]
        assert (p["slice"], a["tag"], b["tag"]) == ("stop", "A", "B") and a["fraction"] < b["fraction"]
        assert st.stop_key(a["desc"]["stop"]) != st.stop_key(b["desc"]["stop"])
        assert a["path"][0] == b["path"][0]                                # one entry
        assert p["separation"] >= st.MIN_SEP_UV and st.exclusive(a, b)
    full = [p for p in pairs if p["routes"][1]["fraction"] == 1.0][0]["routes"][1]
    assert (full["desc"]["stop"]["type"], full["desc"]["stop"]["relation"]) == ("near", "in front of")


def test_frame_pairs_names_and_reasons(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    _, frame = load_frame(sd)
    box, cab = int(frame.lab[25, 55]), int(frame.lab[75, 48])
    names = {f"f:b{box}": "armchair", f"f:b{cab}": "cabinet"}
    pairs, reasons = st.frame_pairs(sd, 0, names, None, {})
    assert pairs and all(p["pair_id"].startswith("steer_stop_f_") for p in pairs) and reasons["stop:kept"] == len(pairs)
    for p in pairs:
        assert all(r["template"] and r["namer"] == "primary" for r in p["routes"])
        assert p["routes"][0]["template"] != p["routes"][1]["template"]
    _, reasons = st.frame_pairs(sd, 0, {}, None, {})
    assert reasons.get("stop:unnamed_object", 0) > 0 and not reasons.get("stop:kept")
    assert st.frame_pairs(sd, 5, names, None, {})[1] == {"bad_candidate": 1}
    assert st.frame_pairs(tmp_path / "missing", 0, names, None, {})[1] == {"no_state": 1}
    json.dumps(pairs)


def test_name_pair_clash_with_an_object_named_only_in_the_fallback():
    stop = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b1"}

    def pair():
        r = lambda f: {"desc": {"events": [], "stop": stop}, "target": "f:b1", "fraction": f}  # noqa: E731
        return {"slice": "stop", "anchor": None, "routes": [r(0.5), r(1.0)]}
    assert st.name_pair(pair(), {"f:b1": "sofa"}, {}, {}) == "same_text"          # one stop twice: identical sentences
    two = pair()
    two["routes"][1]["desc"] = {"events": [], "stop": {**stop, "relation": "just past", "type": "after_last"}}
    assert st.name_pair(two, {"f:b1": "sofa"}, {}, {}) is None                    # two different stops
    assert st.name_pair(pair(), {"f:b1": "sofa"}, {"f:b999": "leather sofa"}, {}) == "name_clash"


def test_name_pair_refuses_two_namers_in_one_pair():
    stop = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None}
    r = lambda o, f: {"desc": {"events": [], "stop": {**stop, "object": o}}, "target": o, "fraction": f}  # noqa: E731
    pair = {"slice": "stop", "anchor": None, "routes": [r("f:b1", 0.5), r("f:b2", 1.0)]}
    assert st.name_pair(pair, {"f:b1": "sofa"}, {"f:b2": "table"}, {}) == "mixed_namer"
    assert st.name_pair(pair, {"f:b1": "sofa", "f:b2": "table"}, {}, {}) is None


def test_stop_pairs_count_frame_level_reasons(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene())
    meta, frame = load_frame(sd)
    pairs, reasons = st.stop_pairs(frame, meta, [(2.5, 0.2), (2.5, 1.0)])         # 0.8 m: no cut keeps 1.5 m
    assert pairs == [] and reasons["route_too_short"] == 1 and not reasons.get("lt2_cuts")
    pairs, reasons = st.stop_pairs(frame, meta, [(2.5, 0.2), (2.5, 1.8)])         # 1.6 m: only the full cut
    assert pairs == [] and reasons["lt2_cuts"] == 1 and not reasons.get("route_too_short")


def test_frame_pairs_prefixes_the_frame_reasons(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [[(2.5, 0.2), (2.5, 1.0)]])
    assert st.frame_pairs(sd, 0, {}, None, {})[1] == {"stop:route_too_short": 1}


def test_prefix_events_marks_a_pure_stop_pair(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    pairs, _ = st.stop_pairs(frame, meta, STOP_ROUTE)
    assert pairs and all(isinstance(p["prefix_events"], bool) for p in pairs)
    assert any(p["prefix_events"] for p in pairs)
    for p in pairs:
        ea, eb = (r["desc"]["events"] for r in p["routes"])
        key = lambda e: (e["type"], e["side"], tuple(e["objects"]))  # noqa: E731
        assert p["prefix_events"] == ([key(e) for e in ea] == [key(e) for e in eb[: len(ea)]])


def _cand(pid, vp, aug, sep):
    return {"pair_id": pid, "slice": "stop", "viewpoint": vp, "is_aug": aug, "separation": sep,
            "routes": [{"tag": "A"}, {"tag": "B"}]}


def test_select_pairs_one_per_viewpoint_original_first_then_separation():
    cands = [_cand("a1", "v1", True, 0.5), _cand("a2", "v1", False, 0.2),
             _cand("b1", "v2", False, 0.2), _cand("b2", "v2", False, 0.3), _cand("c1", "v3", True, 0.4)]
    assert [p["pair_id"] for p in st.select_pairs(cands)] == ["a2", "b2", "c1"]


def test_pair_rows_carry_the_d1_schema_and_steering_fields(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    _, frame = load_frame(sd)
    box, cab = int(frame.lab[25, 55]), int(frame.lab[75, 48])
    pairs, _ = st.frame_pairs(sd, 0, {f"f:b{box}": "armchair", f"f:b{cab}": "cabinet"}, None, {})
    pair = pairs[0]
    pair.update(sample_id="f", viewpoint="f", is_aug=False, scene_id="sc", image="/x.png", sample_dir=str(sd))
    a, b = st.pair_rows(pair)
    pid = pair["pair_id"]
    assert (a["id"], a["sibling_id"], b["id"], b["sibling_id"]) == (f"{pid}_A", f"{pid}_B", f"{pid}_B", f"{pid}_A")
    for r, route in zip((a, b), pair["routes"]):
        assert r["reason"] is None and r["kind"] == "steer" and r["style_key"] == pid and r["slice"] == "stop"
        assert r["anchor"] == route["target"] and r["anchor_event"] is None and r["side"] is None
        assert {"events", "stop", "template", "path", "outlines", "desc_check"} <= set(r)
        assert r["prefix_events"] is pair["prefix_events"]
    json.dumps([a, b])


def _audit():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_instruction_labels", Path(__file__).resolve().parent.parent / "scripts" / "audit_instruction_labels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arow(pid, tag, img, anchor):
    return {"id": f"{pid}_{tag}", "pair_id": pid, "slice": "stop", "image": str(img), "scene_id": "sc",
            "anchor": anchor, "template": f"Go ... stop ({tag})", "path": [[0.5, 1.0, 1], [0.3, 0.6, 1]],
            "prefix_events": True,
            "outlines": {"f:b1": [[0.4, 0.5], [0.6, 0.5], [0.6, 0.6]], "f:b2": [[0.1, 0.5], [0.2, 0.5], [0.2, 0.6]]}}


def test_audit_samples_pairs_and_draws_both_routes(tmp_path):
    from PIL import Image
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64), (90, 90, 90)).save(img)
    rows = [r for i in range(4) for r in (_arow(f"p{i}", "A", img, "f:b1"), _arow(f"p{i}", "B", img, None if i % 2 else "f:b2"))]
    au = _audit()
    picked, pop = au.sample_steer(rows, 3, seed=0)
    assert pop == 4 and len(picked) == 3
    assert all(a["pair_id"] == b["pair_id"] and a["id"] < b["id"] for a, b in picked)
    item = au.steer_item(*picked[0])
    assert item["id"] == picked[0][0]["pair_id"] and item["img"].startswith("data:image/jpeg;base64,")
    assert picked[0][0]["template"] in item["instruction"] and picked[0][1]["template"] in item["instruction"]
    assert [v["v"] for v in au.VERDICTS["steer"]] == ["ok", "stop", "event", "object", "same", "unsure"]
    assert item["pure"] is True


def test_audit_shows_both_routes_when_a_is_a_prefix_of_b(tmp_path):
    import io
    from PIL import Image
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64), (90, 90, 90)).save(img)
    a, b = _arow("p", "A", img, "f:b1"), _arow("p", "B", img, "f:b2")
    b["path"] = [[0.5, 1.0, 1], [0.3, 0.6, 1], [0.3, 0.3, 1]]       # A is a prefix of B
    au = _audit()
    item = au.steer_item(a, b)
    im = Image.open(io.BytesIO(base64.b64decode(item["img"].split(",", 1)[1]))).convert("RGB")
    px = np.asarray(im, int).reshape(-1, 3)
    near = lambda c: int((np.abs(px - np.array(c)).max(axis=1) < 60).sum())  # noqa: E731
    assert near(au.STEER_A) > 50 and near(au.STEER_B) > 50


def _base(sid):
    return {"id": f"{sid}_L", "image": [f"/d/{sid}/fpv_enhanced.png"],
            "conversations": [{"from": "human", "value": "old"}, {"from": "gpt", "value": '{"path":[[0,1,1],[0.5,0.5,0]]}'}],
            "habitat_meta": {"scene_id": "sc", "sample_id": sid, "parent_sample_id": None, "source": "original",
                             "yaw_deg": None, "flavor": "doorway", "framing": "endpoints", "entry_edge": "bottom",
                             "alt_path": [[0, 1, 1]], "candidate_index": 2, "instruction": "old text"}}


def _srow(pid, tag, sib):
    return {"id": f"{pid}_{tag}", "pair_id": pid, "sibling_id": f"{pid}_{sib}", "slice": "stop", "sample_id": "f",
            "viewpoint": "f", "path": [[0.5, 1.0, 1], [0.4, 0.7, 1], [0.5, 0.6, 0 if tag == "B" else 1]],
            "desc_check": {"pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 3], "events": []},
            "anchor": "f:b1", "anchor_event": None, "side": None, "prefix_events": True,
            "template": f"Go left of the sofa, and stop {tag} it."}


def test_finalize_steer_writes_four_records_per_route_and_drops_template_pairs_from_composed():
    rows = [_srow("p1", "A", "B"), _srow("p1", "B", "A"), _srow("p2", "A", "B"), _srow("p2", "B", "A")]
    desc = {"p1_A": {"passed": True, "description": "Keep left of the sofa, then stop just after it.", "instruction_source": "cosmos"},
            "p1_B": {"passed": True, "description": "Keep left of the sofa, then stop well past it.", "instruction_source": "cosmos"},
            "p2_A": {"passed": True, "description": "Go left of the sofa, and stop A it.", "instruction_source": "template"},
            "p2_B": {"passed": True, "description": "x", "instruction_source": "cosmos"}}
    out = st.finalize_steer({"f": _base("f")}, rows, desc)
    files = out["files"]
    assert set(files) == set(st.STEER_FILES) and len(st.STEER_FILES) == 4
    assert len(files["eval_steer_stop_template_entry"]) == 4 and len(files["eval_steer_stop_template_goal"]) == 4
    assert [r["id"] for r in files["eval_steer_stop_composed_entry"]] == ["p1_A", "p1_B"]
    assert out["stats"]["stop:composed_dropped"] == 1 and out["stats"]["stop:pairs"] == 2
    r = files["eval_steer_stop_composed_entry"][0]
    hm = r["habitat_meta"]
    assert hm["steer"] == {"slice": "stop", "anchor": "f:b1", "anchor_event": None, "side": None, "viewpoint": "f",
                           "prefix_events": True}
    assert hm["sibling_id"] == "p1_B" and "alt_path" not in hm and "candidate_index" not in hm
    assert hm["prompt_form"] == "entry" and "stop just after it" in r["conversations"][0]["value"]
    assert json.loads(r["conversations"][1]["value"])["path"][0] == [0.5, 1.0, 1]
    g = files["eval_steer_stop_template_goal"][1]
    assert g["habitat_meta"]["goal_in_prompt"] and g["habitat_meta"]["goal_visible"] == 0
    assert "[0.5, 0.6]" in g["conversations"][0]["value"]


def test_finalize_steer_counts_unlabelled_pairs():
    rows = [_srow("p1", "A", "B"), _srow("p1", "B", "A")]
    out = st.finalize_steer({"f": _base("f")}, rows, {})
    assert out["stats"]["stop:unlabelled"] == 1 and out["files"]["eval_steer_stop_composed_entry"] == []
    assert len(out["files"]["eval_steer_stop_template_entry"]) == 2


def test_finalize_steer_stage_refuses_missing_compose_rows(tmp_path):
    import types
    prep = _prep()
    sd = tmp_path / "steer"
    sd.mkdir()
    (sd / "events_steer_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (_srow("p1", "A", "B"), _srow("p1", "B", "A"))))
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps([_base("f")]))
    args = types.SimpleNamespace(frames=frames, steer_dir=sd, desc=tmp_path / "none.jsonl", out_dir=tmp_path,
                                 allow_missing=False)
    with pytest.raises(SystemExit, match="no compose row"):
        prep.stage_finalize_steer(args)
    args.allow_missing = True
    prep.stage_finalize_steer(args)
    assert len(json.loads((tmp_path / "eval_steer_stop_template_entry.json").read_text())) == 2


def _viz():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "visualize_steering", Path(__file__).resolve().parent.parent / "scripts" / "visualize_steering.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pred(rid, sib, pred, gt, side):
    return {"id": rid, "generated": json.dumps(pred), "parsed": {"path": pred}, "gt": {"path": gt},
            "habitat_meta": {"sibling_id": sib, "instruction": f"go {side}",
                             "desc_check": {"pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 3], "events": []},
                             "steer": {"slice": "stop", "anchor": "x", "anchor_event": None, "side": None, "viewpoint": "v"}}}


def test_metric_cells_reads_predictions_per_set_and_tag(tmp_path):
    vz = _viz()
    run = tmp_path / "eval_steer_stop_template_entry" / "d1"
    run.mkdir(parents=True)
    a = [[0.5, 1.0, 1], [0.3, 0.6, 1]]
    b = [[0.5, 1.0, 1], [0.7, 0.6, 1]]
    (run / "predictions.json").write_text(json.dumps([_pred("p_A", "p_B", a, a, "a"), _pred("p_B", "p_A", b, b, "b")]))
    cells = vz.metric_cells(tmp_path, ["d1", "r4"])
    assert len(cells) == 1
    c = cells[0]
    assert (c["slice"], c["text"], c["form"], c["tag"]) == ("stop", "template", "entry", "d1")
    assert c["steering"]["pair_success"] == 1.0 and c["parse_rate"] == 1.0


def test_pair_panel_size(tmp_path):
    from PIL import Image
    vz = _viz()
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    a = _pred("p_A", "p_B", [[0.5, 1.0, 1], [0.3, 0.6, 1]], [[0.5, 1.0, 1], [0.35, 0.6, 1]], "a")
    b = _pred("p_B", "p_A", [[0.5, 1.0, 1], [0.7, 0.6, 1]], [[0.5, 1.0, 1], [0.65, 0.6, 1]], "b")
    assert vz.pair_panel(img, a, b, px=200).size == (400, 200 + vz.TEXT_H)
