"""Manifest-driven prep: record building from fabricated sample dirs, and the three
split-integrity assertions that make a round trustworthy.

No live dataset is touched -- every sample dir here is written into tmp_path, which is
also what gives build_record its first offline coverage.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from rover_vlm.habitat_data import build_record

_SPEC = importlib.util.spec_from_file_location(
    "prepare_habitat", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat.py"
)
prepare_habitat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_habitat)


def make_sample(root, sample_id, scene_id, *, goal_uv=(256.0, 300.0), goal_hidden=False,
                parent=None, yaw=None, in_fov=True, path_uv=None):
    """Write a minimal but structurally faithful Habitat sample dir; return its path."""
    d = Path(root) / scene_id / "samples" / sample_id
    d.mkdir(parents=True)
    uv = path_uv if path_uv is not None else [[256.0, 500.0], [256.0, 400.0], list(goal_uv)]
    fpv = {
        "image_size": [512, 512],
        "goal": {"uv": list(goal_uv), "hidden": goal_hidden},
        "candidates": [
            {"runs": [{"hidden": False, "uv": [[100.0, 500.0], [100.0, 400.0]]}]},
            {"runs": [{"hidden": False, "uv": uv}]},
        ],
    }
    meta = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "flavor": "doorway",
        "label": 1,
        "accepted": [1],
        "correct_path_in_fov": in_fov,
        "candidates": [{"display_index": 0}, {"display_index": 1}],
    }
    if parent is not None:
        meta["aug"] = {"parent_sample_id": parent, "yaw_deg": yaw, "goal_x_target": 0.5}
    (d / "fpv_paths.json").write_text(json.dumps(fpv))
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "fpv_enhanced.png").write_bytes(b"")
    return d


def row(d, sample_id, source="original", parent=None):
    return {"sample_id": sample_id, "dir": str(d), "source": source,
            "parent_sample_id": parent, "label_side": "left", "margin": 0.3,
            "decidability": 0.5, "near_symmetric": False, "n_candidates": 2}


# --- build_record, offline ----------------------------------------------------------

def test_legacy_mode_keeps_separate_goal_key(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a")
    rec = build_record(d)
    obj = json.loads(rec["conversations"][1]["value"])
    assert "goal" in obj and "path" in obj
    assert "straight ahead" in rec["conversations"][0]["value"]


def test_goal_in_prompt_names_the_goal_and_drops_the_goal_key(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a", goal_uv=(128.0, 320.0))
    rec = build_record(d, goal_in_prompt=True)
    prompt = rec["conversations"][0]["value"]
    obj = json.loads(rec["conversations"][1]["value"])
    assert "[0.25, 0.625]" in prompt
    assert "straight ahead" not in prompt
    assert "goal" not in obj
    # the answer's final waypoint IS the goal named in the prompt
    assert obj["path"][-1] == [0.25, 0.625, 1]


def test_goal_visibility_is_stated_in_the_prompt(tmp_path):
    vis = build_record(make_sample(tmp_path / "a", "s_c000", "sc", goal_hidden=False),
                       goal_in_prompt=True)
    hid = build_record(make_sample(tmp_path / "b", "s_c001", "sc", goal_hidden=True),
                       goal_in_prompt=True)
    assert "it is visible" in vis["conversations"][0]["value"]
    assert "hidden behind an obstacle. Predict" in hid["conversations"][0]["value"]
    assert json.loads(vis["conversations"][1]["value"])["path"][-1][2] == 1
    assert json.loads(hid["conversations"][1]["value"])["path"][-1][2] == 0


def test_offscreen_goal_is_rejected_only_in_goal_mode(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a", goal_uv=(600.0, 300.0),
                    path_uv=[[256.0, 500.0], [400.0, 400.0], [600.0, 300.0]])
    reasons = {}
    assert build_record(d, goal_in_prompt=True, reasons=reasons) is None
    assert reasons == {"goal_offscreen": 1}
    assert build_record(d) is not None  # legacy mode clamps instead


def test_skip_reasons_are_counted(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a", in_fov=False)
    reasons = {}
    assert build_record(d, reasons=reasons) is None
    assert reasons == {"not_in_fov": 1}


def test_sidecar_carries_source_and_parent(tmp_path):
    orig = build_record(make_sample(tmp_path / "o", "p_c000", "sc"))
    aug = build_record(make_sample(tmp_path / "a", "p_c000_y0", "sc",
                                   parent="p_c000", yaw=12.5))
    assert orig["habitat_meta"]["source"] == "original"
    assert orig["habitat_meta"]["parent_sample_id"] is None
    assert aug["habitat_meta"]["source"] == "augmented"
    assert aug["habitat_meta"]["parent_sample_id"] == "p_c000"
    assert aug["habitat_meta"]["yaw_deg"] == 12.5


# --- split integrity ----------------------------------------------------------------

def _records(tmp_path, specs):
    out = []
    for i, (sid, scene, parent) in enumerate(specs):
        d = make_sample(tmp_path / f"r{i}", sid, scene, parent=parent)
        out.append(build_record(d, goal_in_prompt=True,
                                meta_extra={"source": "augmented" if parent else "original"}))
    return out


def test_clean_split_passes(tmp_path):
    train = _records(tmp_path / "t", [("a_c000", "scene_a", None), ("a_c000_y0", "scene_a", "a_c000")])
    ev = _records(tmp_path / "e", [("b_c000", "scene_b", None)])
    prepare_habitat.check_split_integrity(train, ev)


def test_shared_sample_id_is_fatal(tmp_path):
    train = _records(tmp_path / "t", [("a_c000", "scene_a", None)])
    ev = _records(tmp_path / "e", [("a_c000", "scene_b", None)])
    with pytest.raises(AssertionError, match="sample ids in both splits"):
        prepare_habitat.check_split_integrity(train, ev)


def test_shared_scene_is_fatal(tmp_path):
    """The v2 failure: distinct ids, same scene in both splits."""
    train = _records(tmp_path / "t", [("a_c000", "scene_a", None)])
    ev = _records(tmp_path / "e", [("a_c001", "scene_a", None)])
    with pytest.raises(AssertionError, match="scenes in both splits"):
        prepare_habitat.check_split_integrity(train, ev)


def test_parent_split_from_child_is_fatal(tmp_path):
    """Same scene would also trip; use distinct scenes so only the parent rule can fire."""
    train = _records(tmp_path / "t", [("a_c000", "scene_a", None)])
    ev = _records(tmp_path / "e", [("z_c000_y0", "scene_b", "a_c000")])
    with pytest.raises(AssertionError, match="parent in the other split"):
        prepare_habitat.check_split_integrity(train, ev)


# --- manifest reading ---------------------------------------------------------------

def test_records_from_manifest_carries_manifest_fields(tmp_path):
    d = make_sample(tmp_path, "a_c000", "scene_a")
    reasons = {}
    recs = prepare_habitat.records_from_manifest(
        [row(d, "a_c000")], framing="goal", reasons=reasons)
    assert len(recs) == 1
    hm = recs[0]["habitat_meta"]
    assert hm["source"] == "original" and hm["decidability"] == 0.5
    assert hm["scene_id"] == "scene_a" and reasons == {}


def test_split_stats_reports_terminus_spread(tmp_path):
    recs = []
    for i, gx in enumerate((128.0, 256.0, 384.0)):
        d = make_sample(tmp_path / f"s{i}", f"a_c00{i}", "scene_a", goal_uv=(gx, 300.0),
                        path_uv=[[256.0, 500.0], [gx, 300.0]])
        recs.append(build_record(d, goal_in_prompt=True, meta_extra={"source": "augmented"}))
    stats = prepare_habitat.split_stats(recs)
    assert stats["n"] == 3 and stats["scenes"] == 1
    assert stats["terminus_x_sd"] > 0.1
    assert stats["frac_terminus_at_0.5"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["by_source"] == {"augmented": 3}


def test_read_manifest_rejects_empty(tmp_path):
    f = tmp_path / "m.jsonl"
    f.write_text("\n")
    with pytest.raises(SystemExit):
        prepare_habitat.read_manifest(f)


# --- round 4: both endpoints handed over ---------------------------------------------

def test_endpoints_framing_names_both_ends(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a", goal_uv=(128.0, 320.0),
                    path_uv=[[400.0, 500.0], [200.0, 400.0], [128.0, 320.0]])
    rec = build_record(d, framing="endpoints")
    prompt = rec["conversations"][0]["value"]
    path = json.loads(rec["conversations"][1]["value"])["path"]
    assert "enters the frame at [0.781, 0.977]" in prompt
    assert "goal at [0.25, 0.625]" in prompt
    # first and last waypoints are exactly the two points named in the prompt
    assert path[0] == [0.781, 0.977, 1]
    assert path[-1] == [0.25, 0.625, 1]


def test_endpoints_framing_states_each_end_visibility(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a", goal_hidden=True)
    prompt = build_record(d, framing="endpoints")["conversations"][0]["value"]
    assert "where it is visible, and ends at the goal" in prompt
    assert "where it is hidden behind an obstacle" in prompt


def test_sidecar_records_start_and_entry_edge(tmp_path):
    """entry_edge must come from consistency.start_edge, whose corner handling
    (bottom wins at a corner) differs from the obvious left/right-first version."""
    from rover_vlm.consistency import start_edge

    bottom = build_record(make_sample(tmp_path / "b", "a_c000", "sc",
                                      path_uv=[[500.0, 511.0], [300.0, 400.0], [256.0, 300.0]]),
                          framing="endpoints")
    left = build_record(make_sample(tmp_path / "l", "a_c001", "sc",
                                    path_uv=[[0.0, 400.0], [128.0, 350.0], [256.0, 300.0]]),
                        framing="endpoints")
    assert bottom["habitat_meta"]["entry_edge"] == "bottom"
    assert left["habitat_meta"]["entry_edge"] == "left"
    assert left["habitat_meta"]["start_uv_norm"] == [0.0, 0.781]
    for rec in (bottom, left):
        path = json.loads(rec["conversations"][1]["value"])["path"]
        assert rec["habitat_meta"]["entry_edge"] == start_edge({"path": path})


def test_corner_start_is_bottom_not_side(tmp_path):
    """(0.98, 1.0) is a bottom entry; the naive left/right-first test would call it right."""
    rec = build_record(make_sample(tmp_path, "a_c000", "sc",
                                   path_uv=[[502.0, 512.0], [300.0, 400.0], [256.0, 300.0]]),
                       framing="endpoints")
    assert rec["habitat_meta"]["entry_edge"] == "bottom"


def test_goal_in_prompt_alias_still_reproduces_round_3(tmp_path):
    """The v3 prep must stay byte-reproducible after the framing refactor."""
    d = make_sample(tmp_path, "s_c000", "scene_a", goal_uv=(128.0, 320.0))
    assert build_record(d, goal_in_prompt=True) == build_record(d, framing="goal")
    assert build_record(d, goal_in_prompt=False) == build_record(d, framing="legacy")


def test_unknown_framing_is_rejected(tmp_path):
    d = make_sample(tmp_path, "s_c000", "scene_a")
    with pytest.raises(ValueError, match="unknown framing"):
        build_record(d, framing="both-ends")


def test_split_stats_reports_entry_edge_distribution(tmp_path):
    recs = []
    for i, uv in enumerate(([[256.0, 511.0], [256.0, 300.0]],
                            [[0.0, 400.0], [256.0, 300.0]],
                            [[511.0, 400.0], [256.0, 300.0]])):
        recs.append(build_record(
            make_sample(tmp_path / f"s{i}", f"a_c00{i}", "scene_a", path_uv=uv),
            framing="endpoints", meta_extra={"source": "original"}))
    assert prepare_habitat.split_stats(recs)["by_entry_edge"] == {
        "bottom": 1, "left": 1, "right": 1}
