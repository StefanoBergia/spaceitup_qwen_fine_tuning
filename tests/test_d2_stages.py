"""d2 stages (spec 2026-10-13 "Data flow"): events from the v2 labelsets, seeding the d2 label files from d1,
finalize --round d2 re-checking every text."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from rover_vlm.descriptions import finalize_descriptions, same_description_geometry, seed_d2_labels, turn_rejects

_SPEC = importlib.util.spec_from_file_location(
    "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

STOP = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}


def _row(rid, events, reason=None, stop=STOP):
    return {"id": rid, "reason": reason, "events": events, "stop": stop if reason is None else None,
            "pair_id": None, "kind": "instructed"}


PASS = {"type": "pass", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
TURN = {"type": "turn", "side": "left", "relation": "behind", "objects": ["f:b1"], "arc_m": 3.0,
        "turn_xz": [0, 3], "pre_heading_xz": [0, 1], "gt_lateral_m": 1.0}


def test_same_description_geometry():
    assert same_description_geometry(_row("a", [PASS]), _row("a", [dict(PASS, arc_m=1.2)]))   # arc does not matter
    assert not same_description_geometry(_row("a", [PASS]), _row("a", [PASS, TURN]))
    assert not same_description_geometry(_row("a", [PASS]), _row("a", [], reason="spur"))
    assert not same_description_geometry(_row("a", [PASS]), _row("a", [PASS], stop=dict(STOP, object="f:b9")))


def test_seed_reuses_unchanged_routes_and_lists_changed_ones():
    old = [_row("a", [PASS]), _row("b", [PASS]), _row("c", [], reason="no_events_bent")]
    new = [_row("a", [PASS]), _row("b", [PASS, TURN]), _row("c", [TURN]), _row("d", [], reason="spur")]
    objects = [{"object_id": "f:b1"}, {"object_id": "f:b2"}]
    names = {"gemma": [{"id": "f:b1", "name": "sofa", "passed": True}, {"id": "f:b7", "name": "x", "passed": True},
                       {"id": "f:b2", "error": "timeout"}], "cosmos": []}
    desc = [{"id": "a", "description": "Go left of the sofa.", "passed": True},
            {"id": "b", "description": "Go left of the sofa.", "passed": True}]
    s = seed_d2_labels(old, new, objects, names, desc)
    assert [r["id"] for r in s["desc"]] == ["a"] and s["desc"][0]["reused_from"] == "d1"
    assert [r["id"] for r in s["changed"]] == ["b", "c"]
    assert [r["id"] for r in s["names"]["gemma"]] == ["f:b1"]          # gone object and error row are not seeded


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _seed_args(tmp_path, force=False):
    d1, v4, labels = tmp_path / "v2", tmp_path / "v4", tmp_path / "labels"
    for s in ("train", "eval"):
        _write(d1 / f"events_{s}.jsonl", [_row(f"{s}a", [PASS])])
        _write(v4 / f"events_{s}.jsonl", [_row(f"{s}a", [PASS]), _row(f"{s}b", [PASS, TURN])])
        _write(v4 / f"objects_{s}.jsonl", [{"object_id": "f:b1"}, {"object_id": "f:b2"}])
        _write(labels / f"names_{s}_gemma_d1.jsonl", [{"id": "f:b1", "name": "sofa", "passed": True}])
        _write(labels / f"names_{s}_d1.jsonl", [])
        _write(labels / f"desc_{s}_mix_d1.jsonl", [{"id": f"{s}a", "description": "Go left of the sofa.", "passed": True}])
    return Namespace(d1_events_dir=d1, events_dir=v4, labels=labels, force=force)


def test_d2_seed_stage_writes_seeded_files_and_the_gate_subset(tmp_path):
    args = _seed_args(tmp_path)
    prep.stage_d2_seed(args)
    desc = [json.loads(x) for x in (args.labels / "desc_eval_mix_d2.jsonl").read_text().splitlines()]
    assert [r["id"] for r in desc] == ["evala"]
    changed = [json.loads(x) for x in (args.events_dir / "events_eval_changed.jsonl").read_text().splitlines()]
    assert [r["id"] for r in changed] == ["evalb"]
    objs = [json.loads(x)["object_id"] for x in (args.events_dir / "objects_eval_changed.jsonl").read_text().splitlines()]
    assert set(objs) == {"f:b1", "f:b2"}
    meta = json.loads((args.events_dir / "seed_meta.json").read_text())
    assert meta["eval"]["reused"] == 1 and meta["eval"]["to_compose"] == 1 and meta["eval"]["changed_with_turn"] == 1


def test_d2_seed_refuses_to_overwrite_label_work(tmp_path):   # Review Focus 3
    args = _seed_args(tmp_path)
    prep.stage_d2_seed(args)
    with pytest.raises(SystemExit, match="refusing"):
        prep.stage_d2_seed(args)
    prep.stage_d2_seed(_seed_args(tmp_path, force=True))


def test_finalize_recheck_drops_a_reused_text_that_breaks_a_d2_rule():
    rec = {"id": "a", "image": ["/x.png"], "conversations": [
        {"from": "human", "value": "<image>\n…"},
        {"from": "gpt", "value": json.dumps({"path": [[0.5, 1.0, 1], [0.5, 0.6, 1]]})}],
        "habitat_meta": {"sample_id": "s", "pair_id": None}}
    ev = {"a": {**_row("a", [PASS]), "pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 3]}}
    objects = {"f:b1": {"cells_xz": [[0.5, 1.0]]}}
    bad = {"a": {"id": "a", "passed": True, "description": "Go left of the sofa, then turn right.", "instruction_source": "cosmos"}}
    out = finalize_descriptions([rec], ev, bad, objects, recheck=turn_rejects)
    assert out["stats"]["recheck_failed"] == 1 and out["kept"] == []
    assert finalize_descriptions([rec], ev, bad, objects)["stats"]["kept"] == 1          # d1 behaviour unchanged


def test_desc_path_round():
    assert prep.desc_path(Path("L"), "eval", "_mix").name == "desc_eval_mix_d1.jsonl"
    assert prep.desc_path(Path("L"), "eval", "_mix", "d2").name == "desc_eval_mix_d2.jsonl"
