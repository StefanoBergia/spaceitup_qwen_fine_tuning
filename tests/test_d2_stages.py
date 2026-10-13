"""d2 stages (spec 2026-10-13 "Data flow"): events from the v2 labelsets, seeding the d2 label files from d1,
finalize --round d2 re-checking every text."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from rover_vlm.descriptions import finalize_descriptions, same_description_geometry, seed_d2_labels, turn_rejects

_SPEC = importlib.util.spec_from_file_location(
    "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
prep = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_habitat_instr"] = prep      # the stage's Pool pickles event_row by module name
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


_ASPEC = importlib.util.spec_from_file_location(
    "audit_instruction_labels", Path(__file__).resolve().parent.parent / "scripts" / "audit_instruction_labels.py")
audit = importlib.util.module_from_spec(_ASPEC)
_ASPEC.loader.exec_module(audit)


def _ev(rid, reason=None, rel=None):
    events = [] if rel is False else [dict(TURN, relation=rel)]
    return {"id": rid, "reason": reason, "events": events if reason is None else [], "pair_id": None}


def test_sample_turns_stratifies():
    d2 = ([_ev(f"h{i}", rel="behind") for i in range(40)] + [_ev(f"v{i}", rel=None) for i in range(10)]
          + [_ev(f"r{i}", rel="round") for i in range(5)] + [_ev(f"b{i}", rel=False) for i in range(30)]
          + [_ev(f"s{i}", reason="spur") for i in range(50)])
    d1 = [{"id": f"b{i}", "reason": "no_events_bent"} for i in range(25)]
    picked, pop = audit.sample_turns(d2, d1, seed=0)
    groups = [g for g, _ in picked]
    assert groups.count("turn_visible") == 15 and groups.count("turn_hidden") == 35   # 20 visible wanted, 15 exist
    assert groups.count("bent") == 20 and groups.count("spur") == 30
    assert pop == {"turn_visible": 15, "turn_hidden": 40, "rescued_bent": 25, "spur": 50}


def test_select_d2_rows_half_with_a_turn():
    ok = [{"id": f"t{i}", "prompt_version": "d4"} for i in range(80)] + \
         [{"id": f"n{i}", "prompt_version": "d4"} for i in range(80)] + [{"id": "old", "prompt_version": "d3"}]
    ev = {**{f"t{i}": {"events": [TURN]} for i in range(80)}, **{f"n{i}": {"events": [PASS]} for i in range(80)},
          "old": {"events": [PASS]}}
    rows = audit.select_d2_rows(ok, ev, 100, seed=0)
    assert len(rows) == 100 and sum(r["id"].startswith("t") for r in rows) == 50 and "old" not in {r["id"] for r in rows}


def test_select_d2_rows_empty_no_turn_pool_still_returns_n():
    ok = [{"id": f"t{i}", "prompt_version": "d4"} for i in range(156)]
    ev = {f"t{i}": {"events": [TURN]} for i in range(156)}
    rows = audit.select_d2_rows(ok, ev, 100, seed=0)
    assert len(rows) == 100 and all(r["id"].startswith("t") for r in rows)


def test_sample_steer_takes_turn_pairs_first():
    rows = [{"pair_id": f"t{i}", "id": f"t{i}_{k}", "slice": "turn"} for i in range(30) for k in "AB"] + \
           [{"pair_id": f"s{i}", "id": f"s{i}_{k}", "slice": "stop"} for i in range(60) for k in "AB"]
    picked, pop = audit.sample_steer(rows, 50, 0, n_turn=20)
    assert len(picked) == 50 and sum(p[0]["slice"] == "turn" for p in picked) == 20 and pop == 90


def test_audit_template_has_a_turns_mode():
    html = audit.TEMPLATE.read_text()
    assert 'DATA.mode === "turns"' in html and "Turn Events Audit" in html
    assert "red ring" in html and "spur drop" in html and "rescued bent route" in html
    assert 'DATA.mode === "turns" || !it.has_alt' in html     # the alt legend stays hidden for turns


def _labelset(src, tmp_path):
    """One synthetic labelset record per split, built like tests/test_route_turns.py's write_sample."""
    from test_route_turns import L_LEFT, _round_state, write_sample
    src.mkdir(parents=True)
    for split in ("train", "eval"):
        sd = write_sample(tmp_path / f"s_{split}", _round_state(), [L_LEFT])
        rec = {"id": f"{split}_0", "image": [str(sd / "fpv_enhanced.png")],
               "habitat_meta": {"candidate_index": 0, "kind": "instructed", "pair_id": None, "sample_id": split}}
        (src / f"labelset_{split}.json").write_text(json.dumps([rec]))


def test_events_writes_into_out_dir_from_a_labelset_elsewhere(tmp_path):
    _labelset(tmp_path / "v2", tmp_path)
    out = tmp_path / "v4"
    prep.stage_events(Namespace(out_dir=out, labelset_dir=tmp_path / "v2", limit=0, force=False))
    meta = json.loads((out / "events_meta.json").read_text())
    assert meta["schema"] == prep.EVENTS_SCHEMA and meta["labelset_dir"] == str(tmp_path / "v2")
    assert (out / "events_eval.jsonl").exists() and (out / "objects_train.jsonl").exists()
    assert not list((tmp_path / "v2").glob("events_*"))                 # the labelset's dir is untouched
    row = json.loads((out / "events_eval.jsonl").read_text().splitlines()[0])
    assert row["reason"] is None and [e["type"] for e in row["events"]] == ["turn"]


def test_events_refuses_to_overwrite_d1_events_unless_forced(tmp_path):
    _labelset(tmp_path / "v2", tmp_path)
    d1 = tmp_path / "v2"
    (d1 / "events_meta.json").write_text(json.dumps({"schema": "d1-addendumA2"}))
    args = Namespace(out_dir=d1, labelset_dir=None, limit=0, force=False)
    with pytest.raises(SystemExit, match="refusing"):
        prep.stage_events(args)
    assert not (d1 / "events_eval.jsonl").exists()
    prep.stage_events(Namespace(out_dir=d1, labelset_dir=None, limit=0, force=True))
    assert json.loads((d1 / "events_meta.json").read_text())["schema"] == prep.EVENTS_SCHEMA
