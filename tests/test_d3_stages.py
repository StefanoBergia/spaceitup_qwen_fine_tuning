"""d3 stages (spec 2026-11-10-stop-phrases-d3-design.md "Data flow"): events --stops d3 into v5, d3-seed rewriting
the d2 stop clauses, finalize --round d3 re-checking every text."""
import json
from argparse import Namespace

import pytest

from rover_vlm.descriptions import rewrite_ending, rewrite_stop, seed_d3_labels
from test_d2_stages import _labelset, prep

PASS = {"type": "pass", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
TURN = {"type": "turn", "side": "left", "relation": "behind", "objects": ["f:b1"], "arc_m": 3.0,
        "turn_xz": [0, 3], "pre_heading_xz": [0, 1], "gt_lateral_m": 1.0}
DIST = {"type": "distance", "relation": "past", "amount": "well", "distance_m": 4.0, "object": "f:b1"}
BEHIND = {"type": "behind", "relation": "behind", "amount": None, "distance_m": None, "object": "f:b1",
          "disappear_arc_m": 2.4}
ROOM = {"type": "place", "relation": None, "amount": None, "distance_m": None, "object": None, "place": "room_middle",
        "measures": {"free_share": 0.9, "clear_m": 1.5}}
CORNER = {**ROOM, "place": "corner", "measures": {}}
NAMES = {"f:b1": "sofa"}


def _row(rid, events, stop, reason=None):
    return {"id": rid, "reason": reason, "events": events, "stop": stop if reason is None else None,
            "pair_id": None, "kind": "instructed", "pose_xz": [2.5, 0.2], "heading_xz": [0, 1], "goal_xz": [2.5, 3.0]}


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_rewrite_stop_turns_a_distance_stop_into_a_place():
    assert rewrite_stop("Go left of the sofa, then stop well past the sofa.", [PASS], ROOM, NAMES) == \
        ("Go left of the sofa, then stop in the middle of the room.", None)


def test_rewrite_stop_replaces_only_the_final_clause():                # Review Focus 4
    text, why = rewrite_stop("Go left of the sofa without stopping, then stop well past the sofa.", [PASS], ROOM, NAMES)
    assert why is None and text == "Go left of the sofa without stopping, then stop in the middle of the room."


def test_rewrite_stop_reports_its_reasons():
    assert rewrite_stop("Go left of the sofa.", [PASS], ROOM, NAMES) == (None, "no_stop_clause")
    assert rewrite_stop("Stop well past the sofa.", [PASS], ROOM, NAMES) == (None, "no_connector")


def test_rewrite_ending_unchanged():
    text, why = rewrite_ending("Go left of the sofa, then stop well past the sofa.", [PASS], BEHIND, NAMES, "r1")
    assert why is None and text.startswith("Go left of the sofa, then ") and text.endswith(" behind it.")


def _names():
    return {"gemma": [{"id": "f:b1", "name": "sofa", "passed": True}], "cosmos": []}


def test_seed_d3_reuses_rewrites_and_lists_changed():
    old = [_row("a", [PASS], BEHIND), _row("b", [PASS], DIST), _row("c", [PASS], DIST)]
    new = [_row("a", [PASS], BEHIND), _row("b", [PASS], ROOM), _row("c", [PASS, TURN], ROOM)]
    desc = [{"id": "a", "description": "Go left of the sofa, then carry on behind it.", "passed": True,
             "rewritten_from": "d1"},
            {"id": "b", "description": "Go left of the sofa, then stop well past the sofa.", "passed": True},
            {"id": "c", "description": "Go left of the sofa, then stop well past the sofa.", "passed": True}]
    s = seed_d3_labels(old, new, [{"object_id": "f:b1"}], _names(), desc)
    by = {r["id"]: r for r in s["desc"]}
    assert by["a"]["reused_from"] == "d2"
    assert by["b"]["rewritten_from"] == "d2" and by["b"]["description"].endswith("stop in the middle of the room.")
    assert [r["id"] for r in s["changed"]] == ["c"] and s["rewrite_failed"] == {"events_changed": 1}


def test_seed_d3_counts_rewrite_failures():
    old, new = [_row("b", [PASS], DIST)], [_row("b", [PASS], ROOM)]
    desc = [{"id": "b", "description": "Stop well past the sofa.", "passed": True}]
    s = seed_d3_labels(old, new, [{"object_id": "f:b1"}], _names(), desc)
    assert [r["id"] for r in s["changed"]] == ["b"] and s["rewrite_failed"] == {"no_connector": 1}


def test_events_stops_d3_stamps_schema_and_stop_kinds(tmp_path):
    _labelset(tmp_path / "v2", tmp_path)
    out = tmp_path / "v5"
    prep.stage_events(Namespace(out_dir=out, labelset_dir=tmp_path / "v2", limit=0, force=False, stops="d3"))
    meta = json.loads((out / "events_meta.json").read_text())
    assert meta["schema"] == "d3-stops"
    for s in ("train", "eval"):
        assert sum(meta[s]["stop_kinds"].values()) == 1 and meta[s]["visible_drop_share"] == 0.0
    row = _read(out / "events_eval.jsonl")[0]
    assert row["stop"]["type"] in ("object", "place")


def test_events_refuses_to_overwrite_d2_events_with_d3(tmp_path):
    _labelset(tmp_path / "v2", tmp_path)
    out = tmp_path / "v4"
    prep.stage_events(Namespace(out_dir=out, labelset_dir=tmp_path / "v2", limit=0, force=False))
    with pytest.raises(SystemExit, match="refusing"):
        prep.stage_events(Namespace(out_dir=out, labelset_dir=tmp_path / "v2", limit=0, force=False, stops="d3"))


def _d3_seed_args(tmp_path, force=False):
    v4, v5, labels = tmp_path / "v4", tmp_path / "v5", tmp_path / "labels"
    for s in ("train", "eval"):
        _write(v4 / f"events_{s}.jsonl", [_row(f"{s}a", [PASS], DIST), _row(f"{s}b", [PASS], DIST)])
        _write(v5 / f"events_{s}.jsonl", [_row(f"{s}a", [PASS], ROOM), _row(f"{s}b", [PASS, TURN], ROOM)])
        _write(v5 / f"objects_{s}.jsonl", [{"object_id": "f:b1"}])
        _write(labels / f"names_{s}_gemma_d2.jsonl", [{"id": "f:b1", "name": "sofa", "passed": True}])
        _write(labels / f"names_{s}_d2.jsonl", [])
        _write(labels / f"desc_{s}_mix_d2.jsonl",
               [{"id": f"{s}{k}", "description": "Go left of the sofa, then stop well past the sofa.", "passed": True}
                for k in "ab"])
    return Namespace(d2_events_dir=v4, events_dir=v5, labels=labels, force=force)


def test_d3_seed_stage_writes_d3_files(tmp_path):
    args = _d3_seed_args(tmp_path)
    prep.stage_d3_seed(args)
    desc = _read(args.labels / "desc_eval_mix_d3.jsonl")
    assert [(r["id"], r.get("rewritten_from")) for r in desc] == [("evala", "d2")]
    assert [r["id"] for r in _read(args.events_dir / "events_eval_changed.jsonl")] == ["evalb"]
    assert [r["id"] for r in _read(args.labels / "names_eval_gemma_d3.jsonl")] == ["f:b1"]
    meta = json.loads((args.events_dir / "seed_meta.json").read_text())
    assert meta["eval"]["rewritten"] == 1 and meta["eval"]["to_compose"] == 1


def test_d3_seed_refuses_to_overwrite_label_work(tmp_path):
    args = _d3_seed_args(tmp_path)
    prep.stage_d3_seed(args)
    with pytest.raises(SystemExit, match="refusing"):
        prep.stage_d3_seed(args)
    prep.stage_d3_seed(_d3_seed_args(tmp_path, force=True))


def _rec(rid, sample_id, scene):
    return {"id": rid, "image": ["/x.png"], "conversations": [
        {"from": "human", "value": "<image>\n…"},
        {"from": "gpt", "value": json.dumps({"path": [[0.5, 1.0, 1], [0.5, 0.6, 1]]})}],
        "habitat_meta": {"sample_id": sample_id, "scene_id": scene, "pair_id": None}}


def test_finalize_round_d3_rechecks_with_d3_rejects(tmp_path):
    lab, v5, labels = tmp_path / "v2", tmp_path / "v5", tmp_path / "labels"
    lab.mkdir()
    good = "Go left of the sofa, then stop in the corner."
    splits = {"train": [("t1", "00001-a_c000", "00001-a", good),
                        ("t2", "00002-b_c000", "00002-b", "Go left of the sofa, then stop well past the sofa.")],
              "eval": [("e1", "00800-c_c000", "00800-c", good), ("e2", "00801-d_c000", "00801-d", good)]}
    for s, rows in splits.items():
        (lab / f"labelset_{s}.json").write_text(json.dumps([_rec(i, sid, sc) for i, sid, sc, _ in rows]))
        _write(v5 / f"events_{s}.jsonl", [_row(i, [PASS], CORNER) for i, *_ in rows])
        _write(v5 / f"objects_{s}.jsonl", [{"object_id": "f:b1", "cells_xz": [[2.0, 1.0]]}])
        _write(labels / f"desc_{s}_mix_d3.jsonl", [{"id": i, "description": t, "passed": True,
                                                    "instruction_source": "cosmos"} for i, *_, t in rows])
    prep.stage_finalize_descriptions(Namespace(round="d3", events_dir=v5, labelset_dir=lab, out_dir=v5, labels=labels,
                                               desc_suffix="_mix", allow_missing=False))
    meta = json.loads((v5 / "meta_d3.json").read_text())
    assert meta["train"]["recheck_failed"] == 1 and meta["train"]["kept"] == 1 and meta["eval"]["kept"] == 2


# --- Task 7: stop audit page (G1) and the d3 label-audit picker (G3) -------------------------------------------------
from test_d2_stages import audit   # noqa: E402

KINDS = ["object:behind", "object:in front of", "object:next to", "place:doorway", "place:corner",
         "place:hallway_end", "place:wall", "place:room_middle"]


def _kind_stop(kind):
    t, v = kind.split(":")
    return {**ROOM, "place": v} if t == "place" else \
        {"type": "object", "relation": v, "amount": None, "distance_m": None, "object": "f:b1", "place": None}


def test_select_d3_rows_takes_d2_rewrites_and_d6_composed_rows():
    ev, ok = {}, []
    for i in range(6):
        ok.append({"id": f"rw{i}", "rewritten_from": "d2", "passed": True})
        ev[f"rw{i}"] = _row(f"rw{i}", [PASS], ROOM)
    for i in range(3):
        ok.append({"id": f"ru{i}", "rewritten_from": "d1", "reused_from": "d2", "passed": True})
        ev[f"ru{i}"] = _row(f"ru{i}", [PASS], BEHIND)
    for kind, tag in (("object:next to", "o"), ("place:corner", "p")):
        for i in range(10):
            ok.append({"id": f"{tag}{i}", "prompt_version": "d6", "passed": True})
            ev[f"{tag}{i}"] = _row(f"{tag}{i}", [PASS], _kind_stop(kind))
    ok.append({"id": "old", "prompt_version": "d5", "passed": True})
    ev["old"] = _row("old", [PASS], ROOM)
    rows = audit.select_d3_rows(ok, ev, 10, seed=0)
    ids = [r["id"] for r in rows]
    assert len(rows) == 10 and not any(i.startswith("ru") or i == "old" for i in ids)
    assert sum(i.startswith("rw") for i in ids) == 4
    assert sum(i.startswith("o") for i in ids) == 3 and sum(i.startswith("p") for i in ids) == 3


def _kind_rows():
    rows = [_row(f"{k}-{i}", [PASS], _kind_stop(k)) for k in KINDS for i in range(15)]
    rows += [_row(f"drop-{i}", [PASS], None, reason="no_stop_place") for i in range(20)]
    rows += [_row(f"hid-{i}", [PASS], BEHIND) for i in range(5)]
    return rows


def test_sample_stops_covers_every_kind():
    picked, pop = audit.sample_stops(_kind_rows(), seed=0, n=100, min_per_kind=10, n_drop=10)
    groups = [g for g, _ in picked]
    assert all(groups.count(k) >= 10 for k in KINDS) and groups.count("drop") == 10
    assert len(picked) == 110 and "behind" not in groups
    assert pop["drop"] == 20 and pop["place:corner"] == 15
    again, _ = audit.sample_stops(_kind_rows(), seed=0, n=100, min_per_kind=10, n_drop=10)
    assert [r["id"] for _, r in again] == [r["id"] for _, r in picked]


def test_score_stops_reports_each_kind_and_the_gate():
    items = [{"id": "a", "group": "place:corner"}, {"id": "b", "group": "place:corner"},
             {"id": "c", "group": "object:next to"}, {"id": "d", "group": "object:next to"},
             {"id": "e", "group": "drop"}]
    marks = {"a": {"verdict": "ok"}, "b": {"verdict": "place"}, "c": {"verdict": "ok"}, "d": {"verdict": "unsure"},
             "e": {"verdict": "ok"}}
    s = audit.score_stops(items, marks)
    assert s["kinds"]["place:corner"] == {"n": 2, "ok": 1, "rate": 0.5}
    assert s["kinds"]["object:next to"] == {"n": 1, "ok": 1, "rate": 1.0}          # "can't tell" is not counted
    assert s["all"] == {"n": 4, "ok": 3, "rate": 0.75} and s["gate"] == "FAIL"


def test_stops_item_shows_the_place(tmp_path):
    from PIL import Image
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    rec = {"id": "r1", "image": [str(img)], "conversations": [{"from": "human", "value": ""},
           {"from": "gpt", "value": json.dumps({"path": [[0.5, 1.0, 1], [0.5, 0.6, 1]]})}],
           "habitat_meta": {"sample_id": "s", "scene_id": "x", "kind": "instructed"}}
    row = _row("r1", [PASS], CORNER)
    it = audit.stops_item(rec, row, {"f:b1": {"polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]], "kind_hint": "object"}},
                          "place:corner")
    assert it["group"] == "place:corner" and it["instruction"].endswith("stop in the corner.")
    assert "corner" in it["caption"] and it["img"].startswith("data:image/jpeg;base64,")
    drop = audit.stops_item(rec, _row("r1", [], None, reason="no_stop_place"), {}, "drop")
    assert drop["instruction"].startswith("DROPPED")


def test_audit_template_has_a_stops_mode():
    html = audit.TEMPLATE.read_text()
    assert 'DATA.mode === "stops"' in html and "Stop Phrases Audit" in html


def test_d3_seed_seeds_the_steer_names(tmp_path):                       # final review I3
    args = _d3_seed_args(tmp_path)
    _write(args.labels / "names_steer_gemma_d2.jsonl", [{"id": "f:b9", "name": "lamp", "passed": True},
                                                       {"id": "f:b8", "error": "timeout"}])
    prep.stage_d3_seed(args)
    assert [r["id"] for r in _read(args.labels / "names_steer_gemma_d3.jsonl")] == ["f:b9"]
