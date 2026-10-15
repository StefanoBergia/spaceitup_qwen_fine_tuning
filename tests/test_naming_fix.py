"""Object naming fix (spec 2026-10-15): prompt n2, the naming sample, the variants job, the names audit, renaming."""
import json
from pathlib import Path

import pytest

from rover_vlm.descriptions import label_object, naming_prompt

REPO = Path(__file__).resolve().parent.parent


def _norm(p):
    return " ".join(p.split())


def _reply(payload):
    return f"<answer>{json.dumps(payload)}</answer>"


@pytest.mark.parametrize("need", ["opening", "wall", "object", "anything"])
def test_n1_is_the_default_and_unchanged(need):
    assert naming_prompt(need) == naming_prompt(need, "n1")


def test_n2_wording():
    p = _norm(naming_prompt("wall", "n2"))
    assert "Name the single thing that fills most of the outline" in p
    assert "Look at what is inside the outline, not at what is next to it or behind it." in p
    assert "If the outline is mostly part of the building, say which part" in p
    assert "Name furniture or an appliance only if the outline is clearly on it" in p
    assert "such as a sofa or a counter" not in p
    assert "something long that the rover drives alongside or stops in front of" in p
    assert '"wall" (part of the building: a wall, a column or pillar, a door, a partition, a railing)' in p
    assert "Never use the words left or right" in p and "not a small part on it" in p
    assert "The rover drives through it." in _norm(naming_prompt("opening", "n2"))
    assert "something the rover drives past" in _norm(naming_prompt("object", "n2"))
    assert "something the rover drives past" in _norm(naming_prompt("anything", "n2"))
    with pytest.raises(ValueError):
        naming_prompt("object", "n9")


def test_label_object_uses_and_records_the_prompt_version():
    seen = []
    obj = {"object_id": "f:b1", "kind_hint": "wall", "image": None, "polygon": [], "crop_box": None}

    def ask(img, prompt, seed):
        seen.append(prompt)
        return _reply({"name": "white wall", "kind": "wall"}), "stop"
    row = label_object(ask, obj, seed=0, image=b"x", name_prompt="n2")
    assert row["passed"] and row["name_prompt"] == "n2" and seen[0] == naming_prompt("wall", "n2")
    assert label_object(ask, obj, seed=0, image=b"x")["name_prompt"] == "n1"


def test_label_descriptions_has_the_flag():
    text = (REPO / "scripts" / "label_descriptions.py").read_text()
    assert '"--name-prompt"' in text and 'choices=["n1", "n2"]' in text
    assert "name_prompt=args.name_prompt" in text


import importlib.util  # noqa: E402
import sys  # noqa: E402
from argparse import Namespace  # noqa: E402

# one module object per session: test_d2_stages registers it too, and its stages' Pool pickles functions by module name
prep = sys.modules.get("prepare_habitat_instr")
if prep is None:
    _SPEC = importlib.util.spec_from_file_location("prepare_habitat_instr", REPO / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(_SPEC)
    sys.modules["prepare_habitat_instr"] = prep
    _SPEC.loader.exec_module(prep)


def _obj(i, kind):
    return {"object_id": f"s{i}:b1", "kind_hint": kind, "image": f"/x/{i}.png", "polygon": [[0, 0], [1, 0], [1, 1]],
            "bbox": [0, 0, 1, 1], "crop_box": [0, 0, 1, 1], "area": 0.5, "depth_fallback": False}


def test_naming_sample_is_stratified_reproducible_and_adds_flagged():
    objs = [_obj(i, "wall") for i in range(60)] + [_obj(100 + i, "object") for i in range(60)] + \
           [_obj(200 + i, "opening") for i in range(5)]
    flagged = [dict(_obj(999, "object"))]
    rows = prep.naming_sample_rows(objs, prep.NAMING_STRATA, 0, flagged)
    rnd = [r for r in rows if r["group"] == "random"]
    assert [sum(r["kind_hint"] == k for r in rnd) for k in ("wall", "object", "opening")] == [45, 45, 5]  # 5 openings exist
    assert [r["object_id"] for r in rows if r["group"] == "flagged"] == ["s999:b1"]
    assert rows == prep.naming_sample_rows(objs, prep.NAMING_STRATA, 0, flagged)
    assert rows != prep.naming_sample_rows(objs, prep.NAMING_STRATA, 1, flagged)


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _stage_args(tmp_path, flagged_ids):
    ev, lab = tmp_path / "v4", tmp_path / "labels"
    _write(ev / "objects_train.jsonl", [_obj(i, "wall") for i in range(50)] + [_obj(100 + i, "object") for i in range(50)]
           + [_obj(200 + i, "opening") for i in range(12)])
    _write(ev / "objects_eval.jsonl", [_obj(500, "object")])
    _write(ev / "steer" / "objects_eval.jsonl", [_obj(600, "wall")])
    _write(lab / "names_train_gemma_d2.jsonl", [{"id": "s0:b1", "name": "white wall", "passed": True},
                                                 {"id": "s100:b1", "name": None, "passed": False}])
    _write(lab / "names_train_d2.jsonl", [{"id": "s100:b1", "name": "grey sofa", "passed": True}])
    _write(lab / "names_eval_gemma_d2.jsonl", [{"id": "s500:b1", "name": "shelf", "passed": True}])
    _write(lab / "names_gate_gemma_d2.jsonl", [{"id": "s500:b1", "name": "bookshelf", "passed": True}])  # gate wins
    _write(lab / "names_eval_d2.jsonl", [])                     # names_gate_d2.jsonl missing: tolerated
    _write(lab / "names_steer_gemma_d2.jsonl", [{"id": "s600:b1", "name": "glass door", "passed": True},
                                                 {"id": "s500:b1", "name": "wall artwork", "passed": True}])  # gate still wins
    fl = tmp_path / "flagged.txt"
    fl.write_text("\n".join(flagged_ids) + "\n")
    return Namespace(events_dir=ev, labels=lab, out_dir=tmp_path / "ab", n=100, seed=0, flagged=fl)


def test_naming_sample_stage_writes_sample_names_v0_and_flagged(tmp_path):
    args = _stage_args(tmp_path, ["s500:b1", "s600:b1", "s100:b1"])
    prep.stage_naming_sample(args)
    rows = [json.loads(x) for x in (args.out_dir / "objects_sample.jsonl").read_text().splitlines()]
    v0 = {r["id"]: r for r in map(json.loads, (args.out_dir / "names_V0.jsonl").read_text().splitlines())}
    rnd = [r["object_id"] for r in rows if r["group"] == "random"]
    assert len(rnd) == 100 and "s100:b1" not in rnd                 # a flagged object is never also a random one
    assert [r["object_id"] for r in rows if r["group"] == "flagged"] == ["s500:b1", "s600:b1", "s100:b1"]
    assert v0["s500:b1"]["name"] == "bookshelf" and v0["s600:b1"]["name"] == "glass door"
    assert v0["s100:b1"] == {"id": "s100:b1", "name": "grey sofa", "passed": True, "namer": "cosmos"}
    assert (args.out_dir / "flagged_ids.txt").read_text().split() == ["s500:b1", "s600:b1", "s100:b1"]


def test_naming_sample_stops_on_unknown_flagged_ids(tmp_path):            # Review Focus 5
    with pytest.raises(SystemExit, match="nope:b9"):
        prep.stage_naming_sample(_stage_args(tmp_path, ["s500:b1", "nope:b9"]))


def test_naming_variants_sbatch():
    text = (REPO / "slurm" / "naming_variants.sbatch").read_text()
    assert "#SBATCH --gres=gpu:a100:1" in text and "#SBATCH --partition=A100" in text
    for model in ("google/gemma-3-12b-it", "Qwen/Qwen3.6-35B-A3B-FP8", "nvidia/Cosmos-Reason2-32B"):
        assert model in text
    assert 'VARIANTS="${VARIANTS:-V1 V2 V3}"' in text and 'SETS="${SETS:-sample}"' in text
    assert 'NAME_PROMPT="${NAME_PROMPT:-n2}"' in text and '--name-prompt "$NAME_PROMPT"' in text
    assert "( cosmos3_wait_healthy )" in text            # a model that fails to start is skipped (Review Focus 1)
    assert "names_${v}.jsonl" in text and 'names_${s}_${v}_${NAME_PROMPT}.jsonl' in text


_ASPEC = importlib.util.spec_from_file_location("audit_instruction_labels", REPO / "scripts" / "audit_instruction_labels.py")
audit = importlib.util.module_from_spec(_ASPEC)
_ASPEC.loader.exec_module(audit)

SAMPLE = [{"object_id": "a", "kind_hint": "wall", "group": "random"}, {"object_id": "b", "kind_hint": "object", "group": "random"},
          {"object_id": "c", "kind_hint": "object", "group": "random"}, {"object_id": "f", "kind_hint": "wall", "group": "flagged"}]
VNAMES = {"V0": {"a": "grey sofa", "b": "Coffee table", "c": "chair", "f": "bookshelf"},
          "V1": {"a": "white wall", "b": "coffee table", "c": "armchair", "f": "steel column"},
          "V2": {"a": "white wall", "b": None, "c": "armchair", "f": "bookshelf"}}


def test_names_items_list_distinct_names_shuffled(monkeypatch):
    monkeypatch.setattr(audit, "_naming_jpeg", lambda row: b"jpg")
    items = audit.names_items(SAMPLE, VNAMES, seed=0)
    by = {it["id"]: it for it in items}
    assert sorted(by["a"]["names"]) == ["grey sofa", "white wall"]
    assert sorted(n.lower() for n in by["b"]["names"]) == ["coffee table"]            # case-insensitive distinct
    assert by["f"]["group"] == "flagged" and by["a"]["img"].startswith("data:image/jpeg;base64,")


def test_score_names():
    marks = {"a": {"verdict": "done", "right": ["white wall"]}, "b": {"verdict": "done", "right": ["coffee table"]},
             "c": {"verdict": "done", "right": []},                                      # done, nothing ticked = none
             "f": {"verdict": "done", "right": ["steel column"]}}
    s = audit.score_names(SAMPLE, VNAMES, marks)
    assert s["random"]["V0"]["correct"] == 1 and s["random"]["V1"]["correct"] == 2
    assert s["random"]["V2"]["correct"] == 1                       # no name for "b" counts as wrong (Review Focus 1)
    assert s["random"]["V1"]["accuracy"] == pytest.approx(2 / 3)
    assert s["random"]["V1"]["by_kind"] == {"wall": {"n": 1, "correct": 1}, "object": {"n": 2, "correct": 1}}
    assert s["flagged"]["V1"] == {"n": 1, "correct": 1} and s["flagged"]["V0"] == {"n": 1, "correct": 0}
    assert (s["best"], s["gate"], s["unmarked"]) == ("V1", False, 0)


def test_score_names_none_and_unmarked():                          # Review Focus 4
    s = audit.score_names(SAMPLE, VNAMES, {"a": {"verdict": "none", "right": ["white wall"]}})
    assert s["random"]["V1"]["correct"] == 0 and s["unmarked"] == 3
    # names ticked but Done never pressed: the ticks count; a note-only mark is unmarked
    s = audit.score_names(SAMPLE, VNAMES, {"a": {"right": ["White Wall"]}, "b": {"note": "?"}})
    assert s["random"]["V1"]["correct"] == 1 and s["random"]["V1"]["n"] == 1 and s["unmarked"] == 3
    assert s["best_accuracy"] == 1.0 and s["gate"] is False          # no gate reading on a partly marked audit


def test_read_marks_takes_the_object_id_from_the_mark(tmp_path):
    d = tmp_path / "marks"
    d.mkdir()
    (d / "x.json").write_text(json.dumps({"oid": "00001-UVdNNRcVyV1_c000:b1", "verdict": "done", "right": ["sofa"]}))
    (d / "a.json").write_text(json.dumps({"verdict": "none"}))
    assert audit._read_marks(tmp_path) == {"00001-UVdNNRcVyV1_c000:b1": {"oid": "00001-UVdNNRcVyV1_c000:b1",
                                                                          "verdict": "done", "right": ["sofa"]},
                                           "a": {"verdict": "none"}}


def test_template_has_a_names_panel():
    html = audit.TEMPLATE.read_text()
    assert 'DATA.mode === "names"' in html and 'id="names"' in html and "right:" in html
    # the template's display rules beat the hidden attribute: the names box must hide in other modes, and the
    # route legend and side chip must go in names mode
    assert ".namepick[hidden] { display: none; }" in html
    assert '.legend").style.display = "none"' in html and '$("side").style.display = "none"' in html
    assert {v["v"] for v in audit.VERDICTS["names"]} == {"done", "none"}
    # a tick after "None is right" undoes it, so the page never shows both while scoring ignores the tick
    assert 'if (marks[it.id].verdict === "none") delete marks[it.id].verdict;' in html


PASS_A = {"type": "pass", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
BETWEEN = {"type": "between", "side": None, "objects": ["f:b1", "f:b2"], "arc_m": 1.0}
NEAR2 = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}


def _ev(rid, events, stop=NEAR2):
    return {"id": rid, "reason": None, "events": events, "stop": stop}


def test_substitute_names_swaps_and_overlaps():                    # Review Focus 2
    text = "Go between the sofa and the dining table, then stop in front of the bed."
    old = {"f:b1": "sofa", "f:b2": "dining table", "f:b3": "bed"}
    new = {"f:b1": "bed", "f:b2": "desk", "f:b3": "table"}
    assert prep.substitute_names(text, old, new) == "Go between the bed and the desk, then stop in front of the table."
    assert prep.substitute_names("Keep the Sofa on your left.", {"f:b1": "sofa"}, {"f:b1": "grey wall"}) == \
        "Keep the Grey wall on your left."                          # a capitalised mention stays capitalised
    assert prep.substitute_names("Pass the sofabed.", {"f:b1": "sofa"}, {"f:b1": "bed"}) == "Pass the sofabed."
    assert prep.substitute_names("Pass the table.", {"f:b1": "table", "f:b2": None}, {"f:b1": "desk"}) == "Pass the desk."


def test_rename_desc_rows_keeps_passing_texts_and_drops_clashes():
    ev = {"a": _ev("a", [PASS_A]), "b": _ev("b", [BETWEEN], stop={**NEAR2, "object": None, "type": "after_last",
                                                                  "relation": "just after"}), "c": _ev("c", [PASS_A])}
    desc = [{"id": "a", "description": "Go left of the sofa, then stop in front of the fridge.", "passed": True,
             "namer": "primary", "instruction_source": "cosmos"},
            {"id": "b", "description": "Go between the sofa and the fridge, then stop just after it.", "passed": True,
             "instruction_source": "cosmos"},
            {"id": "c", "description": None, "passed": False, "rejects": ["unnamed_object"]},     # recomposed
            {"id": "gone", "description": "Go left of the sofa.", "passed": True}]               # route not described now
    old = {"primary": {"f:b1": "sofa", "f:b2": "fridge"}, "fallback": {}}
    new_primary = {"f:b1": "white wall", "f:b2": "refrigerator"}
    kept, dropped = prep.rename_desc_rows(desc, ev, old, new_primary, {})
    assert [r["id"] for r in kept] == ["a", "b"] and dropped == {"not_passed": 1, "no_route": 1}
    assert kept[0]["description"] == "Go left of the white wall, then stop in front of the refrigerator."
    assert kept[0]["renamed"] == 2 and kept[0]["namer"] == "primary"
    clash = {"f:b1": "white cabinet", "f:b2": "tall cabinet"}           # two objects, one head noun
    kept, dropped = prep.rename_desc_rows(desc[:2], ev, {"primary": {}, "fallback": {}}, clash, {})
    assert kept == [] and dropped == {"name:clash": 2}                 # nor are the text's own names known: compose
    kept, dropped = prep.rename_desc_rows(desc[:2], ev, old, clash, {})
    assert [r["namer"] for r in kept] == ["fallback", "fallback"] and dropped == {}   # its own names still serve it
    kept, _ = prep.rename_desc_rows(desc[:2], ev, old, clash, {"f:b1": "sofa", "f:b2": "fridge"})  # old names serve
    assert [(r["description"], r["namer"], r["renamed"]) for r in kept] == [(desc[0]["description"], "fallback", 0),
                                                                            (desc[1]["description"], "fallback", 0)]


def _rename_fixture(tmp_path):
    lab, v4 = tmp_path / "labels", tmp_path / "v4"
    for s in ("train", "eval"):
        _write(lab / f"names_{s}_gemma_d2.jsonl", [{"id": "f:b1", "name": "sofa", "passed": True}])
        _write(lab / f"names_{s}_d2.jsonl", [])
        _write(lab / f"names_{s}_V2_n2.jsonl", [{"id": "f:b1", "name": "white wall", "passed": True}])
        _write(lab / f"desc_{s}_mix_d2.jsonl", [{"id": f"{s}a", "description": "Go left of the sofa, then stop in front of it.",
                                                  "passed": True}])
        _write(v4 / f"events_{s}.jsonl", [_ev(f"{s}a", [PASS_A], stop={**NEAR2, "object": "f:b1"})])
        _write(v4 / f"objects_{s}.jsonl", [{"object_id": "f:b1", "kind_hint": "wall"}])
    _write(v4 / "steer" / "objects_eval.jsonl", [])
    _write(lab / "names_steer_gemma_d2.jsonl", [])
    _write(lab / "desc_gate_mix_d2.jsonl", [])
    return Namespace(labels=lab, events_dir=v4, variant="V2", force=False, discard_later_work=False)


def test_rename_texts_refuses_a_second_run(tmp_path):              # Review Focus 3
    args = _rename_fixture(tmp_path)
    lab = args.labels
    with pytest.raises(SystemExit, match="names_steer_V2_n2"):        # the rollout job has not run: nothing touched
        prep.stage_rename_texts(args)
    assert not (lab / "names_train_gemma_d2.pre_n2.jsonl").exists()
    _write(lab / "names_steer_V2_n2.jsonl", [])
    prep.stage_rename_texts(args)
    got = [json.loads(x) for x in (lab / "desc_train_mix_d2.jsonl").read_text().splitlines()]
    assert got[0]["description"] == "Go left of the white wall, then stop in front of it."
    assert (lab / "names_train_gemma_d2.pre_n2.jsonl").exists()
    fb = {r["id"]: r for r in map(json.loads, (lab / "names_train_d2.jsonl").read_text().splitlines())}
    assert fb["f:b1"]["name"] == "sofa" and fb["f:b1"]["passed"]                 # today's name is the fallback
    with pytest.raises(SystemExit, match="already"):
        prep.stage_rename_texts(args)


def test_rename_texts_refuses_an_incomplete_winner(tmp_path):       # final review I2: a TIMEOUT / error rows
    args = _rename_fixture(tmp_path)
    _write(args.labels / "names_steer_V2_n2.jsonl", [])
    _write(args.labels / "names_train_V2_n2.jsonl", [{"id": "f:b1", "error": "ReadTimeout"}])
    with pytest.raises(SystemExit, match="train: 1 of 1 objects have no name row"):
        prep.stage_rename_texts(args)
    assert not list(args.labels.glob("*.pre_n2.jsonl"))              # nothing touched


def test_rename_desc_rows_keeps_a_text_its_own_names_still_serve():
    # written with the old Cosmos (fallback) names because the old Gemma names clashed; the winner clashes too and the
    # merged old fallback (Gemma over Cosmos) clashes again -- the names the text was written with still serve it
    ev = {"b": _ev("b", [BETWEEN], stop={**NEAR2, "object": None, "type": "after_last", "relation": "just after"})}
    desc = [{"id": "b", "description": "Go between the sofa and the fridge, then stop just after it.", "passed": True,
             "namer": "fallback"}]
    old = {"primary": {"f:b1": "white cabinet", "f:b2": "tall cabinet"}, "fallback": {"f:b1": "sofa", "f:b2": "fridge"}}
    kept, dropped = prep.rename_desc_rows(desc, ev, old, {"f:b1": "white wall", "f:b2": "brick wall"},
                                          {"f:b1": "white cabinet", "f:b2": "tall cabinet"})
    assert dropped == {} and [(r["description"], r["namer"], r["renamed"]) for r in kept] == \
        [(desc[0]["description"], "fallback", 0)]


def test_substitute_names_leaves_an_unchanged_longer_name_alone():     # final review I1
    old, new = {"f:b1": "wall", "f:b2": "wall shelf"}, {"f:b1": "brick wall", "f:b2": "wall shelf"}
    assert prep.substitute_names("Go left of the wall shelf, then stop in front of the wall.", old, new) == \
        "Go left of the wall shelf, then stop in front of the brick wall."
    old, new = {"f:b1": "tv", "f:b2": "tv stand"}, {"f:b1": "television", "f:b2": None}       # no new name: kept
    assert prep.substitute_names("Pass the tv stand and the tv.", old, new) == "Pass the tv stand and the television."


def test_rename_desc_rows_drops_a_stale_short_mention():          # final review (a shortened mention keeps the old noun)
    after_last = {**NEAR2, "object": None, "type": "after_last", "relation": "just after"}
    ev = {"a": _ev("a", [PASS_A], stop=after_last)}
    desc = [{"id": "a", "description": "Go left of the kitchen island. Stop just after the island.", "passed": True}]
    old = {"primary": {"f:b1": "kitchen island"}, "fallback": {}}
    kept, dropped = prep.rename_desc_rows(desc, ev, old, {"f:b1": "white counter"}, {})
    assert kept == [] and dropped == {"stale_name": 1}
    kept, dropped = prep.rename_desc_rows(desc, ev, old, {"f:b1": "white island"}, {})   # same head noun: still true
    assert [r["description"] for r in kept] == ["Go left of the white island. Stop just after the island."]


def test_rename_texts_force_redoes_but_never_discards_later_work(tmp_path):   # final review I3
    args = _rename_fixture(tmp_path)
    _write(args.labels / "names_steer_V2_n2.jsonl", [])
    desc = args.labels / "desc_train_mix_d2.jsonl"
    prep.stage_rename_texts(args)
    first = desc.read_text()
    args.force = True
    prep.stage_rename_texts(args)                                   # nothing changed since: redone from the backups
    assert desc.read_text() == first
    with desc.open("a") as f:                                       # the full job's compose --resume appended a row
        f.write(json.dumps({"id": "composed", "description": "Go left of the white wall.", "passed": True}) + "\n")
    with pytest.raises(SystemExit, match="desc_train_mix_d2.jsonl"):
        prep.stage_rename_texts(args)
    assert "composed" in desc.read_text()
    args.discard_later_work = True
    prep.stage_rename_texts(args)
    assert desc.read_text() == first


def test_naming_variants_sbatch_survives_a_failed_pass():         # final review: a server that dies after starting
    text = (REPO / "slurm" / "naming_variants.sbatch").read_text()
    assert text.count('--resume || echo "WARN $v $s pass') == 2
