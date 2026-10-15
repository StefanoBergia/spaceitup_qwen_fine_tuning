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
    _write(lab / "names_steer_gemma_d2.jsonl", [{"id": "s600:b1", "name": "glass door", "passed": True}])
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
