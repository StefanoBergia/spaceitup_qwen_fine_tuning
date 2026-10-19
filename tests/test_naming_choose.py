"""Naming chooser (spec 2026-10-15 Addendum C)."""
import json

import pytest

from rover_vlm.descriptions import (candidate_names, choose_object, choose_prompt, parse_choice_reply,
                                    summarize_d1)


def _norm(p):
    return " ".join(p.split())


def _r(choice):
    return f"<answer>{json.dumps({'choice': choice})}</answer>"


def test_candidate_names_distinct_and_stable():
    vn = {"V0": {"o": "Grey Sofa"}, "V1": {"o": "grey sofa"}, "V2": {"o": "white wall"}, "V3": {"o": None}}
    got = candidate_names("o", vn)
    assert sorted(c["name"] for c in got) == ["Grey Sofa", "white wall"]
    assert next(c for c in got if c["name"] == "Grey Sofa")["from"] == ["V0", "V1"]
    assert got == candidate_names("o", vn)
    assert candidate_names("missing", vn) == []


def test_choose_prompt_c1():
    p = _norm(choose_prompt("wall", ["white wall", "kitchen island"]))
    assert "a. white wall" in p and "b. kitchen island" in p
    assert "Pick the one name that best names the single thing that fills most of the outline" in p
    assert "Look at what is inside the outline, not at what is next to it or behind it." in p
    assert 'If no name fits, answer "none".' in p
    assert "something long that the rover drives alongside or stops in front of" in p
    assert '{"choice": "<letter>|none"}' in p
    with pytest.raises(ValueError):
        choose_prompt("wall", ["x"], "c9")


def test_parse_choice_reply():                                       # Review Focus 1
    assert parse_choice_reply(_r("b"), 3) == "b"
    assert parse_choice_reply(_r("B"), 3) == "b"
    assert parse_choice_reply(_r("c. white wall"), 3) == "c"
    assert parse_choice_reply("<answer>b</answer>", 3) == "b"              # a bare answer counts
    assert parse_choice_reply(_r("none"), 3) == "none"
    assert parse_choice_reply(_r("None of these"), 3) == "none"
    for bad in ("d", "b, c", "white wall", ""):                            # past the list, two, a name, empty
        assert parse_choice_reply(_r(bad), 3) is None, bad
    assert parse_choice_reply("I think it is a wall", 3) is None


OBJ = {"object_id": "o", "kind_hint": "wall", "image": None, "polygon": [], "crop_box": None}
CANDS = [{"name": "white wall", "from": ["V2", "V3"]}, {"name": "grey sofa", "from": ["V0"]}]


def _ask(*replies):
    calls = []

    def ask(img, prompt, seed):
        calls.append(seed)
        return replies[min(len(calls), len(replies)) - 1], "stop"
    return ask, calls


def test_choose_object_pick_none_and_failure():
    ask, calls = _ask(_r("b"))
    row = choose_object(ask, OBJ, CANDS, "kitchen island", seed=0, image=b"x")
    assert (row["name"], row["name_source"], row["passed"], row["attempts"]) == ("grey sofa", ["V0"], True, 1)
    assert row["candidates"] == ["white wall", "grey sofa"] and row["need"] == "wall" and row["choose_prompt"] == "c1"
    ask, _ = _ask(_r("none"))
    row = choose_object(ask, OBJ, CANDS, "kitchen island", seed=0, image=b"x")
    assert (row["choice"], row["name"], row["name_source"], row["passed"]) == ("none", "kitchen island", ["V0_fallback"], True)
    ask, calls = _ask("garbage", "more garbage")
    row = choose_object(ask, OBJ, CANDS, "kitchen island", seed=0, image=b"x")
    assert (row["choice"], row["rejects"], row["attempts"], row["name"]) == (None, ["parse"], 2, "kitchen island")
    assert calls == [0, 1] and "raw" in row
    ask, _ = _ask("garbage")
    assert choose_object(ask, OBJ, CANDS, None, seed=0, image=b"x")["passed"] is False      # Review Focus 2


def test_choose_object_single_and_no_candidates_make_no_call():
    ask, calls = _ask(_r("a"))
    row = choose_object(ask, OBJ, CANDS[:1], "kitchen island", seed=0, image=b"x")
    assert (row["single"], row["choice"], row["attempts"], row["name"]) == (True, "a", 0, "white wall")
    row = choose_object(ask, OBJ, [], "kitchen island", seed=0, image=b"x")
    assert (row["name"], row["name_source"], row["passed"]) == ("kitchen island", ["V0_fallback"], True)
    assert calls == []


def test_summarize_choose():
    base = {"passed": True, "rejects": [], "single": False, "candidates": ["x", "y"]}
    rows = [{**base, "id": "1", "choice": "a", "name_source": ["V2"]},
            {**base, "id": "2", "choice": "none", "name_source": ["V0_fallback"]},
            {**base, "id": "3", "choice": None, "rejects": ["parse"], "name_source": ["V0_fallback"]},
            {**base, "id": "4", "choice": "a", "single": True, "candidates": ["x"], "name_source": ["V1", "V2"]},
            {**base, "id": "5", "choice": None, "candidates": [], "name_source": ["V0_fallback"]}]
    s = summarize_d1(rows, "choose")
    assert s["choices"] == {"picked": 1, "none": 1, "failed": 1, "single": 1, "no_candidates": 1}
    assert s["from"] == {"V2": 2, "V0_fallback": 3, "V1": 1}


import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
_LSPEC = importlib.util.spec_from_file_location("label_descriptions", REPO / "scripts" / "label_descriptions.py")
label = importlib.util.module_from_spec(_LSPEC)
_LSPEC.loader.exec_module(label)


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _cli_fixture(tmp_path):
    from PIL import Image
    png = tmp_path / "f.png"
    Image.new("RGB", (16, 16), (90, 90, 90)).save(png)
    objs = [{"object_id": f"o{i}", "kind_hint": "wall", "image": str(png), "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
             "bbox": [0.1, 0.1, 0.9, 0.9], "crop_box": [0.1, 0.1, 0.9, 0.9]} for i in range(4)]
    _write(tmp_path / "objects.jsonl", objs)
    _write(tmp_path / "V0.jsonl", [{"id": "o0", "name": "sofa", "passed": True}, {"id": "o2", "name": "old name", "passed": True}])
    _write(tmp_path / "V1.jsonl", [{"id": "o0", "name": "white wall", "passed": True}, {"id": "o1", "name": "door", "passed": True},
                                   {"id": "o2", "name": "bad", "passed": False}])
    return tmp_path


def _run(monkeypatch, tmp_path, answer, *extra, candidates=("V0", "V1"), calls=None):
    calls = [] if calls is None else calls

    def fake_chat(client, model, image, prompt, max_tokens=None, seed=None):
        calls.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return _r(answer), "stop"
    monkeypatch.setattr(label, "resolve_endpoint", lambda *a, **k: "http://x/v1")
    monkeypatch.setattr(label, "connect", lambda *a, **k: (None, "fake-model"))
    monkeypatch.setattr(label, "chat_with_image", fake_chat)
    monkeypatch.setattr(sys, "argv", ["label_descriptions.py", "--stage", "choose", "--objects", str(tmp_path / "objects.jsonl"),
                                      "--candidates", *[f"{v}={tmp_path / (v + '.jsonl')}" for v in candidates],
                                      "--out", str(tmp_path / "out.jsonl"), *extra])
    label.main()
    return calls


def test_choose_stage_writes_one_row_per_object(monkeypatch, tmp_path):
    calls = _run(monkeypatch, _cli_fixture(tmp_path), "a")
    rows = {r["id"]: r for r in _rows(tmp_path / "out.jsonl")}
    assert set(rows) == {"o0", "o1", "o2", "o3"} and len(calls) == 1    # only o0 has two candidates
    assert all(r["chooser"] == "fake-model" for r in rows.values())
    assert rows["o0"]["name"] in ("sofa", "white wall") and rows["o0"]["attempts"] == 1
    assert rows["o1"]["name"] == "door" and rows["o1"]["single"]
    assert (rows["o2"]["name"], rows["o2"]["name_source"]) == ("old name", ["V0"])   # V0 is a candidate (V1's failed)
    assert rows["o3"]["passed"] is False and rows["o3"]["candidates"] == []          # Review Focus 2: named nowhere


def test_choose_stage_resumes_after_errors(monkeypatch, tmp_path):     # Review Focus 3
    _run(monkeypatch, _cli_fixture(tmp_path), RuntimeError("server gone"))
    assert [r["id"] for r in _rows(tmp_path / "out.jsonl") if r.get("error")] == ["o0"]
    _run(monkeypatch, tmp_path, "b", "--resume")
    rows = _rows(tmp_path / "out.jsonl")
    assert not any(r.get("error") for r in rows) and {r["id"] for r in rows} == {"o0", "o1", "o2", "o3"}
    assert next(r for r in rows if r["id"] == "o0")["passed"]


def test_choose_stage_needs_v0(monkeypatch, tmp_path):               # Review Focus 4
    calls = []
    with pytest.raises(SystemExit, match="V0"):
        _run(monkeypatch, _cli_fixture(tmp_path), "a", candidates=("V1",), calls=calls)
    assert calls == [] and not (tmp_path / "out.jsonl").exists()


from argparse import Namespace  # noqa: E402

prep = sys.modules.get("prepare_habitat_instr")      # one module object per session (test_d2_stages' Pool pickles by name)
if prep is None:
    _PSPEC = importlib.util.spec_from_file_location("prepare_habitat_instr", REPO / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(_PSPEC)
    sys.modules["prepare_habitat_instr"] = prep
    _PSPEC.loader.exec_module(prep)


def _o(oid, kind="wall"):
    return {"object_id": oid, "kind_hint": kind, "image": "/x.png", "polygon": [], "bbox": [0, 0, 1, 1], "crop_box": None}


def test_today_names_stage_covers_every_object(tmp_path):
    v4, lab = tmp_path / "v4", tmp_path / "labels"
    _write(v4 / "objects_train.jsonl", [_o("t1"), _o("t2", "object")])
    _write(v4 / "objects_eval.jsonl", [_o("e1"), _o("e2")])
    _write(v4 / "steer" / "objects_eval.jsonl", [_o("s1")])
    _write(lab / "names_train_gemma_d2.jsonl", [{"id": "t1", "name": "white wall", "passed": True}])
    _write(lab / "names_train_d2.jsonl", [{"id": "t2", "name": "grey sofa", "passed": True}])        # Cosmos only
    _write(lab / "names_eval_gemma_d2.jsonl", [])
    _write(lab / "names_gate_gemma_d2.jsonl", [{"id": "e1", "name": "tall dark cabinet", "passed": True}])
    _write(lab / "names_steer_gemma_d2.jsonl", [{"id": "e1", "name": "wall artwork", "passed": True},  # gate wins
                                                 {"id": "s1", "name": "glass door", "passed": True}])
    prep.stage_today_names(Namespace(labels=lab, events_dir=v4))
    got = {s: {r["id"]: r for r in _rows(lab / f"names_{s}_V0.jsonl")} for s in ("train", "eval", "steer")}
    assert got["train"]["t1"] == {"id": "t1", "need": "wall", "name": "white wall", "passed": True, "namer": "gemma"}
    assert (got["train"]["t2"]["name"], got["train"]["t2"]["namer"], got["train"]["t2"]["need"]) == ("grey sofa", "cosmos", "object")
    assert got["eval"]["e1"]["name"] == "tall dark cabinet"
    assert got["eval"]["e2"] == {"id": "e2", "need": "wall", "name": None, "passed": False, "namer": None}
    assert got["steer"]["s1"]["name"] == "glass door" and set(got["steer"]) == {"s1"}


def test_naming_sample_proportional_excludes(tmp_path):
    v4, lab = tmp_path / "v4", tmp_path / "labels"
    _write(v4 / "objects_train.jsonl", [_o(f"w{i}") for i in range(180)] + [_o(f"b{i}", "object") for i in range(20)])
    _write(lab / "names_train_gemma_d2.jsonl", [])
    ex = tmp_path / "earlier.jsonl"
    _write(ex, [_o(f"w{i}") for i in range(10)])

    def run(out):
        prep.stage_naming_sample(Namespace(events_dir=v4, labels=lab, out_dir=out, n=50, seed=1, flagged=None,
                                           proportional=True, exclude=ex))
        return _rows(out / "objects_sample.jsonl")
    rows = run(tmp_path / "a")
    ids = [r["object_id"] for r in rows]
    assert len(rows) == 50 and all(r["group"] == "random" for r in rows)
    assert not set(ids) & {f"w{i}" for i in range(10)}
    assert ids == [r["object_id"] for r in run(tmp_path / "b")]
    assert sum(r["kind_hint"] == "object" for r in rows) < 20          # strata would take all 20 objects


SBATCH = REPO / "slurm" / "naming_variants.sbatch"


def test_naming_variants_sbatch_choose_mode():
    text = SBATCH.read_text()
    for model in ("Qwen/Qwen3.6-35B-A3B-FP8", "nvidia/Cosmos-Reason2-32B", "google/gemma-3-12b-it"):
        assert model in text
    for s in ('MODE="${MODE:-name}"', 'CHOOSERS="${CHOOSERS:-C1 C2 C3}"', 'CHOOSE_PROMPT="${CHOOSE_PROMPT:-c1}"',
              'MAX_TOKENS="${MAX_TOKENS:-8192}"', 'CONCURRENCY="${CONCURRENCY:-32}"', 'OUT="${OUT:-outputs/instr_labels_v3}"',
              '--stage choose', '--max-tokens "$MAX_TOKENS"', "naming_ab/choose_${c}.jsonl", "naming_check/choose_${c}.jsonl",
              'names_${s}_${c}_${NAME_PROMPT}.jsonl'):
        assert s in text, s


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _dry_env(tmp_path):
    import os
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    (tmp_path / "www").mkdir()
    (tmp_path / "www" / "health").write_text("ok")
    vllm = tmp_path / "venv" / "bin" / "vllm"
    vllm.write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = --port ] && PORT=$2; shift; done\n'
                    f'cd {tmp_path / "www"} && exec /usr/bin/python3 -m http.server "$PORT" --bind 0.0.0.0\n')
    (tmp_path / "bin" / "nvidia-smi").write_text("#!/bin/bash\necho fake-nvidia-smi\n")
    uv = tmp_path / "fake_uv"
    uv.write_text(f'#!/bin/bash\necho "$*" >> {tmp_path / "calls.txt"}\n')
    for f in (vllm, tmp_path / "bin" / "nvidia-smi", uv):
        f.chmod(0o755)
    data, out = tmp_path / "data", tmp_path / "out"
    _write(data / "naming_ab" / "objects_sample.jsonl", [_o("a")])
    for v in ("V0", "V1", "V2", "V3"):
        _write(data / "naming_ab" / f"names_{v}.jsonl", [])
    _write(data / "objects_train.jsonl", [_o("t")])
    _write(out / "names_train_V0.jsonl", [])
    for v in ("V1", "V2", "V3"):
        _write(out / f"names_train_{v}_n2.jsonl", [])
    return {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "VENV": str(tmp_path / "venv"),
            "ENDPOINT_FILE": str(tmp_path / "endpoint.txt"), "PORT": str(_free_port()), "UV": str(uv),
            "DATA": str(data), "OUT": str(out), "MODE": "choose", "CHOOSERS": "C1", "SETS": "sample train"}


def test_naming_variants_sbatch_choose_dry_run(tmp_path):
    import subprocess
    env = _dry_env(tmp_path)
    r = subprocess.run(["bash", str(SBATCH)], cwd=REPO, env=env, capture_output=True, text=True, timeout=150)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    calls = (tmp_path / "calls.txt").read_text()
    assert "--stage choose" in calls and f"V0={tmp_path}/data/naming_ab/names_V0.jsonl" in calls
    assert f"--out {tmp_path}/out/names_train_C1_n2.jsonl" in calls and "--max-tokens 8192" in calls
    assert f"V2={tmp_path}/out/names_train_V2_n2.jsonl" in calls
    (tmp_path / "out" / "names_train_V2_n2.jsonl").unlink()            # a missing input: stop before any model starts
    r = subprocess.run(["bash", str(SBATCH)], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "names_train_V2_n2.jsonl" in r.stdout and "node=" not in r.stdout


_ASPEC = importlib.util.spec_from_file_location("audit_instruction_labels", REPO / "scripts" / "audit_instruction_labels.py")
audit = importlib.util.module_from_spec(_ASPEC)
_ASPEC.loader.exec_module(audit)


def _sample_dir(tmp_path):
    d = tmp_path / "check"
    _write(d / "objects_sample.jsonl", [{**_o("a"), "group": "random"}, {**_o("b", "object"), "group": "random"},
                                        {**_o("c", "object"), "group": "random"}])
    _write(d / "names_V0.jsonl", [{"id": o, "name": "old", "passed": True} for o in "abc"])
    _write(d / "choose_C1.jsonl", [
        {"id": "a", "name": "white wall", "passed": True, "choice": "b", "single": False, "candidates": ["x", "white wall"],
         "rejects": [], "name_source": ["V2"]},
        {"id": "b", "name": "old", "passed": True, "choice": "none", "single": False, "candidates": ["x", "y"],
         "rejects": [], "name_source": ["V0_fallback"]},
        {"id": "c", "name": "old", "passed": True, "choice": None, "single": False, "candidates": ["x", "y"],
         "rejects": ["parse"], "name_source": ["V0_fallback"]}])
    return d


def test_score_choosers(tmp_path):
    d = _sample_dir(tmp_path)
    marks = {"a": {"verdict": "done", "right": ["white wall"]}, "b": {"verdict": "none"},
             "c": {"verdict": "done", "right": ["chair"]}}
    s = audit.score_dir(d, marks, "choose_*.jsonl")
    assert s["random"]["C1"]["correct"] == 1 and s["random"]["C1"]["n"] == 3 and set(s["random"]) == {"C1"}
    assert s["choosers"]["C1"]["picked"] == 1 and s["choosers"]["C1"]["none"] == 1 and s["choosers"]["C1"]["failed"] == 1
    assert s["choosers"]["C1"]["from"] == {"V2": 1, "V0_fallback": 2}
    assert "choosers" not in audit.score_dir(d, marks, "names_V*.jsonl")        # name files carry no choice


def test_names_glob_picks_only_the_chooser_file(tmp_path, monkeypatch):     # Review Focus 5
    d = _sample_dir(tmp_path)
    vn = audit._variant_names(d, "choose_*.jsonl")
    assert set(vn) == {"C1"} and set(audit._variant_names(d)) == {"V0"}
    monkeypatch.setattr(audit, "_naming_jpeg", lambda row: b"jpg")
    items = audit.names_items(_rows(d / "objects_sample.jsonl"), vn, seed=0)
    assert [it["names"] for it in items] == [["white wall"], ["old"], ["old"]]


def test_page_sanitizes_doc_ids():
    html = audit.TEMPLATE.read_text()
    assert 'const docId = (id) => id.replace(/[^A-Za-z0-9_\\-.~:@+]/g, "_");' in html
    assert 'db.collection("marks").doc(docId(id)).set(' in html
    assert "marks[oid || d.id] = { verdict, note, right, oid };" in html
