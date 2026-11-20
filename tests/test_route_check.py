"""Route checker (spec docs/superpowers/specs/2026-11-16-route-checker-design.md): prompt k1, reply parsing, the
per-route check call."""
import json

from rover_vlm.route_check import CHECK_PROMPT_VERSION, check_prompt, check_route, parse_check_reply

GOOD = {"names": [{"name": "grey sofa", "fits": "yes", "is_object": "yes"}], "missing_obstacle": "no",
        "stop_ok": "yes", "better_stop": "", "overall_true": "yes", "confidence": 80}


def test_prompt_states_legend_and_never_sides():
    text = "Go left of the sofa, then stop next to the lamp."
    p = check_prompt(text)
    assert text in p
    for word in ("magenta", "dashed", "green", "orange", "middle of the room"):
        assert word in p
    assert "already correct" in p
    assert CHECK_PROMPT_VERSION == "k1"


def test_parse_plain_json():
    r = parse_check_reply(json.dumps(GOOD))
    assert r == {"names": [{"number": None, "name": "grey sofa", "fits": "yes", "is_object": "yes"}],
                 "missing_obstacle": False, "stop_ok": True, "better_stop": "", "overall_true": True, "confidence": 80}


def test_prompt_asks_only_about_outlined_objects():
    """Review fix: places (walls, corners, doorways, hallways, rooms) must not be listed as named things, or
    object_no fires for every place stop and the score turns into a stop-kind proxy."""
    p = check_prompt("Turn right, then stop by the wall.")
    assert "outlined" in p and '"number"' in p
    assert "do not list" in p and all(w in p for w in ("wall", "floor", "doorway", "corner", "hallway", "room"))
    assert "calls it an object or furniture" in p


def test_parse_keeps_object_number():
    r = parse_check_reply(json.dumps({**GOOD, "names": [{"number": 2, "name": "lamp", "fits": "no",
                                                         "is_object": "yes"}]}))
    assert r["names"] == [{"number": 2, "name": "lamp", "fits": "no", "is_object": "yes"}]


def test_parse_after_think_and_code_fence():
    raw = "<think>the sofa is grey {not json}</think>```json\n" + json.dumps({**GOOD, "overall_true": False}) \
        + "\n```\nThat is my answer."
    r = parse_check_reply(raw)
    assert r is not None and r["overall_true"] is False and r["stop_ok"] is True


def test_parse_bad_returns_none():
    assert parse_check_reply("The route looks fine to me.") is None
    assert parse_check_reply(json.dumps({k: v for k, v in GOOD.items() if k != "overall_true"})) is None


def test_parse_empty_names_ok():
    r = parse_check_reply(json.dumps({**GOOD, "names": []}))
    assert r is not None and r["names"] == []


def test_confidence_forms():
    def conf(v):
        d = dict(GOOD)
        if v is None:
            d.pop("confidence")
        else:
            d["confidence"] = v
        return parse_check_reply(json.dumps(d))["confidence"]
    assert conf("85%") == 85
    assert conf(85.0) == 85
    assert conf(140) == 100
    assert conf(-3) == 0
    assert conf(None) == 50


def test_check_route_retries_then_parse_reject():
    item = {"id": "g3/00800-TEEsavR23oF_c000_L", "image": "/x.jpg", "text": "Stop by the wall."}
    seeds = []

    def prose(image, prompt, seed):
        seeds.append(seed)
        return "no json here", "stop"
    row = check_route(prose, item, 7)
    assert row["rejects"] == ["parse"] and row["passed"] is False and row["attempts"] == 2
    assert len(row["raw"]) == 2 and row["id"] == item["id"] and seeds == [7, 8]

    replies = iter(["prose first", json.dumps(GOOD)])
    seeds.clear()

    def second(image, prompt, seed):
        seeds.append(seed)
        return next(replies), "stop"
    row = check_route(second, item, 7)
    assert row["passed"] is True and row["attempts"] == 2 and seeds == [7, 8]
    assert row["overall_true"] is True and row["prompt_version"] == "k1" and row["rejects"] == []


# --- Task 2: manifests -------------------------------------------------------------------------------------------
import base64
import io

from PIL import Image

from rover_vlm.route_check import audit_page_items, calibration_rows, mark_wrong


def _jpeg_uri(colour=(200, 10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_audit_page_items_reads_payload():
    items = [{"id": "00800-TEEsavR23oF_c000_L", "instruction": "Stop by the wall.", "img": "data:x"}]
    html = "<html><script>\nconst DATA = " + json.dumps({"items": items, "mode": "d1"}) + ";\nrender();\n</script>"
    assert audit_page_items(html) == items


def test_mark_wrong_mapping():
    assert mark_wrong("ok") is False
    assert mark_wrong("unsure") is None and mark_wrong("?") is None and mark_wrong(None) is None
    for v in ("stop", "name", "keep", "neither", "unnatural", "object", "place"):
        assert mark_wrong(v) is True


def test_calibration_rows_skip_and_ids(tmp_path):
    items = [{"id": "00800-X_c000_L", "instruction": "Go left of the sofa, then stop by the wall.", "img": _jpeg_uri()},
             {"id": "00800-X_c001_R", "instruction": "Stop in the corner.", "img": _jpeg_uri()},
             {"id": "00800-X_c002_R", "instruction": "DROPPED: no place fits the end of this route. Is dropping it right?",
              "img": _jpeg_uri()},
             {"id": "00800-X_c003_R", "instruction": "Stop in the doorway.", "img": _jpeg_uri()}]
    marks = {"00800-X_c000_L": {"verdict": "stop", "note": "middle of the room"},
             "00800-X_c001_R": {"verdict": "unsure"}, "00800-X_c002_R": {"verdict": "keep"}}
    rows = calibration_rows("g3", items, marks, tmp_path)
    assert [r["id"] for r in rows] == ["g3/00800-X_c000_L"]
    r = rows[0]
    assert r["route_id"] == "00800-X_c000_L" and r["page"] == "g3" and r["wrong"] is True
    assert r["verdict"] == "stop" and r["note"] == "middle of the room" and r["text"].startswith("Go left")
    assert Image.open(r["image"]).size == (8, 8)


def test_same_route_two_pages_two_rows(tmp_path):
    item = {"id": "00800-X_c000_L", "instruction": "Stop by the wall.", "img": _jpeg_uri()}
    marks = {"00800-X_c000_L": {"verdict": "ok"}}
    ids = [r["id"] for s in ("a", "b") for r in calibration_rows(s, [item], marks, tmp_path)]
    assert ids == ["a/00800-X_c000_L", "b/00800-X_c000_L"]


# --- Task 3: --stage check summary ------------------------------------------------------------------------------
from rover_vlm.descriptions import summarize_d1


def test_summarize_check():
    rows = [{"id": "a", "passed": True, "overall_true": True, "rejects": []},
            {"id": "b", "passed": False, "rejects": ["parse"]},
            {"id": "c", "error": "APIConnectionError: x"}]
    s = summarize_d1(rows, "check")
    assert s["errors"] == 1 and s["parse_failed"] == 1 and s["overall_true_share"] == 1.0


# --- Task 4: score fit, AUC, removal curve ----------------------------------------------------------------------
import math

import numpy as np

from rover_vlm.route_check import FEATURES, auc, features, fit_logistic, group_folds, oof_scores, predict, removal_curve


def test_auc_known():
    assert auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75
    assert auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5


def test_fit_separates():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.3
    X = np.column_stack([y.astype(float), rng.random(200)])
    w = fit_logistic(X, y)
    assert auc(predict(w, X), y) > 0.95


def test_oof_groups_never_straddle():
    groups = [f"00800-X_c{i % 7:03d}" for i in range(70)]
    folds = group_folds(groups, 5, 0)
    by = {}
    for g, f in zip(groups, folds):
        by.setdefault(g, set()).add(f)
    assert all(len(s) == 1 for s in by.values()) and set(folds) <= set(range(5))
    rng = np.random.default_rng(1)
    y = rng.random(70) < 0.3
    X = np.column_stack([y.astype(float) + rng.normal(0, 0.1, 70), rng.random(70)])
    s = oof_scores(X, y, groups, k=5, seed=0)
    assert s.shape == (70,) and auc(s, y) > 0.9


def test_removal_curve_monotone():
    wrong = np.array([True] * 10 + [False] * 30)
    scores = np.linspace(1, 0, 40)            # perfectly ranked: the 10 wrong score highest
    cur = removal_curve(scores, wrong)
    acc = [c["kept_accuracy"] for c in cur]
    assert all(b >= a for a, b in zip(acc, acc[1:]))
    c10 = next(c for c in cur if c["share"] == 0.10)
    assert c10["removed"] == math.ceil(0.10 * 40) and c10["wrong_caught"] == 0.4


def test_features_none_on_parse_reject():
    assert features({"id": "a", "passed": False, "rejects": ["parse"]}) is None
    row = {"passed": True, "names": [{"name": "x", "fits": "no", "is_object": "yes"}], "missing_obstacle": True,
           "stop_ok": False, "better_stop": "", "overall_true": True, "confidence": 70}
    f = features(row)
    assert list(FEATURES) == ["name_no", "object_no", "missing_obstacle", "stop_no", "overall_no", "p_false"]
    assert f.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 0.3]


# --- Task 5: train drop list ------------------------------------------------------------------------------------
from rover_vlm.route_check import drop_records


def test_drop_records_train_only():
    kept, matched = drop_records([{"id": "a"}, {"id": "b"}, {"id": "c"}], {"b", "zz"})
    assert [r["id"] for r in kept] == ["a", "c"] and matched == 1


def test_removal_curve_matches_apply_with_tied_scores():
    """Review fix: each curve row must describe exactly what `score >= threshold` (what --apply does) removes."""
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    wrong = np.array([True, True, False, False, True, False, False, False, True, False])
    for c in removal_curve(scores, wrong, shares=(0.1, 0.2, 0.3)):
        gone = scores >= c["threshold"]
        assert c["removed"] == int(gone.sum())
        assert c["share_actual"] == round(gone.mean(), 4)
        assert c["kept_accuracy"] == round(float(1 - wrong[~gone].mean()), 4)
        assert c["wrong_caught"] == round(float(wrong[gone].sum() / wrong.sum()), 4)
    c20 = removal_curve(scores, wrong, shares=(0.2,))[0]
    assert c20["threshold"] == 0.5 and c20["removed"] == 5


from rover_vlm.route_check import kind_matches_text


def test_kind_matches_the_judged_text():
    """Review fix: d3_g3 was judged before Addendum A2; a row whose current stop kind is not what its judged text
    says must not be binned under that kind."""
    assert kind_matches_text("object:next to", "Go left of the sofa, then stop by the wall.") is False
    assert kind_matches_text("object:next to", "Go between the chair and the table, then stop next to the lamp.")
    assert kind_matches_text("place:wall", "Turn right, then stop by the wall.")
    assert kind_matches_text("object:behind", "Turn left behind the sofa, then stop in the corner.") is False
    assert kind_matches_text("place:corner", "Turn left behind the sofa, then stop in the corner.")
    assert kind_matches_text("place:room_middle", "Keep the sofa on your left and stop in the middle of the room.")
    assert kind_matches_text("wall:in front of", "Go ahead and stop in front of the white wall.")
    assert kind_matches_text("place:hallway_end", "Go ahead, then stop at the end of the hallway.")
    assert kind_matches_text("place:doorway", "Go to the doorway.")


# --- review fix: the calibration report's breakdowns -------------------------------------------------------------
import importlib.util
from argparse import Namespace
from pathlib import Path

_RSPEC = importlib.util.spec_from_file_location(
    "route_check_report", Path(__file__).resolve().parent.parent / "scripts" / "route_check_report.py")
report = importlib.util.module_from_spec(_RSPEC)
_RSPEC.loader.exec_module(report)


def test_calibrate_report_breakdowns(tmp_path, capsys):
    """Spec Validation: each answer against its own mark type, the p_false baseline next to every breakdown, the
    top / bottom lists, and d3 rows binned under a stop kind only when the judged text says it."""
    rng = np.random.default_rng(0)
    man, chk, ev = [], [], []
    for i in range(60):
        page = "d3_g3" if i % 2 else "d2_g3"
        rid = f"{800 + i % 30:05d}-AAAAAAAAAAA_c{i:03d}_L"
        verdict = "ok" if i % 3 else ("name" if i % 2 else "stop")
        wrong = verdict != "ok"
        text = "Go left of the sofa, then stop by the wall." if i % 5 == 0 else "Go left, then stop next to the lamp."
        man.append({"id": f"{page}/{rid}", "route_id": rid, "page": page, "image": "/x.jpg", "text": text,
                    "wrong": wrong, "verdict": verdict, "note": ""})
        chk.append({"id": f"{page}/{rid}", "passed": True, "rejects": [],
                    "names": [{"number": 1, "name": "lamp", "fits": "no" if verdict == "name" else "yes",
                               "is_object": "yes"}],
                    "missing_obstacle": bool(rng.random() < 0.1), "stop_ok": verdict != "stop", "better_stop": "",
                    "overall_true": not wrong, "confidence": int(rng.integers(50, 95))})
        ev.append({"id": rid, "reason": None, "stop": {"type": "object", "relation": "next to", "object": "f:b1",
                                                      "place": None, "amount": None, "distance_m": None}})
    for name, rows in (("manifest.jsonl", man), ("check.jsonl", chk)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    evd = tmp_path / "v5"
    evd.mkdir()
    (evd / "events_train.jsonl").write_text("")
    (evd / "events_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ev))
    report.calibrate(Namespace(manifest=tmp_path / "manifest.jsonl", check=tmp_path / "check.jsonl", events_dir=evd,
                               out=tmp_path / "out"))
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["feature_by_verdict"]["name"]["name_no"] == 1.0
    assert rep["feature_by_verdict"]["stop"]["stop_no"] == 1.0
    assert all("auc_baseline" in v for v in rep["per_page"].values())
    assert all("auc_baseline" in v for v in rep["per_verdict"].values())
    assert all("auc_baseline" in v for v in rep["per_stop_kind_d3"].values())
    d3_by_wall = sum(1 for r in man if r["page"] == "d3_g3" and "by the wall" in r["text"])
    assert rep["kinds_excluded_text_mismatch"] == d3_by_wall > 0
    assert rep["per_stop_kind_d3"]["object:next to"]["n"] == 30 - d3_by_wall
    assert all("share_actual" in c for c in rep["curve"])
    out = capsys.readouterr().out
    assert "highest-scored" in out and "lowest-scored wrong" in out
