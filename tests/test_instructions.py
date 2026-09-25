import json

from conftest import write_sample
from rover_vlm.instructions import (
    SHARED_ENTRY_TOL, TEMPLATE_INSTRUCTIONS, build_pair, side_candidates,
)


def _path(rec):
    return json.loads(rec["conversations"][1]["value"])["path"]


def test_side_candidates_prefers_label_and_needs_both_sides():
    meta = {"label": 2, "accepted": [0, 2, 1],
            "candidates": [{"label": "left", "feasible": True},
                           {"label": "right", "feasible": True},
                           {"label": "left", "feasible": True}]}
    assert side_candidates(meta) == {"left": 2, "right": 1}
    meta["accepted"] = [0, 2]
    assert side_candidates(meta) is None


def test_build_pair_two_records_same_prompt_but_instruction(two_sided_sample):
    left, right = build_pair(two_sided_sample)
    assert (left["id"], right["id"]) == ("scene_a_c000_L", "scene_a_c000_R")
    pl, pr = left["conversations"][0]["value"], right["conversations"][0]["value"]
    assert pl.replace(TEMPLATE_INSTRUCTIONS["left"], "X") == pr.replace(TEMPLATE_INSTRUCTIONS["right"], "X")
    assert left["habitat_meta"]["alt_path"] == _path(right)
    assert right["habitat_meta"]["alt_path"] == _path(left)
    assert _path(left)[0] == _path(right)[0]
    assert {left["habitat_meta"]["side"], right["habitat_meta"]["side"]} == {"left", "right"}
    assert left["habitat_meta"]["pair_id"] == "scene_a_c000"
    assert left["habitat_meta"]["instruction_source"] == "template"


def test_build_pair_snaps_small_entry_gap(tmp_path):
    d = write_sample(tmp_path / "s_c001",
                     left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(266, 512), (370, 400), (256, 300)])   # 10 px = 0.0195 apart
    left, right = build_pair(d)
    assert _path(left)[0] == _path(right)[0]
    assert 0 < left["habitat_meta"]["entry_gap"] <= SHARED_ENTRY_TOL
    assert right["habitat_meta"]["start_uv_norm"] == _path(left)[0][:2]


def test_build_pair_drops_distinct_entries(tmp_path):
    d = write_sample(tmp_path / "s_c002",
                     left_uv=[(0, 400), (140, 350), (256, 300)],
                     right_uv=[(512, 400), (370, 350), (256, 300)])
    reasons = {}
    assert build_pair(d, reasons=reasons) is None
    assert reasons == {"entry_differs": 1}


def test_build_pair_one_side_failing_drops_both(tmp_path):
    d = write_sample(tmp_path / "s_c003",
                     left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(900, 900), (950, 950)])            # clipped away entirely
    reasons = {}
    assert build_pair(d, reasons=reasons) is None
    assert reasons == {"record_failed": 1}


def test_build_pair_uses_given_texts(two_sided_sample):
    left, right = build_pair(two_sided_sample,
                             texts={"left": "Pass left of the lamp.", "right": "Pass right of the lamp."})
    assert '"Pass right of the lamp."' in right["conversations"][0]["value"]
    assert right["habitat_meta"]["instruction_source"] == "given"


from PIL import Image

from rover_vlm.instructions import (
    check_numbering, check_prompt, generation_prompt, instruction_rejects, is_side_only,
    parse_choice, parse_instruction, render_check_image, render_generation_image,
)


def test_check_numbering_is_deterministic_and_mixed():
    a = [check_numbering(f"s_c{i:03d}_L")["target"] for i in range(40)]
    assert a == [check_numbering(f"s_c{i:03d}_L")["target"] for i in range(40)]
    assert set(a) == {1, 2}


def test_render_images(two_sided_sample, tmp_path):
    left, _ = build_pair(two_sided_sample)
    render_generation_image(left, tmp_path / "g.jpg")
    expected = render_check_image(left, tmp_path / "c.jpg")
    assert expected == check_numbering(left["id"])["target"]
    for name in ("g.jpg", "c.jpg"):
        assert Image.open(tmp_path / name).size == (768, 768)


def test_prompts_have_no_concrete_nouns():
    text = (generation_prompt("left") + generation_prompt("right") + check_prompt("{x}")).lower()
    for noun in ("sofa", "table", "chair", "counter", "bed", "desk", "cabinet", "couch", "kitchen"):
        assert noun not in text


def test_parse_instruction_and_choice():
    raw = '<think>hm</think>\n<answer>\n{"instruction": "Pass left of the grey sofa."}\n</answer>'
    assert parse_instruction(raw) == "Pass left of the grey sofa."
    assert parse_instruction("<think>x</think><answer>Pass right of the bed.</answer>") == "Pass right of the bed."
    assert parse_instruction("<think>never closed") is None
    assert parse_choice('<answer>{"choice": 2}</answer>') == 2
    assert parse_choice("<answer>Route 1.</answer>") == 1
    assert parse_choice('<answer>{"choice": 3}</answer>') is None


def test_instruction_rejects():
    assert instruction_rejects("Pass to the left of the blue armchair, then through the door.") == []
    assert any(r.startswith("drawing") for r in instruction_rejects("Follow the magenta line past the chair on your left."))
    assert any(r.startswith("drawing") for r in instruction_rejects("Take route 2 along the wall next to the doorway."))
    assert "digits" in instruction_rejects("Drive to the point at 0.5 near the door on the left.")
    assert instruction_rejects("Go left.") == ["length:2"]
    assert instruction_rejects(None) == ["empty"]


def test_is_side_only():
    assert is_side_only("Go around the obstacle to the left.")
    assert not is_side_only("Go around the wooden table to the left.")


from rover_vlm.instructions import finalize_pairs, gate_failures, summarize_labels, with_instruction


def _row(rid, pair, passed, rejects=(), correct=True, instruction="Pass left of the lamp by the door."):
    return {"id": rid, "pair_id": pair, "instruction": instruction, "rejects": list(rejects),
            "side_only": False, "check_correct": correct, "passed": passed}


def test_summarize_and_gate():
    rows = [_row("a_L", "a", True), _row("a_R", "a", True),
            _row("b_L", "b", True), _row("b_R", "b", False, correct=False),
            _row("c_L", "c", False, rejects=["drawing:line"]), _row("c_R", "c", True),
            {"id": "d_L", "pair_id": "d", "error": "Timeout"}]
    s = summarize_labels(rows)
    assert (s["pairs_total"], s["pairs_kept"]) == (4, 1)
    assert s["n_drawing_refs"] == 1 and s["n_rejected"] == 1
    assert gate_failures(s, min_pair_survival=0.8, max_drawing_refs=0) == [
        "pair survival 0.250 < 0.8", "drawing references 1 > 0"]


def test_finalize_pairs_keeps_only_fully_passed(two_sided_sample):
    left, right = build_pair(two_sided_sample)
    rows = {left["id"]: _row(left["id"], "p", True, instruction="Pass left of the lamp."),
            right["id"]: _row(right["id"], "p", True, instruction="Pass right of the lamp.")}
    tmpl, cosmos, stats = finalize_pairs([left, right], rows)
    assert [r["id"] for r in tmpl] == [r["id"] for r in cosmos] == [left["id"], right["id"]]
    assert '"Pass right of the lamp."' in cosmos[1]["conversations"][0]["value"]
    assert cosmos[1]["habitat_meta"]["instruction_source"] == "cosmos"
    assert tmpl[1]["habitat_meta"]["instruction"] == TEMPLATE_INSTRUCTIONS["right"]
    assert stats == {"pairs_in": 1, "pairs_kept": 1}
    del rows[right["id"]]                                   # missing label row -> pair dropped
    assert finalize_pairs([left, right], rows)[2] == {"pairs_in": 1, "pairs_kept": 0}


def test_with_instruction_does_not_mutate(two_sided_sample):
    left, _ = build_pair(two_sided_sample)
    before = json.dumps(left)
    with_instruction(left, "Keep right of the plant.", "cosmos")
    assert json.dumps(left) == before


import pytest

from rover_vlm.instructions import assert_scene_disjoint


def test_assert_scene_disjoint(two_sided_sample):
    left, right = build_pair(two_sided_sample)
    assert_scene_disjoint([left], [])
    with pytest.raises(AssertionError):
        assert_scene_disjoint([left], [right])


import importlib.util
from pathlib import Path as _P


def _load_report():
    path = _P(__file__).resolve().parent.parent / "scripts" / "visualize_instruction_results.py"
    spec = importlib.util.spec_from_file_location("vir", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_cells(tmp_path):
    d = tmp_path / "cosmos" / "instr_cosmos"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"parse_rate": 1.0, "instruction": {"pair_success": 0.7}}))
    cells = _load_report().collect_cells(tmp_path)
    assert cells[("instr_cosmos", "cosmos")]["instruction"]["pair_success"] == 0.7


def test_build_pair_drops_near_identical_routes(tmp_path):
    d = write_sample(tmp_path / "s_c010",
                     left_uv=[(256, 512), (250, 400), (256, 300)],
                     right_uv=[(256, 512), (258, 400), (256, 300)])   # ~0.016 apart
    reasons = {}
    assert build_pair(d, reasons=reasons) is None
    assert reasons == {"routes_too_similar": 1}


def test_build_pair_snaps_start_visibility_flag(tmp_path):
    import json as _json
    d = write_sample(tmp_path / "s_c011",
                     left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)])
    fpv = _json.loads((d / "fpv_paths.json").read_text())
    # right route starts hidden at the same xy: the prompt's "visible/hidden" would leak the side
    fpv["candidates"][1]["runs"] = [{"hidden": True, "uv": [[256, 512], [300, 450]]},
                                    {"hidden": False, "uv": [[370, 400], [256, 300]]}]
    (d / "fpv_paths.json").write_text(_json.dumps(fpv))
    left, right = build_pair(d)
    assert _path(left)[0] == _path(right)[0]
    pl, pr = left["conversations"][0]["value"], right["conversations"][0]["value"]
    assert pl.replace(TEMPLATE_INSTRUCTIONS["left"], "X") == pr.replace(TEMPLATE_INSTRUCTIONS["right"], "X")


def test_build_pair_given_texts_with_quotes(two_sided_sample):
    left, right = build_pair(two_sided_sample,
                             texts={"left": 'Pass the "big" lamp on its left side.', "right": "Keep right of the lamp stand."})
    assert "'big'" in left["conversations"][0]["value"]


from rover_vlm.instructions import rescore_row


def test_parse_choice_accepts_string_numbers():
    # observed on the first gate run: Cosmos quotes the number
    assert parse_choice('<think>x</think>\n<answer>\n{"choice": "1"}\n</answer>') == 1
    assert parse_choice('<answer>{"choice": " 2 "}</answer>') == 2
    assert parse_choice('<answer>{"choice": "one"}</answer>') is None


def test_rescore_row_reparses_saved_replies():
    row = {"id": "a_L", "instruction": "Pass left of the grey sofa near the window.", "rejects": [],
           "check_choice": None, "check_expected": 1, "check_correct": None, "passed": False,
           "raw_check": '<answer>{"choice": "1"}</answer>'}
    new = rescore_row(row)
    assert (new["check_choice"], new["check_correct"], new["passed"]) == (1, True, True)
    assert row["passed"] is False                        # input untouched
    kept = {"id": "b_L", "passed": True, "check_choice": 2}   # passed rows keep no raw reply
    assert rescore_row(kept) == kept
    assert rescore_row({"id": "c", "error": "Timeout"}) == {"id": "c", "error": "Timeout"}



def test_generation_prompt_hands_over_the_side():
    p = generation_prompt("left")
    assert "rover's left" in p and "'left'" in p and "never the word 'right'" in p
    assert "rover's right" in generation_prompt("right")


def test_instruction_rejects_side_rules():
    ok = "Pass on the left of the grey sofa, then continue to the doorway."
    assert instruction_rejects(ok, side="left") == []
    assert "side:missing" in instruction_rejects(ok, side="right")
    both = "Pass left of the sofa, keeping the table on your right side."
    assert instruction_rejects(both, side="left") == ["side:opposite"]
    assert "no_landmark" in instruction_rejects("Go around the obstacle to the left side.", side="left")
    assert instruction_rejects(ok) == []                  # no side given -> i1 rules only


def test_rescore_row_reapplies_side_rules_for_i2():
    row = {"id": "a_L", "side": "left", "prompt_version": "i2", "rejects": [], "passed": True,
           "instruction": "Keep right of the tall bookshelf and go through the doorway."}
    new = rescore_row(row)
    assert new["passed"] is False and "side:missing" in new["rejects"]


from rover_vlm.instructions import render_generation_jpeg


def test_render_generation_jpeg_bytes_and_no_alt(two_sided_sample):
    left, _ = build_pair(two_sided_sample)
    data = render_generation_jpeg(left)
    assert data[:2] == b"\xff\xd8"                       # JPEG magic
    lone = json.loads(json.dumps(left))
    lone["habitat_meta"]["alt_path"] = None
    assert render_generation_jpeg(lone)[:2] == b"\xff\xd8"


def test_four_word_instructions_are_accepted():
    assert instruction_rejects("Go left around the sofa.", side="left") == []
    assert instruction_rejects("Go left sofa.", side="left") == ["length:3"]


from rover_vlm.habitat_data import build_record
from rover_vlm.instructions import build_frame, build_wrong_side, other_side_candidate, route_side


def _src(d):
    """What data/prepared_habitat_v4 holds for this sample."""
    return build_record(d, framing="endpoints", meta_extra={"sample_id": d.name})


def test_build_frame_pair_frame(two_sided_sample):
    f = build_frame(_src(two_sided_sample))
    assert f["plain"] == _src(two_sided_sample)
    kinds = sorted((r["habitat_meta"]["kind"], r["habitat_meta"]["side"]) for r in f["labelset"])
    assert kinds == [("instructed", "left"), ("opposite", "right")]
    assert all(r["habitat_meta"]["pair_id"] == "scene_a_c000" for r in f["labelset"])


def test_build_frame_single_route(tmp_path):
    # right route blocked: single-route frame; its alt_path is the blocked right route
    d = write_sample(tmp_path / "s_c020", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))
    f = build_frame(_src(d))
    (rec,) = f["labelset"]
    hm = rec["habitat_meta"]
    assert (rec["id"], hm["kind"], hm["side"]) == ("s_c020_L", "instructed", "left")
    assert "pair_id" not in hm and hm["alt_path"] and max(x for x, _, _ in hm["alt_path"]) > 0.6


def test_build_frame_direct_is_plain_only(tmp_path):
    d = write_sample(tmp_path / "s_c021", left_uv=[(256, 512), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], labels=("direct", "right"))
    f = build_frame(_src(d))
    assert f["labelset"] == [] and f["plain"]["id"] == "s_c021"


def test_other_side_candidate_prefers_feasible_opposite():
    meta = {"label": 0, "candidates": [{"label": "left", "feasible": True},
                                       {"label": "right", "feasible": False},
                                       {"label": "right", "feasible": True}]}
    assert other_side_candidate(meta, "left") == 2
    meta["candidates"][2]["feasible"] = False
    assert other_side_candidate(meta, "left") == 1
    assert other_side_candidate({"label": 0, "candidates": [{"label": "left", "feasible": True}]}, "left") is None
    assert route_side(meta) == "left"


def test_single_route_without_alternative_builds_with_none(tmp_path):
    d = write_sample(tmp_path / "s_c022", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(900, 900), (950, 950)], accepted=(0,))    # other candidate off-frame
    (rec,) = build_frame(_src(d))["labelset"]
    assert rec["habitat_meta"]["alt_path"] is None


def test_build_wrong_side(tmp_path):
    # single-route frame (right blocked): instructed to go right, where no route exists;
    # the answer stays the feasible left route, and nothing needs Cosmos
    d = write_sample(tmp_path / "s_c023", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))
    w = build_wrong_side(_src(d))
    hm = w["habitat_meta"]
    assert (w["id"], hm["kind"], hm["side"], hm["feasible_side"]) == ("s_c023_W", "wrong_side", "right", "left")
    assert hm["alt_path"] is None and '"Go around to the right."' in w["conversations"][0]["value"]
    assert min(x for x, _, _ in json.loads(w["conversations"][1]["value"])["path"]) < 0.4
    assert build_wrong_side(_src(write_sample(tmp_path / "s_c024",
        left_uv=[(256, 512), (140, 400), (256, 300)], right_uv=[(256, 512), (370, 400), (256, 300)]))) is None


from rover_vlm.instructions import finalize_general, retry_ids


def test_summarize_accept_rate_and_single_rows():
    rows = [{"id": "a_L", "pair_id": None, "passed": True, "rejects": []},
            {"id": "b_R", "pair_id": None, "passed": False, "rejects": ["length:3"]}]
    s = summarize_labels(rows)
    assert s["accept_rate"] == 0.5 and s["pairs_total"] == 0
    assert gate_failures(s, min_pair_survival=0.0, max_drawing_refs=0, min_accept=0.9) == ["accept rate 0.500 < 0.9"]


def test_retry_ids_skips_errors_passed_and_retried():
    rows = [{"id": "a", "passed": False}, {"id": "b", "passed": False, "retried": True},
            {"id": "c", "passed": True}, {"id": "d", "error": "Timeout"}]
    assert retry_ids(rows) == {"a"}


def test_finalize_general(two_sided_sample, tmp_path):
    f = build_frame(_src(two_sided_sample))
    single = build_frame(_src(write_sample(tmp_path / "s_c030", left_uv=[(256, 512), (140, 400), (256, 300)],
                                           right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))))
    labels = f["labelset"] + single["labelset"]
    ok = lambda r: {"id": r["id"], "passed": True, "instruction": f"Pass {r['habitat_meta']['side']} of the lamp."}
    rows = {r["id"]: ok(r) for r in labels}
    out = finalize_general([f["plain"], single["plain"]], labels, rows)
    assert len(out["train_or_all"]) == 2 + 3 and len(out["pairs"]) == 2 and len(out["instructed"]) == 2
    rows[f["labelset"][1]["id"]]["passed"] = False               # one side of the pair rejected
    out = finalize_general([f["plain"], single["plain"]], labels, rows)
    assert out["pairs"] == [] and len(out["train_or_all"]) == 2 + 1
    assert [r["id"] for r in out["instructed"]] == ["s_c030_L"]
    assert out["stats"]["pairs_dropped"] == 1


def _load_general_report():
    path = _P(__file__).resolve().parent.parent / "scripts" / "visualize_instruction_general.py"
    spec = importlib.util.spec_from_file_location("vig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cells_table(tmp_path):
    d = tmp_path / "eval_pairs" / "general"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"parse_rate": 1.0, "mean_point_error_median": 0.02,
                                                "win_vs_to_goal": 0.7, "instruction": {"pair_success": 0.8, "follow_rate": 0.9}}))
    (row,) = _load_general_report().cells_table(tmp_path)
    assert (row["tag"], row["set"], row["pair_success"]) == ("general", "eval_pairs", 0.8)


def test_single_route_alt_only_from_opposite_side(tmp_path):
    # review I1: the only other candidates are same-side / direct -> no grey route at all
    d = write_sample(tmp_path / "s_c040", left_uv=[(256, 512), (140, 400), (256, 300)],
                     right_uv=[(256, 512), (170, 420), (256, 300)], labels=("left", "left"), accepted=(0,),
                     extra=[([(256, 512), (256, 300)], "direct", False)])
    (rec,) = build_frame(_src(d))["labelset"]
    assert rec["habitat_meta"]["alt_path"] is None
    assert other_side_candidate({"label": 0, "candidates": [{"label": "left"}, {"label": "left"},
                                                            {"label": "direct"}]}, "left") is None


def test_generation_prompt_without_alternative():
    p = generation_prompt("left", has_alt=False)
    assert "grey route" not in p and "rover's left" in p and "never the word 'right'" in p
    assert "grey route" in generation_prompt("left")


def test_wrong_side_probe_requires_gt_to_bend_feasible(tmp_path):
    # review I2: a ground truth that runs straight cannot calibrate "kept feasible"
    d = write_sample(tmp_path / "s_c041", left_uv=[(256, 512), (253, 400), (256, 300)],
                     right_uv=[(256, 512), (370, 400), (256, 300)], accepted=(0,))
    assert build_wrong_side(_src(d)) is None


def test_finalize_general_reports_missing_labels(two_sided_sample):
    # review I3: an unlabelled record is not a rejection
    f = build_frame(_src(two_sided_sample))
    out = finalize_general([f["plain"]], f["labelset"], {})
    assert out["stats"]["missing"] == 2 and out["stats"]["rejected"] == 0
