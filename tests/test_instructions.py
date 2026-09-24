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
    text = (generation_prompt() + check_prompt("{x}")).lower()
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
            {"id": "d_L", "error": "Timeout"}]
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

