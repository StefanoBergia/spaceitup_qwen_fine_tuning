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
