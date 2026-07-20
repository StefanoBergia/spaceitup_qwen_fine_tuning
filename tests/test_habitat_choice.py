import json

import pytest

from rover_vlm.eval import aggregate_choice_metrics, choice_metrics, parse_choice_answer
from rover_vlm.habitat_choice import (
    DATASET_ROOT,
    badge_positions,
    build_choice_record,
    candidate_polylines,
    drawable_candidates,
    format_choice_answer,
)


def _fpv(cands, size=(512, 512)):
    return {"image_size": list(size), "goal": {"uv": [256.0, 300.0], "hidden": False},
            "candidates": cands}


def _cand(uv, badge=(256.0, 256.0), hidden=False):
    return {"runs": [{"hidden": hidden, "uv": uv}], "badge_uv": list(badge)}


# --- geometry -----------------------------------------------------------------------


def test_candidate_polylines_normalizes_and_clips():
    # second point sits well outside the frame and must be clipped to the boundary
    fpv = _fpv([_cand([[256.0, 256.0], [256.0, 1024.0]])])
    polys = candidate_polylines(fpv)
    assert len(polys) == 1
    xs = [p[0] for p in polys[0]]
    ys = [p[1] for p in polys[0]]
    assert xs == [0.5, 0.5]
    assert ys[0] == 0.5 and ys[-1] == pytest.approx(1.0)


def test_candidate_entirely_outside_frame_is_empty():
    fpv = _fpv([_cand([[2000.0, 2000.0], [3000.0, 3000.0]])])
    assert candidate_polylines(fpv) == [[]]
    assert drawable_candidates(fpv) == []


def test_drawable_candidates_reports_indices():
    fpv = _fpv([
        _cand([[100.0, 100.0], [200.0, 200.0]]),
        _cand([[9000.0, 9000.0], [9100.0, 9100.0]]),
        _cand([[300.0, 100.0], [300.0, 400.0]]),
    ])
    assert drawable_candidates(fpv) == [0, 2]


# --- badge placement ----------------------------------------------------------------


def test_badge_falls_back_on_path_when_badge_uv_is_off_frame():
    # badge_uv is far outside the image; the badge must still land inside the frame
    fpv = _fpv([_cand([[256.0, 100.0], [256.0, 400.0]], badge=(256.0, 5000.0))])
    (pos,) = badge_positions(fpv, candidate_polylines(fpv))
    assert pos is not None
    assert 0.0 < pos[0] < 1.0 and 0.0 < pos[1] < 1.0


def test_badges_do_not_collide_when_badge_uv_coincides():
    # both candidates declare the SAME badge_uv — placement must separate them
    fpv = _fpv([
        _cand([[100.0, 100.0], [100.0, 400.0]], badge=(256.0, 256.0)),
        _cand([[400.0, 100.0], [400.0, 400.0]], badge=(256.0, 256.0)),
    ])
    a, b = badge_positions(fpv, candidate_polylines(fpv))
    dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    assert dist >= 0.07, f"badges overlap ({dist:.3f})"


def test_undrawable_candidate_gets_no_badge():
    fpv = _fpv([
        _cand([[100.0, 100.0], [100.0, 400.0]]),
        _cand([[9000.0, 9000.0], [9100.0, 9100.0]]),
    ])
    positions = badge_positions(fpv, candidate_polylines(fpv))
    assert positions[0] is not None and positions[1] is None


# --- record building ----------------------------------------------------------------


def _meta(label=1, accepted=(0, 1), kinds=("direct", "left", "right")):
    return {
        "label": label,
        "accepted": list(accepted),
        "margin": 0.1,
        "near_symmetric": False,
        "candidates": [{"display_index": i, "label": k} for i, k in enumerate(kinds)],
    }


def test_build_choice_record_shape(tmp_path):
    rec = build_choice_record(tmp_path / "scene_c001", tmp_path / "img.jpg", meta=_meta())
    assert rec["id"] == "scene_c001"
    assert json.loads(rec["conversations"][1]["value"]) == {"choice": 1}
    assert rec["choice_meta"]["accepted"] == [0, 1]
    assert rec["choice_meta"]["n_candidates"] == 3
    assert rec["choice_meta"]["kinds"][0] == "direct"


def test_build_choice_record_rejects_misaligned_display_index(tmp_path):
    meta = _meta()
    meta["candidates"][1]["display_index"] = 7  # badge number would no longer be the index
    assert build_choice_record(tmp_path / "s", tmp_path / "i.jpg", meta=meta) is None


def test_build_choice_record_rejects_label_outside_accepted(tmp_path):
    meta = _meta(label=2, accepted=(0, 1))
    assert build_choice_record(tmp_path / "s", tmp_path / "i.jpg", meta=meta) is None


def test_build_choice_record_rejects_out_of_range_label(tmp_path):
    assert build_choice_record(tmp_path / "s", tmp_path / "i.jpg", meta=_meta(label=9)) is None
    assert build_choice_record(tmp_path / "s", tmp_path / "i.jpg", meta=_meta(label=-1)) is None


def test_format_choice_answer_round_trip():
    assert json.loads(format_choice_answer(4)) == {"choice": 4}


# --- parsing ------------------------------------------------------------------------


def test_parse_clean_choice():
    assert parse_choice_answer('{"choice": 2}') == 2


def test_parse_choice_with_surrounding_prose():
    assert parse_choice_answer('Path 4 runs into a wall, so {"choice": 1} is best.') == 1


def test_parse_choice_prefers_object_over_stray_numbers():
    # numbers appear earlier in the reasoning; the object must win
    text = 'Candidates 0, 3 and 4 look plausible. Final answer: {"choice": 3}'
    assert parse_choice_answer(text) == 3


def test_parse_choice_bare_integer_fallback():
    assert parse_choice_answer("2") == 2


def test_parse_choice_garbage_returns_none():
    assert parse_choice_answer("I cannot tell which path to take.") is None


# --- metrics ------------------------------------------------------------------------


CM = {"label": 4, "accepted": [2, 4], "n_candidates": 5,
      "kinds": ["left", "direct", "right", "left", "right"], "near_symmetric": False}


def test_choice_metrics_exact_label():
    m = choice_metrics(4, CM)
    assert m == {"valid": 1, "strict_correct": 1, "accepted_correct": 1, "picked_direct": 0}


def test_choice_metrics_accepted_but_not_canonical():
    # the crux of the multi-acceptable case: strict says wrong, accepted says right
    m = choice_metrics(2, CM)
    assert m["strict_correct"] == 0
    assert m["accepted_correct"] == 1


def test_choice_metrics_wrong_and_direct():
    m = choice_metrics(1, CM)
    assert m["strict_correct"] == 0 and m["accepted_correct"] == 0
    assert m["picked_direct"] == 1


def test_choice_metrics_out_of_range_is_invalid():
    m = choice_metrics(9, CM)
    assert m["valid"] == 0 and m["accepted_correct"] == 0


def test_choice_metrics_unparseable_is_invalid():
    assert choice_metrics(None, CM)["valid"] == 0


def test_aggregate_counts_unparseable_as_wrong():
    recs = [
        {"parsed": 4, "gt": CM, "metrics": choice_metrics(4, CM)},
        {"parsed": None, "gt": CM, "metrics": choice_metrics(None, CM)},
    ]
    agg = aggregate_choice_metrics(recs)
    assert agg["parse_rate"] == 0.5
    # accuracy is over ALL samples, not just parsed ones
    assert agg["strict_accuracy"] == 0.5
    assert agg["accepted_accuracy"] == 0.5
    assert agg["valid_choice_rate"] == 0.5


def test_aggregate_chance_baselines_account_for_direct():
    recs = [{"parsed": 4, "gt": CM, "metrics": choice_metrics(4, CM)}]
    agg = aggregate_choice_metrics(recs)
    assert agg["chance_strict"] == pytest.approx(1 / 5)
    assert agg["chance_accepted"] == pytest.approx(2 / 5)
    # dropping the single 'direct' candidate leaves 4 options, both accepted ones kept
    assert agg["chance_strict_excluding_direct"] == pytest.approx(1 / 4)
    assert agg["chance_accepted_excluding_direct"] == pytest.approx(2 / 4)


def test_aggregate_breakdowns_present():
    recs = [{"parsed": 2, "gt": CM, "metrics": choice_metrics(2, CM)}]
    agg = aggregate_choice_metrics(recs)
    assert agg["accepted_accuracy_by_n_candidates"] == {"5": 1.0}
    assert agg["accepted_accuracy_by_near_symmetric"] == {"False": 1.0}


# --- live dataset -------------------------------------------------------------------


def test_live_sample_indices_align():
    if not DATASET_ROOT.exists():
        pytest.skip("dataset not mounted")
    sample = next(DATASET_ROOT.glob("*/samples/*/"))
    meta = json.loads((sample / "meta.json").read_text())
    fpv = json.loads((sample / "fpv_paths.json").read_text())
    assert len(fpv["candidates"]) == len(meta["candidates"])
    rec = build_choice_record(sample, sample / "x.jpg", meta=meta)
    if rec is None:
        pytest.skip("first sample filtered out; covered by unit tests")
    cm = rec["choice_meta"]
    assert cm["label"] in cm["accepted"]
    assert 0 <= cm["label"] < cm["n_candidates"]
