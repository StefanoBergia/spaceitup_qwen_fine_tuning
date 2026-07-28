"""Reasoning-trace quality metrics (rover_vlm.trace_eval): ROUGE-L distillation fidelity and
the occlusion-grounding heuristic."""

from rover_vlm.trace_eval import (
    aggregate_trace_quality,
    claims_occlusion,
    goal_occluded,
    rouge_l,
)


def test_rouge_l_identical_partial_disjoint():
    assert rouge_l("the floor is clear", "the floor is clear")["f1"] == 1.0
    assert rouge_l("alpha beta gamma", "delta epsilon zeta")["f1"] == 0.0
    # subsequence match: "the goal is hidden" shares "the goal ... hidden" with the ref
    mid = rouge_l("the goal is hidden", "the goal is clearly hidden")["f1"]
    assert 0.0 < mid < 1.0
    # empty guards
    assert rouge_l("", "x")["f1"] == 0.0 and rouge_l("x", "")["f1"] == 0.0


def test_claims_occlusion_net_stance():
    assert claims_occlusion("the goal is hidden behind the cabinet") is True
    assert claims_occlusion("the floor is unobstructed and the goal is in view") is False
    # net: one clear mention but two occlusion mentions -> claims occlusion
    assert claims_occlusion("the floor is unobstructed but the goal is hidden, out of sight") is True
    # a bare clear-view statement is not an occlusion claim
    assert claims_occlusion("clear line of sight to the goal") is False
    assert claims_occlusion("") is False


def test_goal_occluded_reads_visibility_flag():
    assert goal_occluded({"goal": [0.5, 0.6, 0]}) is True     # 0 = obstructed
    assert goal_occluded({"goal": [0.5, 0.6, 1]}) is False    # 1 = visible
    assert goal_occluded(None) is False


def test_aggregate_reports_rouge_and_balanced_grounding():
    # 3 occluded frames (2 called right), 1 visible frame (called right)
    recs = [
        {"rouge_f1": 0.4, "occ_claim": True, "gt_occ": True},
        {"rouge_f1": 0.2, "occ_claim": True, "gt_occ": True},
        {"rouge_f1": 0.6, "occ_claim": False, "gt_occ": True},   # missed occlusion
        {"rouge_f1": 0.8, "occ_claim": False, "gt_occ": False},  # correct visible
    ]
    a = aggregate_trace_quality(recs)
    assert a["n"] == 4
    assert abs(a["rouge_l_mean"] - 0.5) < 1e-9
    assert a["occ_recall_occluded"] == 2 / 3 and a["occ_recall_visible"] == 1.0
    assert abs(a["occ_balanced"] - (2 / 3 + 1.0) / 2) < 1e-9
    assert a["occ_accuracy"] == 3 / 4
    # rouge-only records (no GT) still aggregate, grounding keys simply absent
    b = aggregate_trace_quality([{"rouge_f1": 0.3}, {"rouge_f1": 0.5}])
    assert b["n"] == 2 and "occ_balanced" not in b


def test_aggregate_empty():
    assert aggregate_trace_quality([]) == {"n": 0}
