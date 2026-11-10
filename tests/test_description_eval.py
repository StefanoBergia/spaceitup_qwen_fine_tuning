"""Description-followed metric: predicted image paths mapped back onto the frame's floor."""
import json

import numpy as np
import pytest

from rover_vlm.eval import (aggregate_description_metrics, aggregate_habitat_metrics,
                            description_follow, image_to_floor)
from rover_vlm.landmarks import Camera

CAM = Camera([0.0, 0.0], [0.0, 1.0])          # at the origin facing +z; image-left is +x


def to_image(xz):
    uv, _ = CAM.uv(np.asarray(xz, float))
    return [[float(u), float(v), 1] for u, v in uv]


def check(events, goal=(0.0, 4.0)):
    return {"pose_xz": [0.0, 0.0], "heading_xz": [0.0, 1.0], "goal_xz": list(goal), "events": events}


LEFT_ROUTE = [(0.0, 0.8), (0.8, 2.0), (0.0, 4.0)]      # bows to +x = the rover's left
RIGHT_ROUTE = [(0.0, 0.8), (-0.8, 2.0), (0.0, 4.0)]
POST = [[[0.0, 2.0], [0.05, 2.0]]]


def test_image_to_floor_inverts_the_camera():
    xz = image_to_floor(to_image([(0.5, 2.0), (-1.0, 3.0)]), [0.0, 0.0], [0.0, 1.0])
    assert np.allclose(xz, [[0.5, 2.0], [-1.0, 3.0]], atol=1e-6)


def test_points_above_the_horizon_are_dropped():
    assert len(image_to_floor([[0.5, 0.4, 1], [0.5, 0.5, 1]], [0, 0], [0, 1])) == 0


def test_pass_left_followed_and_mirror_not():
    c = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    assert description_follow(to_image(LEFT_ROUTE), c) == {"follows": 1, "events": [1], "stop": 1}
    assert description_follow(to_image(RIGHT_ROUTE), c)["follows"] == 0


def test_between_needs_opposite_sides():
    c = check([{"type": "between", "side": None,
                "obstacles": [[[1.0, 2.0]], [[-1.0, 2.0]]], "gap_xz": None}])
    assert description_follow(to_image([(0.0, 0.8), (0.0, 2.0), (0.0, 4.0)]), c)["events"] == [1]
    assert description_follow(to_image([(0.0, 0.8), (1.6, 2.0), (0.0, 4.0)]), c)["events"] == [0]


def test_stop_tolerance():
    c = check([], goal=(0.0, 4.0))
    assert description_follow(to_image([(0.0, 0.8), (0.0, 4.5)]), c)["stop"] == 1
    assert description_follow(to_image([(0.0, 0.8), (0.0, 5.0)]), c)["stop"] == 0


def test_degenerate_predictions_count_as_not_followed():
    c = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    assert description_follow([[0.5, 0.3, 1]], c)["follows"] == 0
    recs = [{"id": "a", "parsed": None, "gt": {"goal": [0.5, 0.6, 1]}, "metrics": None,
             "habitat_meta": {"desc_check": c, "pair_id": None}}]
    agg = aggregate_description_metrics(recs)
    assert agg["n_records"] == 1 and agg["follow_rate"] == 0.0


def test_aggregate_pairs_and_constant_goal_baseline():
    c_l = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    c_r = check([{"type": "pass", "side": "right", "obstacles": POST, "gap_xz": None}])
    rows = []
    for rid, c, pred, goal in (("p_L", c_l, LEFT_ROUTE, [0.5, 0.6]), ("p_R", c_r, LEFT_ROUTE, [0.4, 0.6])):
        rows.append({"id": rid, "parsed": {"path": to_image(pred)}, "gt": {"goal": goal + [1]},
                     "metrics": {"goal_point_error": 0.05}, "habitat_meta": {"desc_check": c, "pair_id": "p"}})
    agg = aggregate_description_metrics(rows)
    assert agg["follow_rate"] == 0.5 and agg["n_pairs"] == 1 and agg["pair_success"] == 0.0
    assert agg["constant_goal_endpoint_error_median"] == pytest.approx(0.05)
    assert agg["endpoint_error_median"] == pytest.approx(0.05)
    assert agg["event_rate_by_type"] == {"pass": 0.5}
    assert aggregate_description_metrics([{"id": "x", "habitat_meta": {}}]) is None


def test_habitat_aggregate_carries_description_block_only_with_desc_check():
    c = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    with_check = [{"id": "a", "parsed": {"path": to_image(LEFT_ROUTE)}, "gt": {"goal": [0.5, 0.6, 1]},
                   "metrics": None, "habitat_meta": {"desc_check": c, "pair_id": None}}]
    assert "description" in aggregate_habitat_metrics(with_check)
    without = [{"id": "b", "parsed": None, "habitat_meta": {}}]
    assert "description" not in aggregate_habitat_metrics(without)


STRAIGHT = [(0.0, 0.8), (0.0, 2.0), (0.0, 4.0)]


@pytest.mark.parametrize("event", [
    {"type": "pass", "side": "left", "obstacles": [[]], "gap_xz": None},
    {"type": "pass", "side": "left", "obstacles": [], "gap_xz": None},
    {"type": "through", "side": None, "obstacles": [], "gap_xz": None},
    {"type": "between", "side": None, "obstacles": [[[1.0, 2.0]]], "gap_xz": None},
])
def test_malformed_events_count_as_not_followed(event):
    res = description_follow(to_image(STRAIGHT), check([event]))
    assert res["events"] == [0] and res["follows"] == 0


def test_through_gap():
    c = check([{"type": "through", "side": None, "obstacles": [],
                "gap_xz": [[-0.4, 2.0], [0.4, 2.0]]}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [1]
    outside = [(1.2, 0.8), (1.2, 2.0), (1.2, 4.0)]
    assert description_follow(to_image(outside), c)["events"] == [0]


def test_duplicate_waypoints_do_not_default_to_right():
    c = check([{"type": "pass", "side": "left", "obstacles": [[[0.0, 2.0]]], "gap_xz": None}])
    route = [(0.5, 0.8), (0.5, 2.0), (0.5, 2.0), (0.5, 2.0), (0.5, 4.0)]
    assert description_follow(to_image(route), c)["events"] == [1]


def test_constant_goal_baseline_uses_the_endpoint_error_population():
    c = check([])
    rows = [{"id": "a", "parsed": {"path": to_image(STRAIGHT)}, "gt": {"goal": [0.5, 0.6, 1]},
             "metrics": {"goal_point_error": 0.1}, "habitat_meta": {"desc_check": c}},
            {"id": "b", "parsed": None, "gt": {"goal": [0.9, 0.9, 1]},
             "metrics": None, "habitat_meta": {"desc_check": c}}]
    agg = aggregate_description_metrics(rows)
    assert agg["constant_goal_endpoint_error_median"] == pytest.approx(0.0)


# --- final-review fixes (F6) ---------------------------------------------------------------
from rover_vlm.eval import PASS_TIE_M, STOP_TOL_UV  # noqa: E402


def test_stop_in_image_space_when_the_gt_end_is_known():
    c = check([], goal=(0.0, 4.0))
    pred = to_image([(0.0, 0.8), (0.0, 5.0)])         # 1 m past goal_xz: the floor check says no
    end = pred[-1][:2]
    assert description_follow(pred, c)["stop"] == 0
    assert description_follow(pred, c, gt_end_uv=end)["stop"] == 1
    assert description_follow(pred, c, gt_end_uv=[end[0] + 0.8 * STOP_TOL_UV, end[1]])["stop"] == 1
    assert description_follow(pred, c, gt_end_uv=[end[0] + 1.5 * STOP_TOL_UV, end[1]])["stop"] == 0


def test_grazing_pass_counts_regardless_of_sign():
    c = check([{"type": "pass", "side": "right", "obstacles": POST, "gap_xz": None}])
    graze = [(0.0, 0.8), (0.05 + 0.6 * PASS_TIE_M, 2.0), (0.0, 4.0)]   # on the left, but grazing
    clear = [(0.0, 0.8), (0.05 + 2.0 * PASS_TIE_M, 2.0), (0.0, 4.0)]   # clearly on the left
    assert description_follow(to_image(graze), c)["events"] == [1]
    assert description_follow(to_image(clear), c)["events"] == [0]


def test_aggregate_uses_the_gt_terminus_and_reports_the_gt_ceiling():
    c_l = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    c_r = check([{"type": "pass", "side": "right", "obstacles": POST, "gap_xz": None}])
    gt_path = to_image([(0.0, 0.8), (0.8, 2.0), (0.0, 5.0)])         # ends 1 m past goal_xz
    rows = [{"id": rid, "parsed": {"path": gt_path}, "gt": {"goal": gt_path[-1], "path": gt_path},
             "metrics": None, "habitat_meta": {"desc_check": c, "pair_id": None}}
            for rid, c in (("a", c_l), ("b", c_r))]
    agg = aggregate_description_metrics(rows)
    assert agg["stop_rate"] == 1.0                     # image-space stop vs the GT terminus
    assert agg["follow_rate"] == 0.5 and agg["gt_follow_rate"] == 0.5
    no_gt = [dict(rows[0], gt=None)]
    agg = aggregate_description_metrics(no_gt)
    assert agg["stop_rate"] == 0.0 and agg["gt_follow_rate"] is None   # floor fallback


# --- Addendum A: wall events and multi-object passes (Task 15) ----------------------------
# A wall with side s is on the ROVER's side s: the obstacle is on the route's side s, so the
# route passes it on the opposite side (_pass_side returns the opposite word).
WALL_X = [[[1.2, z] for z in (1.0, 1.5, 2.0, 2.5, 3.0)]]          # +x = the rover's left


def test_wall_followed_when_it_stays_on_the_stated_side():
    c = check([{"type": "wall", "side": "left", "obstacles": WALL_X, "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [1]
    c_r = check([{"type": "wall", "side": "right", "obstacles": WALL_X, "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c_r)["events"] == [0]


def test_wall_on_the_other_side_of_the_route_is_not_followed():
    c = check([{"type": "wall", "side": "left", "obstacles": WALL_X, "gap_xz": None}])
    beyond = [(2.0, 0.8), (2.0, 2.0), (2.0, 4.0)]                  # runs on the wall's far side (+x)
    assert description_follow(to_image(beyond), c)["events"] == [0]


def test_wall_far_away_or_grazing():
    far = [[[3.5, z] for z in (1.0, 2.0, 3.0)]]
    c = check([{"type": "wall", "side": "left", "obstacles": far, "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [0]          # > FOLLOW_NEAR_M
    graze = [[[0.05 + 0.5 * PASS_TIE_M, z] for z in (1.5, 2.0, 2.5)]]
    c = check([{"type": "wall", "side": "right", "obstacles": graze, "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [1]          # tie counts


def test_two_object_pass_needs_every_object_on_the_side():
    both_right = [[[-0.6, 1.5]], [[-0.6, 3.0]]]                   # both on the route's right
    c = check([{"type": "pass", "side": "left", "obstacles": both_right, "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [1]
    split = [[[-0.6, 1.5]], [[0.6, 3.0]]]                         # second one on the route's left
    c = check([{"type": "pass", "side": "left", "obstacles": split, "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [0]
    c = check([{"type": "pass", "side": "left", "obstacles": [[[-0.6, 1.5]], []], "gap_xz": None}])
    assert description_follow(to_image(STRAIGHT), c)["events"] == [0]          # malformed second group


def test_between_member_that_the_path_grazes_counts_on_either_side():
    # final review M-graze: a member within PASS_TIE_M flips side between a dense and a sparse
    # path, so (as for pass/wall) it counts as on either side
    c = check([{"type": "between", "side": None,
                "obstacles": [[[0.1, 2.0]], [[0.6, 2.0]]], "gap_xz": None}])
    straight = to_image([(0.0, 0.8), (0.0, 2.0), (0.0, 4.0)])          # both on the same side, one grazed
    assert description_follow(straight, c)["events"] == [1]
    far = check([{"type": "between", "side": None,
                  "obstacles": [[[0.3, 2.0]], [[0.6, 2.0]]], "gap_xz": None}])
    assert description_follow(straight, far)["events"] == [0]          # no graze: same side fails


from rover_vlm.eval import aggregate_steering_metrics, steer_scores

A_GT, B_GT = [(0.0, 0.8), (0.5, 2.0)], [(0.0, 0.8), (0.5, 2.0), (0.0, 3.5)]   # one route, two stops


def _srec(rid, sib, pred_xz, gt_xz, vp="v1", prefix=None):
    steer = {"slice": "stop", "anchor": None, "anchor_event": None, "side": None, "viewpoint": vp}
    if prefix is not None:
        steer["prefix_events"] = prefix
    return {"id": rid, "generated": json.dumps(pred_xz) if pred_xz else "",
            "parsed": {"path": to_image(pred_xz)} if pred_xz else None, "gt": {"path": to_image(gt_xz)},
            "habitat_meta": {"pair_id": rid[:-2], "sibling_id": sib, "desc_check": check([]), "steer": steer}}


def _pair(pa, pb, pid="p", prefix=None):
    return [_srec(f"{pid}_A", f"{pid}_B", pa, A_GT, prefix=prefix),
            _srec(f"{pid}_B", f"{pid}_A", pb, B_GT, prefix=prefix)]


def test_stop_steering_perfect_blind_and_swapped():
    m = aggregate_steering_metrics(_pair(A_GT, B_GT))
    assert m["pair_success"] == 1.0 and m["swap_pair_rate"] == 0.0 and m["closer_to_own_gt"] == 1.0 and m["slice"] == "stop"
    assert aggregate_steering_metrics(_pair(B_GT, B_GT))["pair_success"] == 0.0          # text-blind
    m = aggregate_steering_metrics(_pair(B_GT, A_GT))
    assert m["pair_success"] == 0.0 and m["swap_pair_rate"] == 1.0
    ra, rb = _pair(B_GT, A_GT)
    assert steer_scores(ra, rb) == (0, 1)


def test_steering_unparseable_and_missing_sibling():
    recs = _pair(None, B_GT)
    m = aggregate_steering_metrics(recs)
    assert m["n_pairs"] == 1 and m["pair_success"] == 0.0 and m["record_hit_rate"] == 0.5
    assert aggregate_steering_metrics(recs[:1]) is None
    assert aggregate_steering_metrics([{"id": "x", "habitat_meta": {}}]) is None


def test_steering_block_in_habitat_summary_and_ci():
    recs = _pair(A_GT, B_GT) + _pair(B_GT, B_GT, pid="q")
    for r in recs:
        r["metrics"] = {"mean_point_error": 0.0, "frechet": 0.0, "path_visibility_acc": 1.0, "goal_point_error": 0.0,
                        "goal_visibility_correct": 1, "goal_copy_correct": 1, "start_point_error": 0.0,
                        "start_copy_correct": 1}
        r["gt"]["goal"] = r["gt"]["path"][-1]
    s = aggregate_habitat_metrics(recs)["steering"]
    assert s["n_pairs"] == 2 and s["pair_success"] == 0.5
    lo, hi = s["pair_success_ci"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert set(s) >= {"strict_pair_success_03", "strict_pair_success_05", "identical_prediction_pairs"}


def test_description_follow_stop_tolerance_keyword():
    c = check([])
    path = to_image(LEFT_ROUTE)
    end = [path[-1][0] + 0.04, path[-1][1]]
    assert description_follow(path, c, end)["stop"] == 0
    assert description_follow(path, c, end, stop_tol_uv=0.05)["stop"] == 1


def test_steering_pair_purity_split():
    recs = _pair(A_GT, B_GT, pid="p", prefix=True) + _pair(B_GT, B_GT, pid="q", prefix=False)
    m = aggregate_steering_metrics(recs)
    assert m["n_pairs"] == 2 and m["pair_success"] == 0.5
    assert m["n_prefix_events"] == 1 and m["pair_success_prefix_events"] == 1.0
    m = aggregate_steering_metrics(_pair(A_GT, B_GT, pid="q", prefix=False))
    assert m["n_prefix_events"] == 0 and m["pair_success_prefix_events"] is None
    assert aggregate_steering_metrics(_pair(A_GT, B_GT))["n_prefix_events"] == 1     # key absent = pure


from rover_vlm.eval import TURN_FOLLOW_M, TURN_FOLLOW_SHARE  # noqa: E402

TURN_L = {"type": "turn", "side": "left", "obstacles": [], "gap_xz": None,
          "turn_xz": [0.0, 3.0], "pre_heading_xz": [0.0, 1.0], "gt_lateral_m": 1.5}


@pytest.mark.parametrize("xz,ok", [
    ([(0.0, 0.8), (0.0, 3.0), (1.5, 3.0)], 1),       # turns left (+x is the rover's left)
    ([(0.0, 0.8), (0.0, 4.5)], 0),                    # carries straight on
    ([(0.0, 0.8), (0.0, 2.5)], 0),                    # stops short of the turn
    ([(0.0, 0.8), (0.0, 3.0), (-1.5, 3.0)], 0),       # turns the other way
])
def test_turn_followed(xz, ok):
    assert description_follow(to_image(xz), check([TURN_L], goal=xz[-1]))["events"] == [ok]


def test_turn_threshold_shrinks_with_a_short_gt_turn():
    assert (TURN_FOLLOW_M, TURN_FOLLOW_SHARE) == (0.5, 0.5)
    path = to_image([(0.0, 0.8), (0.0, 3.0), (0.3, 3.0)])                   # 0.3 m left of the pre-turn line
    assert description_follow(path, check([TURN_L], goal=(0.3, 3.0)))["events"] == [0]
    short = {**TURN_L, "gt_lateral_m": 0.4}                                  # the GT itself ends 0.4 m off
    assert description_follow(path, check([short], goal=(0.3, 3.0)))["events"] == [1]


def test_turn_with_no_floor_points_is_not_followed():           # Review Focus 5
    assert description_follow([[0.5, 0.4, 1], [0.6, 0.45, 1]], check([TURN_L]))["events"] == [0]


def test_by_turn_split():
    def rec(rid, events, path_xz):
        p = to_image(path_xz)
        return {"id": rid, "parsed": {"path": p}, "gt": {"path": p, "goal": p[-1]},
                "metrics": {"goal_point_error": 0.1 if events else 0.02, "mean_point_error": 0.05 if events else 0.01},
                "habitat_meta": {"desc_check": check(events, goal=path_xz[-1]), "pair_id": None}}
    recs = [rec("t", [TURN_L], [(0.0, 0.8), (0.0, 3.0), (1.5, 3.0)]), rec("s", [], [(0.0, 0.8), (0.0, 4.0)])]
    m = aggregate_description_metrics(recs)
    assert m["by_turn"]["turn"] == {"n": 1, "follow_rate": 1.0, "gt_follow_rate": 1.0,
                                       "endpoint_error_median": 0.1, "point_error_median": 0.05}
    assert m["by_turn"]["no_turn"]["gt_follow_rate"] == 1.0
    nogt = dict(recs[0], id="x", gt={"goal": recs[0]["gt"]["goal"]})       # no GT path: left out of the group's rate
    assert aggregate_description_metrics([nogt])["by_turn"]["turn"]["gt_follow_rate"] is None
    assert m["by_turn"]["no_turn"]["n"] == 1 and m["event_rate_by_type"]["turn"] == 1.0


def test_steering_by_slice():
    recs = _pair(A_GT, B_GT, pid="p") + _pair(B_GT, B_GT, pid="q")
    for r in recs[2:]:
        r["habitat_meta"]["steer"]["slice"] = "turn"
    m = aggregate_steering_metrics(recs)
    assert m["slice"] == "all" and m["by_slice"]["stop"]["pair_success"] == 1.0
    assert m["by_slice"]["turn"]["n_pairs"] == 1 and m["by_slice"]["turn"]["pair_success"] == 0.0
    assert aggregate_steering_metrics(_pair(A_GT, B_GT))["slice"] == "stop"


# --- d3: description metrics by stop kind (spec 2026-11-10-stop-phrases-d3-design.md "Eval") ---------------------
def _stop_rec(rid, kind, end):
    chk = {"pose_xz": [0.0, 0.0], "heading_xz": [0.0, 1.0], "goal_xz": [0.0, 4.0], "events": [],
           "ending": {"type": "stop", "stop": kind} if kind else {"type": "stop"}}
    gt = [[0.5, 1.0, 1], [0.5, 0.7, 1]]
    return {"id": rid, "parsed": {"path": [[0.5, 1.0, 1], end]}, "gt": {"path": gt, "goal": gt[-1]},
            "metrics": {"goal_point_error": 0.1, "mean_point_error": 0.05},
            "habitat_meta": {"desc_check": chk, "pair_id": None}}


def test_by_stop_groups_visible_stops_by_kind():
    hit, miss = [0.5, 0.7, 1], [0.5, 0.9, 1]
    recs = [_stop_rec("c1", "place:corner", hit), _stop_rec("c2", "place:corner", hit),
            _stop_rec("c3", "place:corner", miss), _stop_rec("n1", "object:next to", miss),
            _stop_rec("n2", "object:next to", miss)]
    by = aggregate_description_metrics(recs)["by_stop"]
    assert set(by) == {"place:corner", "object:next to"}
    assert by["place:corner"]["n"] == 3 and by["place:corner"]["ending_rate"] == pytest.approx(2 / 3)
    assert by["object:next to"]["n"] == 2 and by["object:next to"]["ending_rate"] == 0.0
    assert {"follow_rate", "gt_follow_rate", "endpoint_error_median", "constant_goal_endpoint_error_median",
            "gt_ending_rate"} <= set(by["place:corner"])


def test_by_stop_is_empty_for_d2_records():
    recs = [_stop_rec("a", None, [0.5, 0.7, 1]), _stop_rec("b", None, [0.5, 0.9, 1])]
    assert aggregate_description_metrics(recs)["by_stop"] == {}
