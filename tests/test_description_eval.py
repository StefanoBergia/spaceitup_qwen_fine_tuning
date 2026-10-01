"""Description-followed metric: predicted image paths mapped back onto the frame's floor."""
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
