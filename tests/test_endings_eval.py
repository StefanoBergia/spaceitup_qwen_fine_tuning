"""d2 endings eval (spec 2026-10-13 Addendum T2): the behind check, rates by ending, desc_check's ending."""
import numpy as np
import pytest

from rover_vlm.descriptions import desc_check
from rover_vlm.eval import aggregate_description_metrics, behind_followed, description_follow, point_polygon_distance
from rover_vlm.landmarks import Camera

BOX = [[0.4, 0.5], [0.6, 0.5], [0.6, 0.7], [0.4, 0.7]]
HID = [[0.5, 1.0, 1], [0.5, 0.8, 1], [0.5, 0.72, 1], [0.52, 0.66, 0], [0.7, 0.6, 0]]   # out of sight inside BOX
VIS = [[0.5, 1.0, 1], [0.5, 0.7, 1]]


def test_point_polygon_distance():
    assert point_polygon_distance([0.5, 0.6], BOX) == 0.0
    assert point_polygon_distance([0.65, 0.6], BOX) == pytest.approx(0.05)
    assert point_polygon_distance([0.5, 0.6], [[0, 0], [1, 1]]) == float("inf")


def test_behind_followed():
    assert behind_followed(HID, BOX) == 1                       # the tail after going out of sight can go anywhere
    assert behind_followed(HID[:3], BOX) == 0                   # ends in view
    assert behind_followed([[0.5, 1.0, 1], [0.9, 0.8, 1], [0.95, 0.7, 0]], BOX) == 0     # out of sight elsewhere
    assert behind_followed([[0.5, 1.0, 1], [0.62, 0.65, 0]], BOX) == 1                   # within the 0.03 dilation


def test_behind_followed_tests_any_waypoint_of_the_final_hidden_run():
    # CPU ceiling 2026-10-14: the GT's FIRST hidden waypoint misses X's outline on ~10% of routes (it goes out of sight
    # just past the outline's edge); any waypoint of the final hidden run within 0.03 passes 0.955 eval / 0.968 train.
    from rover_vlm.eval import BEHIND_POINT
    path = [[0.5, 1.0, 1], [0.75, 0.8, 1], [0.7, 0.7, 0], [0.55, 0.6, 0]]   # out of sight right of the box, then behind it
    assert BEHIND_POINT == "any_hidden"
    assert behind_followed(path, BOX) == 1 and behind_followed(path, BOX, point="first_hidden") == 0


def test_behind_followed_midpoint():
    path = [[0.5, 1.0, 1], [0.5, 0.75, 1], [0.5, 0.45, 0]]     # first hidden waypoint above the box
    assert behind_followed(path, BOX) == 0 and behind_followed(path, BOX, point="midpoint") == 1


@pytest.mark.parametrize("path,outline", [                     # Review Focus 3
    ([[0.5, 1.0, 1], [0.5, 0.6, 0]], []),
    ([[0.5, 1.0, 1], [0.5, 0.6, 0]], None),
    ([[0.5, 1.0], [0.5, 0.6]], BOX),
    ([], BOX),
])
def test_behind_followed_degenerate_is_zero(path, outline):
    assert behind_followed(path, outline) == 0


def _check(ending, events=()):
    return {"pose_xz": [0.0, 0.0], "heading_xz": [0.0, 1.0], "goal_xz": [0.0, 4.0], "events": list(events),
            "ending": ending}


def test_description_follow_judges_the_ending_by_type():
    assert description_follow(HID, _check({"type": "behind", "outline": BOX}), gt_end_uv=[0.9, 0.9])["stop"] == 1
    assert description_follow(HID, _check({"type": "stop"}), gt_end_uv=[0.9, 0.9])["stop"] == 0
    assert description_follow(HID, _check({"type": "turn"}), gt_end_uv=[0.9, 0.9])["stop"] == 1
    old = _check(None)
    del old["ending"]                                                            # d1 records: a stop
    assert description_follow(VIS, old, gt_end_uv=[0.5, 0.7])["stop"] == 1


CAM = Camera([0.0, 0.0], [0.0, 1.0])


def _img(xz, v=1):
    uv, _ = CAM.uv(np.asarray(xz, float))
    return [[float(a), float(b), v] for a, b in uv]


TURN_EV = {"type": "turn", "side": "left", "obstacles": [], "gap_xz": None, "turn_xz": [0.0, 2.0],
           "pre_heading_xz": [0.0, 1.0], "gt_lateral_m": 1.0}


def _rec(rid, pred, gt, ending, events=()):
    return {"id": rid, "parsed": {"path": pred}, "gt": {"path": gt, "goal": gt[-1]},
            "metrics": {"goal_point_error": 0.1, "mean_point_error": 0.05},
            "habitat_meta": {"desc_check": _check(ending, events), "pair_id": None}}


def test_aggregate_reports_rates_by_ending():
    left = _img([(0.0, 0.8), (0.0, 2.0), (1.0, 3.0)])           # turns left (+x is the rover's left)
    recs = [_rec("b1", HID, HID, {"type": "behind", "outline": BOX}),
            _rec("b2", VIS, HID, {"type": "behind", "outline": BOX}),
            _rec("s1", VIS, VIS, {"type": "stop"}),
            _rec("t1", left, left, {"type": "turn"}, [TURN_EV])]
    m = aggregate_description_metrics(recs)
    assert m["behind_rate"] == 0.5 and m["stop_rate"] == 1.0 and m["ending_rate"] == pytest.approx(0.75)
    assert set(m["by_ending"]) == {"behind", "stop", "turn"}
    b = m["by_ending"]["behind"]
    assert b["n"] == 2 and b["ending_rate"] == 0.5 and b["gt_ending_rate"] == 1.0
    assert m["by_ending"]["turn"]["ending_rate"] == 1.0


def test_desc_check_carries_the_ending():
    objects = {"f:b1": {"cells_xz": [[0.0, 2.0]], "polygon": BOX}}
    ev = {"pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 4], "events": []}
    assert desc_check({**ev, "stop": {"type": "behind", "object": "f:b1"}}, objects)["ending"] == {"type": "behind", "outline": BOX}
    assert desc_check({**ev, "stop": {"type": "turn", "object": None}}, objects)["ending"] == {"type": "turn"}
    assert desc_check({**ev, "stop": {"type": "near", "object": "f:b1"}}, objects)["ending"] == {"type": "stop"}
    assert desc_check(ev, objects)["ending"] == {"type": "stop"}                        # rows without a stop
