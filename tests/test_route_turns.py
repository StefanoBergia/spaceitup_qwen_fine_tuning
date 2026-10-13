"""d2 turn events and spur routes (spec 2026-10-13): geometry on plain polylines and synthetic maps.

Polylines in the first part are in free metres. Synthetic maps (second part, from Task 2) are 5 m x 5 m at
5 cm with the camera at (2.5, 0.2) facing +z: the camera's right is -x, so +x is the rover's LEFT.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm.route_events import (
    HEADING_CHORD_M, SPUR_CHECK_M, SPUR_REVERSAL_DEG, TURN_MIN_DEG, detect_turn, has_spur, heading_profile,
)

L_RIGHT = [(0.0, 0.0), (0.0, 3.0), (-1.5, 3.0)]     # travelling +z, then -x: a right turn (right of +z is -x)


def test_constants():
    assert (TURN_MIN_DEG, HEADING_CHORD_M, SPUR_REVERSAL_DEG, SPUR_CHECK_M) == (45.0, 0.5, 150.0, 0.6)


def test_heading_profile_is_relative_to_the_start_goal_line():
    pts, arc, th, line = heading_profile([(0.0, 0.0), (0.0, 2.0)])
    assert arc[-1] == pytest.approx(2.0) and np.allclose(line, [0.0, 1.0])
    assert np.allclose(th, 0.0, atol=1e-6)
    _, _, th, _ = heading_profile(L_RIGHT)
    assert th[0] == pytest.approx(-26.6, abs=0.5)        # the first leg runs left of the start->goal line
    assert th[-1] == pytest.approx(63.4, abs=0.5)        # the last leg ends right of it


def test_l_route_is_one_right_turn_at_its_corner():
    t = detect_turn(L_RIGHT)
    assert t["side"] == "right" and t["angle_deg"] == pytest.approx(63.4, abs=0.5)
    assert abs(t["arc_m"] - 3.0) <= 0.15
    assert 2.6 <= t["start_arc_m"] <= t["arc_m"] <= t["end_arc_m"] <= 3.4
    assert np.allclose(t["pre_heading_xz"], [0.0, 1.0], atol=1e-3)     # it was going +z before the turn
    assert t["xz"][1] == pytest.approx(3.0, abs=0.06)


def test_a_mirrored_l_route_turns_left():
    t = detect_turn([(0.0, 0.0), (0.0, 3.0), (1.5, 3.0)])
    assert t["side"] == "left" and t["angle_deg"] < 0


def test_a_swerve_that_comes_back_to_the_line_is_not_a_turn():
    assert detect_turn([(0.0, 0.0), (0.8, 1.5), (0.0, 3.0), (0.0, 4.5)]) is None


def test_swing_one_way_then_turn_the_other():
    route = [(0.0, 0.0), (1.0, 1.5), (1.0, 3.5), (-1.0, 3.5)]
    t = detect_turn(route)
    assert t["side"] == "right" and t["angle_deg"] == pytest.approx(74.1, abs=0.5)
    assert heading_profile(route)[2].min() < -15            # it swung left of the line first


@pytest.mark.parametrize("phi", [44.0, 46.0])
def test_threshold(phi):
    # the last leg makes exactly phi with the start->goal line (+z), turning left
    route = [(0.0, 0.0), (-1.0, 3.0), (0.0, 3.0 + 1.0 / np.tan(np.radians(phi)))]
    t = detect_turn(route)
    assert (t is not None) == (phi >= 45.0)
    if t is not None:
        assert t["angle_deg"] == pytest.approx(-phi, abs=0.2) and t["side"] == "left"


def test_straight_diagonal_and_short_routes_never_turn():
    # a yaw-augmented frame sees a straight route diagonally; turns are measured on the route alone
    assert detect_turn([(0.0, 0.0), (3.0, 3.0)]) is None
    assert detect_turn([(0.0, 0.0), (0.0, 0.3)]) is None


@pytest.mark.parametrize("route,spur", [
    ([(0, 0), (0, 2), (1.5, 2), (0, 2), (0, 4)], True),            # exact retrace (P[k-1] == P[k+1])
    ([(0, 0), (0, 2), (1.5, 2), (0.05, 2.1), (0, 4)], True),       # near-mirror spike
    (L_RIGHT, False),
    ([(0.0, 0.0), (0.8, 1.5), (0.0, 3.0), (0.0, 4.5)], False),     # swerve
    ([(0, 0), (0, 2), (0.866 * 1.5, 2 - 0.75)], False),            # a sharp but real 120 deg corner
])
def test_has_spur(route, spur):
    assert has_spur(route) is spur


from rover_vlm.landmarks import FREE, OCC                                           # noqa: E402
from rover_vlm.route_events import (                                                # noqa: E402
    EVENTS_SCHEMA, REASONS, describe_route, describe_xz, first_blocker, load_frame, turn_anchor,
)


def write_sample(d, state, routes, *, pose=(2.5, 0.2), heading=(0.0, 1.0)):
    """A synthetic sample dir (as tests/test_route_events.py): state.npz + meta.json, one candidate per route."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    (d / "meta.json").write_text(json.dumps({"pose_xz": list(pose), "heading_xz": list(heading), "flavor": "detour",
                                             "chord": {"crossing_xz": None}, "candidates": cands}))
    return d


def _free():
    return np.full((100, 100), FREE, np.uint8)


def _cabinet(s):
    s[52:68, 96:99] = OCC        # x 4.8..4.95, z 2.6..3.4: the stop, 0.8 m past the routes' ends (no event)
    return s


def _round_state():              # a post on the inner side of the L_LEFT corner, in plain view
    s = _cabinet(_free())
    s[48:54, 57:61] = OCC        # x 2.85..3.05, z 2.4..2.7
    return s


def _behind_state():             # a box straight ahead; the route goes right of it and turns left behind it
    s = _cabinet(_free())
    s[32:52, 43:55] = OCC        # x 2.15..2.75, z 1.6..2.6
    return s


L_LEFT = [(2.5, 0.2), (2.5, 3.0), (4.0, 3.0)]                      # ahead, then +x: a left turn
BEHIND = [(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (3.9, 3.0)]
SWING = [(2.5, 0.2), (1.7, 1.5), (1.7, 3.0), (4.0, 3.0)]           # swings right, then turns left


def _turns(out):
    return [e for e in out["events"] if e["type"] == "turn"]


def test_hidden_turn_is_anchored_behind_the_box_the_route_went_round(tmp_path):
    sd = write_sample(tmp_path / "s", _behind_state(), [BEHIND])
    meta, frame = load_frame(sd)
    t = detect_turn(BEHIND)
    box = first_blocker(frame, t["xz"])
    assert box is not None                                            # scene check: the box hides the corner
    assert turn_anchor(frame, t)["blob"] == box and turn_anchor(frame, t)["relation"] == "behind"
    out = describe_route(sd, 0)
    assert out["reason"] is None
    kinds = [(e["type"], e["side"]) for e in out["events"]]
    assert kinds == [("pass", "right"), ("turn", "left")]
    turn = out["events"][1]
    assert turn["relation"] == "behind" and turn["objects"] == out["events"][0]["objects"]
    assert turn["gt_lateral_m"] == pytest.approx(1.9, abs=0.15)       # the end is 2 m left of the pre-turn line
    assert np.allclose(turn["pre_heading_xz"], [0.0, 1.0], atol=1e-3)
    assert out["stop"]["type"] == "near"


def test_visible_turn_is_anchored_round_the_inner_post_and_absorbs_its_pass(tmp_path):
    out = describe_route(write_sample(tmp_path / "s", _round_state(), [L_LEFT]), 0)
    assert out["reason"] is None
    assert [(e["type"], e["side"], e.get("relation")) for e in out["events"]] == [("turn", "left", "round")]
    assert out["objects"][out["events"][0]["objects"][0]]["kind_hint"] == "object"


def test_turn_with_nothing_nameable_near_it_is_bare(tmp_path):
    out = describe_route(write_sample(tmp_path / "s", _cabinet(_free()), [L_LEFT]), 0)
    assert out["reason"] is None and len(out["events"]) == 1
    assert (out["events"][0]["type"], out["events"][0]["relation"], out["events"][0]["objects"]) == ("turn", None, [])


def test_turn_alone_after_a_swing_is_still_no_events_bent(tmp_path):
    assert describe_route(write_sample(tmp_path / "s", _cabinet(_free()), [SWING]), 0)["reason"] == "no_events_bent"


@pytest.mark.parametrize("cut,has_turn", [
    ([(2.5, 0.2), (1.9, 1.2), (1.9, 2.4)], False),                    # stops before the turn starts
    ([(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (2.0, 3.0)], False),        # stops inside the turn (Review Focus 4)
    ([(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (3.2, 3.0)], True),         # has finished turning
])
def test_a_cut_keeps_the_full_routes_turn_only_once_finished(tmp_path, cut, has_turn):
    meta, frame = load_frame(write_sample(tmp_path / "s", _behind_state(), [BEHIND]))
    out = describe_xz(meta, frame, np.array(cut), turn=detect_turn(BEHIND))
    assert out["reason"] is None
    assert bool(_turns(out)) is has_turn


def test_spur_routes_are_dropped(tmp_path):
    spur = [(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (1.0, 3.0), (1.9, 3.0), (3.9, 3.0)]
    assert describe_route(write_sample(tmp_path / "s", _behind_state(), [spur]), 0)["reason"] == "spur"
    assert "spur" in REASONS and EVENTS_SCHEMA == "d2-turns"


def test_describe_route_passes_the_routes_own_turn(tmp_path):
    sd = write_sample(tmp_path / "s", _behind_state(), [BEHIND])
    meta, frame = load_frame(sd)
    assert describe_xz(meta, frame, np.array(BEHIND), turn=detect_turn(BEHIND)) == describe_route(sd, 0)
    assert not _turns(describe_xz(meta, frame, np.array(BEHIND)))       # no turn handed in -> none emitted
