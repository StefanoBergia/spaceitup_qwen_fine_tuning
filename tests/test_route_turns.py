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
