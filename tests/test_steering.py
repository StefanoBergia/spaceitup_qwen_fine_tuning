"""Steering eval pairs: planner, pair builders, naming, selection, finalize (synthetic maps).

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (as in test_route_events.py). The image's bottom
edge is 0.8 m ahead of the camera (z = 1.0).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm import steering as st
from rover_vlm.landmarks import FREE, OCC, UNKNOWN, Camera, Grid

CAM = Camera([2.5, 0.2], [0.0, 1.0])


def _state():
    return np.full((100, 100), FREE, np.uint8)


def _box(s, rows, cols):
    s[rows[0]:rows[1], cols[0]:cols[1]] = OCC
    return s


def _grid(s):
    return Grid(s, 0.05, 0.0, 0.0)


def cell(x, z):
    return int(z / 0.05), int(x / 0.05)


def test_traversable_keeps_clearance_unknown_and_frame():
    s = _box(_state(), (50, 58), (46, 54))      # box: cell centres x 2.325..2.675, z 2.525..2.875
    s[:, 90:] = UNKNOWN                          # x >= 4.5 never seen
    m = st.traversable(_grid(s), CAM)
    assert not m[cell(2.5, 2.2)]                 # centre 0.30 m from the box
    assert m[cell(2.5, 2.0)]                     # centre 0.50 m from the box
    assert not m[cell(4.6, 4.0)]                 # unknown
    assert not m[cell(2.5, 0.6)]                 # below the image (0.43 m ahead < 0.8 m)
    assert not m[cell(0.2, 1.5)]                 # outside the 90 degree wedge
    assert m[cell(3.0, 1.5)]


def test_astar_goes_round_an_obstacle_and_fails_when_cut_off():
    s = _box(_state(), (50, 58), (46, 54))
    m = st.traversable(_grid(s), CAM)
    a, b = cell(2.5, 1.2), cell(2.5, 3.6)
    path = st.astar(m, a, b)
    assert path[0] == a and path[-1] == b and all(m[p] for p in path)
    wall = _box(_state(), (50, 52), (0, 100))    # wall across the room at z 2.5
    assert st.astar(st.traversable(_grid(wall), CAM), a, b) is None


def test_string_pull_keeps_ends_and_line_of_sight():
    m = st.traversable(_grid(_box(_state(), (50, 58), (46, 54))), CAM)
    path = st.astar(m, cell(2.5, 1.2), cell(2.5, 3.6))
    pulled = st.string_pull(m, path)
    assert pulled[0] == path[0] and pulled[-1] == path[-1] and len(pulled) < len(path)
    assert all(st.line_clear(m, p, q) for p, q in zip(pulled[:-1], pulled[1:]))


def test_plan_route_keeps_its_via_point_and_needs_a_snap():
    g = _grid(_box(_state(), (50, 58), (46, 54)))
    m = st.traversable(g, CAM)
    route = st.plan_route(g, m, [(2.5, 1.0), (3.25, 2.7), (2.5, 3.4)])
    assert route is not None and route.shape[1] == 2
    assert np.min(np.linalg.norm(route - [3.25, 2.7], axis=1)) < 0.1     # via kept as a corner
    assert route[:, 0].max() > 2.9                                        # went round on +x
    assert st.plan_route(g, m, [(2.5, 1.0), (2.5, 2.7)]) is None         # inside the box: no cell within 0.3 m


def test_route_length():
    assert st.route_length([(0, 0), (3, 4), (3, 5)]) == pytest.approx(6.0)
    assert st.route_length([(1, 1)]) == 0.0
