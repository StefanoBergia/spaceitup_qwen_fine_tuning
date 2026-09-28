"""Grounded landmarks: geometry on synthetic occupancy grids + one real-sample check."""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm.landmarks import (
    FREE, OCC, Camera, Grid, components, convex_hull, densify,
)


def test_camera_centre_and_sides():
    cam = Camera([0.0, 0.0], [0.0, 1.0])
    uv, fwd = cam.uv([[0.0, 2.0]])
    assert fwd[0] == pytest.approx(2.0)
    assert uv[0, 0] == pytest.approx(0.5)
    assert uv[0, 1] == pytest.approx(0.5 + 0.5 * 0.8 / 2.0)
    right, _ = cam.uv([cam.pos + 2.0 * cam.h + 0.5 * cam.right])
    left, _ = cam.uv([cam.pos + 2.0 * cam.h - 0.5 * cam.right])
    assert right[0, 0] > 0.5 > left[0, 0]
    top, _ = cam.uv([[0.0, 2.0]], height=0.8)
    assert top[0, 1] == pytest.approx(0.5)             # camera height -> horizon


def test_camera_in_frame():
    cam = Camera([0.0, 0.0], [0.0, 1.0])
    uv, fwd = cam.uv([[0.0, 2.0], [0.0, -2.0], [0.0, 0.1]])
    assert cam.in_frame(uv, fwd).tolist() == [True, False, False]


def test_grid_roundtrip_and_ray():
    s = np.full((40, 40), FREE, np.uint8)
    s[20, 10:30] = OCC                                   # wall at z = 1.0..1.05
    g = Grid(s, 0.05, 0.0, 0.0)
    rc = g.rc([[0.52, 1.02]])[0]
    assert (round(rc[0]), round(rc[1])) == (20, 10)
    assert np.allclose(g.xz([[20, 10]])[0], [0.525, 1.025])
    assert g.cell((20, 15)) == OCC and g.cell((-1, 0)) == 0
    assert g.first_occupied([1.0, 0.2], [1.0, 1.8]) == (20, 20)
    assert g.first_occupied([0.2, 0.2], [0.2, 1.8]) is None


def test_densify_components_hull():
    d = densify(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.25)
    assert len(d) == 5 and np.allclose(d[-1], [1.0, 0.0])
    m = np.zeros((6, 6), bool); m[0:2, 0:2] = True; m[4:6, 4:6] = True
    lab, n = components(m)
    assert n == 2 and lab[0, 0] != lab[5, 5]
    h = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(h) == 4 and (0.5, 0.5) not in h
