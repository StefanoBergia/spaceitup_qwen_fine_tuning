"""Route events (prompt d1): geometry on synthetic occupancy grids + real-frame regressions.

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (LEFT below).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rover_vlm.landmarks import FREE, OCC
from rover_vlm.route_events import (
    NEAR_CAMERA_M, candidates, first_blocker, load_frame, route_geometry, visible_cells,
)

LEFT = np.array([1.0, 0.0])


def _free():
    return np.full((100, 100), FREE, np.uint8)


def write_sample(d, state, routes, *, pose=(2.5, 0.2), heading=(0.0, 1.0), depth=None):
    """A synthetic sample dir: state.npz, meta.json (one candidate per route), optional depth."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    meta = {"pose_xz": list(pose), "heading_xz": list(heading), "flavor": "detour",
            "chord": {"crossing_xz": None}, "candidates": cands}
    (d / "meta.json").write_text(json.dumps(meta))
    if depth is not None:
        Image.fromarray(depth.astype(np.uint8), "L").save(d / "depth.png")
    return d


# S-route: bows to +x (left) past box A, then to -x (right) past box B
S_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (1.8, 2.4), (1.8, 3.0), (2.5, 3.6)]


def _s_state():
    s = _free()
    s[20:30, 52:58] = OCC        # box A: x 2.6..2.9, z 1.0..1.5 (on the route's right there)
    s[46:56, 44:54] = OCC        # box B: x 2.2..2.7, z 2.3..2.8 (on the route's left there)
    return s


def test_route_geometry_right_vector_points_right_of_travel():
    rg = route_geometry([(0.0, 0.0), (0.0, 2.0)])
    assert np.allclose(rg["right"][5], [-1.0, 0.0])      # travel +z -> right is -x (Camera.right)
    assert rg["arc"][-1] == pytest.approx(2.0)
    assert rg["offset"].max() == pytest.approx(0.0)


def test_each_obstacle_gets_its_own_side_on_an_s_route(tmp_path):
    _, frame = load_frame(write_sample(tmp_path / "s", _s_state(), [S_ROUTE]))
    cs = candidates(frame, np.array(S_ROUTE))
    sides = [c["side"] for c in cs]
    assert sides == ["left", "right"]                    # route order: A then B
    assert all(c["pass_m"] <= 0.7 for c in cs)
    assert cs[0]["outline"]["area"] >= 0.01 and len(cs[0]["outline"]["polygon"]) >= 3
    assert cs[0]["score"] > 0


def test_obstacle_beyond_pass_distance_is_not_a_candidate(tmp_path):
    s = _free()
    s[20:30, 52:58] = OCC                                # box A as above
    route = [(2.5, 0.2), (1.6, 0.6), (1.6, 3.6)]   # stays >= 0.78 m from box A after the 0.3 m start skip
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    assert candidates(frame, np.array(route)) == []


def test_occupied_cells_at_the_camera_do_not_hide_what_is_behind(tmp_path):
    # 00372 (2026-09-30): a strip 0.31 m ahead of the rover, bare floor in the image, was
    # the first occupied cell on every ray and hid the armchair.
    s = _s_state()
    s[9:11, 46:54] = OCC                                 # strip z 0.45..0.55, 0.25-0.35 m ahead
    _, frame = load_frame(write_sample(tmp_path / "s", s, [S_ROUTE]))
    target = frame.grid.xz([[25, 55]])[0]                # a cell of box A
    assert first_blocker(frame, target) == int(frame.lab[25, 55])
    cs = candidates(frame, np.array(S_ROUTE))
    assert [c["side"] for c in cs] == ["left", "right"]
    strip = int(frame.lab[10, 50])
    assert strip not in {c["blob"] for c in cs}


def test_hidden_cells_are_not_visible(tmp_path):
    s = _free()
    s[20:22, 40:60] = OCC                                # screen x 2.0..3.0 at z 1.0..1.1
    s[40:44, 48:52] = OCC                                # post behind it, x 2.4..2.6, z 2.0..2.2
    _, frame = load_frame(write_sample(tmp_path / "s", s, [S_ROUTE]))
    post = np.argwhere(frame.lab == frame.lab[41, 49])
    assert len(visible_cells(frame, post, int(frame.lab[41, 49]))) == 0


def test_near_camera_constant():
    assert NEAR_CAMERA_M == pytest.approx(0.35)
