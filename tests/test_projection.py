import numpy as np
import pytest

from rover_vlm.projection import (
    Intrinsics,
    backproject_depth,
    densify,
    drop_to_plane,
    fit_floor_plane,
    front_runs,
    invert_pose,
    level_floor_plane,
    occluded_mask,
    pose_matrix,
    project,
    quat_to_rot,
    transform_points,
    window_by_arclength,
)


def test_from_hfov_matches_habitat_rule():
    K = Intrinsics.from_hfov(512, 512, 90.0)
    assert K.fx == pytest.approx(256.0)
    assert (K.cx, K.cy) == (256.0, 256.0)
    assert K.hfov_deg == pytest.approx(90.0)


def test_quat_identity_and_yaw():
    assert np.allclose(quat_to_rot(0, 0, 0, 1), np.eye(3))
    # 90 deg about z: x -> y
    R = quat_to_rot(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)


def test_pose_roundtrip():
    T = pose_matrix((1.0, -2.0, 0.5), (0.1, 0.2, 0.3, 0.9))
    p = np.array([[0.3, 0.4, 2.0], [-1.0, 0.0, 5.0]])
    back = transform_points(invert_pose(T), transform_points(T, p))
    assert np.allclose(back, p)


def test_project_centre_and_behind():
    K = Intrinsics.from_hfov(640, 480, 90.0)
    uv, z = project([[0, 0, 2.0], [1.0, 0, 1.0], [0, 0, -1.0]], K)
    assert np.allclose(uv[0], [320, 240])
    assert np.allclose(uv[1], [320 + K.fx, 240])  # x = z -> one focal length right of centre
    assert np.isnan(uv[2]).all() and z[2] < 0


def _synthetic_floor_depth(K, height, tilt_deg=0.0, noise=0.0, seed=0):
    """Depth image of an infinite floor `height` below a camera pitched `tilt_deg` down."""
    H, W = K.height, K.width
    vs, us = np.mgrid[0:H, 0:W]
    x = (us - K.cx) / K.fx
    y = (vs - K.cy) / K.fy
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    t = np.radians(tilt_deg)
    n = np.array([0.0, -np.cos(t), np.sin(t)])  # floor normal (pointing up) in camera frame
    denom = rays @ n
    hit = denom < -1e-6
    depth = np.where(hit, -height / np.where(hit, denom, 1), 0.0)
    depth = np.where((depth > 0) & (depth < 8.0), depth, 0.0)
    if noise:
        depth += np.random.default_rng(seed).normal(0, noise, depth.shape) * (depth > 0)
    return depth


def test_fit_floor_plane_recovers_height_and_tilt():
    K = Intrinsics.from_hfov(160, 120, 90.0)
    depth = _synthetic_floor_depth(K, height=0.8, tilt_deg=10.0, noise=0.005)
    pts = backproject_depth(depth, K, stride=2)
    plane = fit_floor_plane(pts, seed=1)
    assert plane is not None
    assert plane.height == pytest.approx(0.8, abs=0.03)
    assert plane.tilt_deg == pytest.approx(10.0, abs=2.0)
    assert plane.inlier_frac > 0.5


def test_fit_floor_plane_rejects_wall():
    K = Intrinsics.from_hfov(160, 120, 90.0)
    depth = np.full((120, 160), 2.0)  # fronto-parallel wall: normal 90 deg from "up"
    assert fit_floor_plane(backproject_depth(depth, K, stride=2)) is None


def test_drop_to_plane_level_camera():
    plane = level_floor_plane(0.8)
    out = drop_to_plane([[0.0, 0.0, 3.0], [1.0, -0.2, 2.0]], plane)
    assert np.allclose(out[:, 1], 0.8)
    assert np.allclose(out[:, [0, 2]], [[0.0, 3.0], [1.0, 2.0]])


def test_window_by_arclength():
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]], dtype=float)
    assert window_by_arclength(pos, 0, 2.5) == 3
    assert window_by_arclength(pos, 1, 3.0) == 4
    assert window_by_arclength(pos, 0, 10.0) is None


def test_densify_and_front_runs():
    d = densify([[0, 0, 0], [0, 0, 1.0]], step_m=0.25)
    assert len(d) == 5 and np.allclose(d[-1], [0, 0, 1.0])
    runs = front_runs(np.array([-1, 0.01, 0.2, 0.3, -0.5, 0.4]))
    assert runs == [(2, 4), (5, 6)]


def test_occluded_mask_rules():
    depth = np.zeros((10, 10))
    depth[5, 5] = 1.0  # something 1 m away at pixel (5,5)
    depth[2, 2] = 3.0
    uv = np.array([[5.2, 5.7], [2.0, 2.0], [50.0, 5.0], [np.nan, np.nan], [7.0, 7.0]])
    z = np.array([2.0, 3.05, 2.0, 2.0, 4.0])
    out = occluded_mask(uv, z, depth, margin_m=0.15)
    # 0: depth 1 < 2-0.15 -> occluded; 1: within margin -> visible;
    # 2 out of frame, 3 NaN, 4 depth hole -> visible
    assert out.tolist() == [True, False, False, False, False]
