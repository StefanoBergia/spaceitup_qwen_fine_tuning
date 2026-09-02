"""Synthetic TUM-format sequence: a level camera 0.6 m above a flat floor driving
straight then curving; checks the record builder end to end and the skip reasons."""

import json

import numpy as np
import pytest
from PIL import Image

from rover_vlm.habitat_data import HABITAT_PROMPT
from rover_vlm.projection import Intrinsics, level_floor_plane, pose_matrix
from rover_vlm.real_data import (
    TUM_FR2_K,
    TumSequence,
    build_real_record,
    initial_heading_deg,
    window_end_index,
)


def _straight_track(n=60, step=0.1, height=0.6):
    """Camera moves along world +x at `height`, optical frame: z fwd = world x, y down."""
    # camera rotation: cam x -> world -y, cam y -> world -z, cam z -> world x
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # columns = cam axes in world
    T = []
    for i in range(n):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [i * step, 0.0, height]
        T.append(M)
    return np.array(T)


def test_build_record_straight_drive_matches_habitat_shape(tmp_path):
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    rec, reason = build_real_record("r0", img, K, T[0], T[:, :3, 3], level_floor_plane(0.6))
    assert reason is None
    ans = json.loads(rec["conversations"][1]["value"])
    xs = [p[0] for p in ans["path"]]
    ys = [p[1] for p in ans["path"]]
    assert rec["conversations"][0]["value"] == HABITAT_PROMPT
    assert 3 <= len(ans["path"]) <= 12
    assert all(abs(x - 0.5) < 0.02 for x in xs)  # straight ahead -> image centre column
    assert ys[0] == pytest.approx(1.0, abs=0.01)  # starts at the bottom edge
    assert all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))  # recedes upward
    assert all(p[2] == 1 for p in ans["path"]) and ans["goal"][2] == 1  # no depth -> visible
    assert ans["goal"][1] == pytest.approx(ys[-1], abs=0.01)
    assert rec["real_meta"]["cam_height_m"] == 0.6


def test_build_record_occlusion_from_depth(tmp_path):
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    depth = np.zeros((120, 60 + 100))  # H x W
    depth[:, :] = 0.0
    depth[60:, 70:90] = 1.0  # a 1 m-away box across the path's column, lower half
    rec, reason = build_real_record("r1", img, K, T[0], T[:, :3, 3], level_floor_plane(0.6), depth_m=depth)
    assert reason is None
    ans = json.loads(rec["conversations"][1]["value"])
    assert any(p[2] == 0 for p in ans["path"])  # something got hidden behind the box


def test_skip_reasons():
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    pts = T[:, :3, 3]
    plane = level_floor_plane(0.6)
    assert build_real_record("x", "no.png", K, T[0], pts[:1], plane)[1] == "track_too_short"
    behind = pts.copy()
    behind[:, 0] *= -1  # drives backwards
    assert build_real_record("x", "no.png", K, T[0], behind, plane)[1] == "endpoint_behind"
    side = pts.copy()
    side[:, 1] = side[:, 0] * 2.0  # veers hard left: endpoint 63 deg off axis
    assert build_real_record("x", "no.png", K, T[0], side, plane)[1] == "endpoint_off_axis"
    spin = pts.copy()
    spin[:20, 1] = np.linspace(0, 1.5, 20)  # first metre sideways, then straight
    spin[:20, 0] = 0.0
    assert build_real_record("x", "no.png", K, T[0], spin, plane, max_goal_angle_deg=89)[1] == "initial_off_axis"


def test_initial_heading_and_window():
    floor = np.array([[0, 0.6, 0], [0, 0.6, 0.5], [0.5, 0.6, 1.0], [2.0, 0.6, 1.0]], dtype=float)
    assert initial_heading_deg(floor, lead_m=0.5) == pytest.approx(0.0)
    assert initial_heading_deg(floor, lead_m=1.2) == pytest.approx(np.degrees(np.arctan2(0.5, 1.0)), abs=1e-6)

    class Traj:
        t = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)

    assert window_end_index(Traj, 0, 2.5, max_window_s=60, max_gap_s=2) == 3
    assert window_end_index(Traj, 0, 3.5, max_window_s=60, max_gap_s=2) == "gt_gap"
    assert window_end_index(Traj, 0, 3.5, max_window_s=5, max_gap_s=20) == "too_slow"
    assert window_end_index(Traj, 0, 9.0, max_window_s=60, max_gap_s=20) == "track_too_short"


def test_tum_sequence_parsing(tmp_path):
    root = tmp_path / "seq"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb, depth, gt = ["# c"], ["# d"], ["# g"]
    for i in range(5):
        t = 100.0 + i * 0.1
        Image.new("RGB", (640, 480)).save(root / "rgb" / f"{t:.6f}.png")
        Image.fromarray((np.ones((480, 640)) * 5000 * 2).astype(np.uint16)).save(root / "depth" / f"{t + 0.01:.6f}.png")
        rgb.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth.append(f"{t + 0.01:.6f} depth/{t + 0.01:.6f}.png")
        for k in range(10):
            gt.append(f"{t + k * 0.01:.4f} {i * 0.05 + k * 0.005:.4f} 0 0.58 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb))
    (root / "depth.txt").write_text("\n".join(depth))
    (root / "groundtruth.txt").write_text("\n".join(gt))
    seq = TumSequence(root)
    assert seq.K == TUM_FR2_K
    assert len(seq.frames) == 5 and all(f.depth is not None for f in seq.frames)
    assert seq.camera_pose(100.2) is not None and seq.camera_pose(300.0) is None
    d = seq.load_depth(seq.frames[0])
    assert d.shape == (480, 640) and d[0, 0] == pytest.approx(2.0)
    assert np.allclose(seq.traj.positions[0], [0, 0, 0.58])
