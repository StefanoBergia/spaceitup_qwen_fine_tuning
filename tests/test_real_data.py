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
    track_length,
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


# --- label recovery: the three options that rescue frames the strict filters drop -------

def test_pose_at_interpolates_between_samples():
    """`index_at`'s tolerance drops a frame whose pose is perfectly bracketed."""
    from rover_vlm.real_data import Trajectory
    T = _straight_track(n=3, step=1.0)           # x = 0, 1, 2 at t = 0, 1, 2
    traj = Trajectory([0.0, 1.0, 2.0], T)
    assert traj.index_at(0.5, tol_s=0.02) is None      # today: no sample near 0.5 -> no_gt
    P = traj.pose_at(0.5, max_gap_s=2.0)
    assert P is not None and P[0, 3] == pytest.approx(0.5)
    assert np.allclose(P[:3, :3], T[0][:3, :3])        # orientation constant here
    assert np.allclose(traj.pose_at(0.0, 2.0), T[0]) and np.allclose(traj.pose_at(2.0, 2.0), T[2])
    assert traj.pose_at(0.5, max_gap_s=0.5) is None    # a real dropout is still refused
    assert traj.pose_at(-1.0, 2.0) is None and traj.pose_at(9.0, 2.0) is None


def test_window_end_index_min_length_keeps_the_tail_of_a_drive():
    class Traj:
        t = np.arange(5.0)
        positions = np.array([[i, 0, 0] for i in range(5)], dtype=float)

    # 4 m of track available, 6 m asked for
    assert window_end_index(Traj, 0, 6.0, max_window_s=60, max_gap_s=2) == "track_too_short"
    assert window_end_index(Traj, 0, 6.0, 60, 2, min_length_m=3.0) == 4   # take what there is
    assert window_end_index(Traj, 0, 6.0, 60, 2, min_length_m=4.5) == "track_too_short"
    assert window_end_index(Traj, 0, 2.0, 60, 2, min_length_m=1.0) == 2   # full horizon still wins
    assert track_length(Traj, 0, 4) == pytest.approx(4.0)


def test_nearest_depth_finds_frames_outside_the_association_window(tmp_path):
    root = tmp_path / "seq"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb, depth = ["# c"], ["# d"]
    for i in range(3):
        t = 100.0 + i * 0.1
        Image.new("RGB", (640, 480)).save(root / "rgb" / f"{t:.6f}.png")
        rgb.append(f"{t:.6f} rgb/{t:.6f}.png")
        dt = t + 0.04  # 40 ms out: past the 20 ms association tolerance, well within 80 ms
        Image.fromarray((np.ones((480, 640)) * 5000).astype(np.uint16)).save(
            root / "depth" / f"{dt:.6f}.png")
        depth.append(f"{dt:.6f} depth/{dt:.6f}.png")
    (root / "rgb.txt").write_text("\n".join(rgb))
    (root / "depth.txt").write_text("\n".join(depth))
    (root / "groundtruth.txt").write_text("# g\n100.0 0 0 0.6 0 0 0 1")
    seq = TumSequence(root)
    assert all(f.depth is None for f in seq.frames)          # today: every frame is no_depth
    path, dt = seq.nearest_depth(seq.frames[1].t)
    assert path is not None and dt == pytest.approx(0.04, abs=1e-6)
    assert seq.load_depth_file(path)[0, 0] == pytest.approx(1.0)


def test_gnd_tf_static_error_names_the_fix():
    """Chunks after the first have no /tf_static; the message must say it can be borrowed."""
    import inspect

    from rover_vlm.real_data import GndBag
    src = inspect.getsource(GndBag.__init__)
    assert "tf_static_from" in inspect.signature(GndBag.__init__).parameters
    assert "tf_static_from=<chunk01" in src


def test_goal_from_path_truncates_a_veering_route_to_its_on_axis_part(tmp_path):
    """A drive that veers off axis is rejected outright by the fixed-horizon rule; with
    --goal-from-path it is kept, truncated at the last point that is still straight ahead.
    That is what makes the label agree with the prompt, which asserts the goal is ahead."""
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    side = T[:, :3, 3].copy()
    side[30:, 1] = np.linspace(0, 8.0, 30)  # straight for 3 m, then hard left (54 deg)

    assert build_real_record("x", img, K, T[0], side, level_floor_plane(0.6))[1] == "endpoint_off_axis"

    rec, reason = build_real_record("x", img, K, T[0], side, level_floor_plane(0.6),
                                    max_goal_angle_deg=20.0, goal_from_path=True)
    assert reason is None
    ans = json.loads(rec["conversations"][1]["value"])
    assert abs(ans["goal"][0] - 0.5) < 0.2          # the goal is now near the centre column
    assert abs(rec["real_meta"]["goal_angle_deg"]) <= 20.0
    assert rec["real_meta"]["goal_from_path"] is True


def test_goal_from_path_keeps_a_detour_that_returns_to_the_axis(tmp_path):
    """The farthest on-axis point is taken, not the first violation, so an obstacle detour
    that comes back on axis keeps its bend -- the Habitat case worth evaluating."""
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    detour = T[:, :3, 3].copy()
    detour[20:40, 1] = 0.8  # swerve sideways for 2 m, then rejoin the centre line

    rec, reason = build_real_record("x", img, K, T[0], detour, level_floor_plane(0.6),
                                    max_goal_angle_deg=20.0, goal_from_path=True)
    assert reason is None
    ans = json.loads(rec["conversations"][1]["value"])
    xs = [p[0] for p in ans["path"]]
    assert abs(ans["goal"][0] - 0.5) < 0.15         # ends back on the axis
    assert max(xs) - min(xs) > 0.1                  # but the route in between still bends
    assert rec["real_meta"]["goal_dist_m"] > 4.0    # and was not cut short at the swerve


def test_goal_from_path_rejects_a_route_with_no_on_axis_point(tmp_path):
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    side = T[:, :3, 3].copy()
    side[:, 1] = side[:, 0] * 3.0  # 72 deg off axis from the first step
    assert build_real_record("x", img, K, T[0], side, level_floor_plane(0.6),
                             max_goal_angle_deg=20.0, goal_from_path=True)[1] == "no_on_axis_goal"


def test_goal_from_path_respects_the_minimum_goal_distance(tmp_path):
    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    pts = T[:, :3, 3]
    assert build_real_record("x", img, K, T[0], pts, level_floor_plane(0.6),
                             goal_from_path=True, min_goal_dist_m=99.0)[1] == "no_on_axis_goal"


def test_goal_tail_samples_a_spread_of_goal_ranges_and_is_seed_reproducible(tmp_path):
    """--goal-tail-m draws the goal from the last few metres of qualifying route, so an
    eval set gets a spread of goal distances instead of every label ending at the same
    place. Same seed -> same set, which is what makes the eval reproducible."""
    import random

    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    pts = T[:, :3, 3]

    def goal_dist(seed):
        rec, reason = build_real_record("x", img, K, T[0], pts, level_floor_plane(0.6),
                                        goal_from_path=True, goal_tail_m=2.0,
                                        rng=random.Random(seed))
        assert reason is None
        return rec["real_meta"]["goal_dist_m"]

    dists = [goal_dist(s) for s in range(12)]
    assert len(set(dists)) > 1                      # the goal actually moves between draws
    assert max(dists) - min(dists) <= 2.1           # but stays inside the requested tail
    assert goal_dist(3) == goal_dist(3)             # and is reproducible for a given seed


def test_goal_from_path_ignores_occlusion(tmp_path):
    """Habitat goals are on screen in 100 % of samples but unoccluded in only 23 %, so a
    goal hidden behind an obstacle is the normal case. Selection must not filter on it."""
    import random

    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    depth = np.zeros((120, 160))
    depth[60:, 60:100] = 1.0  # a box 1 m ahead, hiding the far half of the route

    rec, reason = build_real_record("x", img, K, T[0], T[:, :3, 3], level_floor_plane(0.6),
                                    depth_m=depth, goal_from_path=True,
                                    rng=random.Random(0))
    assert reason is None
    ans = json.loads(rec["conversations"][1]["value"])
    assert ans["goal"][2] == 0                      # the chosen goal is occluded ...
    assert 0.0 <= ans["goal"][0] <= 1.0             # ... but still on screen, as required


def test_goal_arc_and_straight_line_distances_are_both_recorded(tmp_path):
    """--min-goal-dist-m bounds distance ALONG the route; goal_dist_m is the straight line
    to the same point. They diverge whenever the route curves, so both are recorded --
    otherwise a goal 3 m along a bend reads as `goal_dist_m: 1.4` and looks like the floor
    was ignored."""
    import random

    K = Intrinsics.from_hfov(160, 120, 90.0)
    T = _straight_track()
    img = tmp_path / "f.png"
    Image.new("RGB", (160, 120)).save(img)
    bent = T[:, :3, 3].copy()
    bent[20:, 1] = np.linspace(0, 1.2, 40)  # curves away, so arc length > straight line

    rec, reason = build_real_record("x", img, K, T[0], bent, level_floor_plane(0.6),
                                    goal_from_path=True, min_goal_dist_m=3.0,
                                    rng=random.Random(0))
    assert reason is None
    m = rec["real_meta"]
    assert m["goal_arc_m"] >= 3.0                     # the floor is honoured, along the route
    assert m["goal_dist_m"] < m["goal_arc_m"]         # and the straight line is shorter
