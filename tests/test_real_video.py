"""The dense frame walk and the video renderer (scripts/render_real_video.py).

The video's whole value is that it shows EVERY frame, rejects included, so these tests
pin the two properties it depends on: `iter_tum_frames` visits every frame exactly once,
and each result carries a record or a reason but never both. A synthetic TUM directory
(level camera 0.6 m above a flat floor, driving along world +x) stands in for the real
sequences, as in tests/test_real_data.py.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

from rover_vlm.projection import quat_to_rot
from rover_vlm.real_data import TUM_FR2_K, TumSequence, WindowParams, iter_tum_frames

# camera optical frame: z fwd = world +x, y down = world -z (same convention as test_real_data)
CAM_QUAT = (-0.5, 0.5, -0.5, 0.5)
HEIGHT = 0.6


def _video_mod():
    spec = importlib.util.spec_from_file_location(
        "render_real_video", Path(__file__).resolve().parent.parent / "scripts" / "render_real_video.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_real_video"] = module
    spec.loader.exec_module(module)
    return module


def _floor_depth(K, height=HEIGHT, far=6.5):
    """Depth image of a flat floor `height` below a level camera (0 above the horizon)."""
    v = np.arange(K.height)[:, None].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = height / ((v - K.cy) / K.fy)
    z[(v <= K.cy) | ~np.isfinite(z)] = 0.0
    return np.clip(z, 0.0, far) * np.ones((1, K.width))


def _tum_dir(tmp_path, name="seq", n_rgb=12, veer=0.0, dt=0.1, speed=2.0, pose_dt=0.01,
              n_pose=200):
    """A TUM sequence directory: `n_rgb` frames while the camera drives forward at `speed`
    m/s, drifting `veer` m sideways per metre travelled (0 = straight ahead)."""
    root = tmp_path / name
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    assert np.allclose(quat_to_rot(*CAM_QUAT), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    depth_png = Image.fromarray((_floor_depth(TUM_FR2_K) * 5000).astype(np.uint16))
    rgb, depth, gt = ["# color"], ["# depth"], ["# groundtruth"]
    t0 = 100.0
    for i in range(n_rgb):
        t = t0 + i * dt
        Image.new("RGB", (TUM_FR2_K.width, TUM_FR2_K.height), (90, 90, 90)).save(
            root / "rgb" / f"{t:.6f}.png")
        depth_png.save(root / "depth" / f"{t:.6f}.png")
        rgb.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth.append(f"{t:.6f} depth/{t:.6f}.png")
    for k in range(n_pose):
        t = t0 + k * pose_dt
        x = k * pose_dt * speed
        gt.append(f"{t:.6f} {x:.6f} {veer * x:.6f} {HEIGHT:.6f} "
                  f"{CAM_QUAT[0]} {CAM_QUAT[1]} {CAM_QUAT[2]} {CAM_QUAT[3]}")
    (root / "rgb.txt").write_text("\n".join(rgb))
    (root / "depth.txt").write_text("\n".join(depth))
    (root / "groundtruth.txt").write_text("\n".join(gt))
    return TumSequence(root)


def _walk(seq, horizon_m=2.0, stride_s=0.0):
    import random
    return list(iter_tum_frames(seq, WindowParams(horizon_m=horizon_m, stride_s=stride_s),
                                random.Random(0)))


def test_walk_visits_every_frame_once_with_exclusive_outcome(tmp_path):
    seq = _tum_dir(tmp_path)
    res = _walk(seq)
    assert len(res) == len(seq.frames)
    assert [r.index for r in res] == list(range(len(seq.frames)))
    assert [r.frame.t for r in res] == sorted(r.frame.t for r in res)
    for r in res:
        assert (r.record is None) != (r.reason is None), f"{r.index}: record xor reason"
        assert r.meta["sequence"] == seq.root.name and r.meta["timestamp"] == r.frame.t


def test_walk_yields_labels_for_a_straight_drive(tmp_path):
    kept = [r for r in _walk(_tum_dir(tmp_path)) if r.record is not None]
    assert kept, "a straight drive down a flat floor should produce labels"
    import json
    ans = json.loads(kept[0].record["conversations"][1]["value"])
    xs = [p[0] for p in ans["path"]]
    assert all(abs(x - 0.5) < 0.05 for x in xs)  # driving straight -> centre column
    assert kept[0].record["real_meta"]["cam_height_m"] == pytest.approx(HEIGHT, abs=0.02)


def test_rejects_are_yielded_with_a_reason_not_dropped(tmp_path):
    """A hard sideways drift is rejected by the heading filters -- and still yielded."""
    res = _walk(_tum_dir(tmp_path, veer=1.5))
    assert len(res) == 12  # every frame still present
    reasons = {r.reason for r in res if r.record is None}
    assert reasons & {"endpoint_off_axis", "initial_off_axis"}, reasons
    # the track runs out before the last frames can reach the horizon
    far = _walk(_tum_dir(tmp_path, name="far"), horizon_m=100.0)
    assert "track_too_short" in {r.reason for r in far}


def test_stride_subsamples_the_same_walk(tmp_path):
    seq = _tum_dir(tmp_path)
    dense, strided = _walk(seq), _walk(seq, stride_s=0.25)
    assert len(strided) < len(dense)
    assert {r.index for r in strided} <= {r.index for r in dense}
    by_index = {r.index: r for r in dense}
    for r in strided:  # subsampling must not change any frame's outcome
        assert by_index[r.index].reason == r.reason


def test_render_frame_draws_labels_and_dims_rejects(tmp_path):
    rv = _video_mod()
    font = ImageFont.load_default(size=13)
    res = _walk(_tum_dir(tmp_path, name="veer", veer=1.5)) + _walk(_tum_dir(tmp_path))
    good = next(r for r in res if r.record is not None)
    bad = next(r for r in res if r.record is None)
    raw = np.asarray(Image.open(good.frame.image).convert("RGB"), dtype=float)

    for r in (good, bad):
        img = rv.render_frame(r, 12, 100.0, font)
        assert img.size == (TUM_FR2_K.width, TUM_FR2_K.height + rv.HUD_H)
        body = np.asarray(img.crop((0, 0, TUM_FR2_K.width, TUM_FR2_K.height)), dtype=float)
        if r is good:
            assert np.abs(body - raw).max() > 0  # the path was drawn on it
        else:
            assert body.mean() < raw.mean() * 0.6  # rejected frames are dimmed

    assert "REJECTED" in rv.hud_lines(bad, 12, 100.0)[1]
    assert rv.hud_lines(bad, 12, 100.0)[2] == rv.HUD_BAD
    assert "REJECTED" not in rv.hud_lines(good, 12, 100.0)[1]
    assert "L=2.0m" in rv.hud_lines(good, 12, 100.0)[1]


def test_render_frame_scales_to_even_dimensions(tmp_path):
    """yuv420p needs both output dimensions even, or ffmpeg silently pads."""
    rv = _video_mod()
    good = next(r for r in _walk(_tum_dir(tmp_path)) if r.record is not None)
    img = rv.render_frame(good, 12, 100.0, ImageFont.load_default(size=13), scale=321)
    assert img.width == 321 and (img.height - rv.HUD_H) % 2 == 0


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg/ffprobe not on PATH")
def test_encode_writes_every_frame(tmp_path):
    rv = _video_mod()
    frames = [Image.new("RGB", (64, 48), (i * 40, 0, 0)) for i in range(5)]
    out = tmp_path / "v.mp4"
    assert rv.encode(iter(frames), out, 10.0, (64, 48), 23) == 5
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True).stdout.strip()
    assert n.rstrip(",") == "5"


def test_native_fps_matches_the_recording_duration(tmp_path):
    """A gappy stream must still yield a video as long as the drive: mean rate, not median."""
    rv = _video_mod()
    seq = _tum_dir(tmp_path)  # 12 frames, 0.1 s apart
    assert rv.native_fps(seq) == pytest.approx(10.0)

    class Gappy:  # 30 Hz burst then a 5 s stall -- median says 30 fps, the truth is ~1.3
        frames = [type("F", (), {"t": t})() for t in [0.0, 1 / 30, 2 / 30, 3 / 30, 5.0, 5 + 1 / 30]]

    fps = rv.native_fps(Gappy)
    assert fps == pytest.approx(5 / (5 + 1 / 30), abs=0.01)
    assert len(Gappy.frames) / fps == pytest.approx(5 + 1 / 30, rel=0.25)  # duration ~ the span


def test_cli_flags_reach_window_params():
    """Every WindowParams field with a matching CLI flag must survive into the params.

    Regression: `render_real_video.py` used to build WindowParams field by field, so
    --goal-from-path and friends were parsed and then dropped. The render finished, wrote
    a plausible video and index, and used the OLD rule -- a silent wrong answer, the worst
    kind. Building from `from_args` fixes it; this test keeps it fixed.
    """
    import sys
    from dataclasses import fields

    from rover_vlm.real_data import WindowParams

    rrv = _video_mod()
    argv = ["render_real_video.py", "--dataset", "tum", "--source", "seq",
            "--goal-from-path", "--max-goal-angle", "15", "--min-goal-dist-m", "3.0",
            "--goal-tail-m", "1.0", "--min-horizon-m", "3.0", "--depth-tol-s", "0.08"]
    old = sys.argv
    try:
        sys.argv = argv
        args = rrv.parse_args()
    finally:
        sys.argv = old
    p = rrv.window_params(args)
    assert p.goal_from_path is True
    assert p.max_goal_angle == 15.0
    assert p.min_goal_dist_m == 3.0
    assert p.goal_tail_m == 1.0
    assert p.min_horizon_m == 3.0
    assert p.depth_tol_s == 0.08
    assert p.stride_s == 0.0          # the video always walks every frame
    # and nothing silently defaults: every field that has a same-named CLI arg matches it
    for f in fields(WindowParams):
        if hasattr(args, f.name) and f.name not in ("horizon_m", "stride_s"):
            assert getattr(p, f.name) == getattr(args, f.name), f.name
