"""Render a real-image source sequence as video with its ground-truth path overlaid.

CPU only, login node. This is the *temporal* correctness gate on the labels that
scripts/inspect_real_eval.py checks as shuffled stills: driven as video, a projected path
must stay welded to the same floor features while the robot rolls over them. Sliding or
floating means the camera height or the floor plane is wrong, which a single frame hides.

Unlike the eval sets it walks EVERY frame, not one every --stride-s, and it keeps the
frames the label filters reject -- dimmed, with the reason printed -- so you can see what
is being thrown away (TUM keeps only ~1 candidate frame in 8). Uses the same frame walk as
scripts/prepare_real_eval.py (rover_vlm.real_data.iter_*_frames), so what you watch is what
that script would have written.

    uv run scripts/render_real_video.py --dataset tum \
        --source data/real/tum/rgbd_dataset_freiburg2_pioneer_slam ...
    uv run scripts/render_real_video.py --dataset gnd \
        --source data/real/gnd/GMU_1_2_jcScEn_chunk01.bag --cam-height 0.45

One .mp4 per --source -> outputs/videos/<sequence>.mp4. Needs ffmpeg on PATH.
"""

import argparse
import itertools
import json
import random
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rover_vlm.overlay import draw_label
from rover_vlm.real_data import TumSequence, WindowParams, iter_gnd_frames, iter_tum_frames

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "videos"

HUD_H = 36
HUD_BG = (18, 18, 18)
HUD_FG = (232, 232, 232)
HUD_DIM = (150, 150, 150)
HUD_BAD = (240, 96, 80)
REJECT_MIX = 0.55  # fraction of black blended into a rejected frame


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["tum", "gnd"], required=True)
    p.add_argument("--source", nargs="+", type=Path, required=True,
                   help="TUM: sequence dirs; GND: .bag files")
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--horizon-m", type=float, default=5.0,
                   help="fixed path length; a random length per frame makes the path flicker. "
                        "Pass 0 to draw uniformly in [--min-len, --max-len] like the eval sets")
    p.add_argument("--min-len", type=float, default=3.0)
    p.add_argument("--max-len", type=float, default=12.0)
    p.add_argument("--every", type=int, default=1, help="render every Nth source frame")
    p.add_argument("--max-frames", type=int, default=None, help="stop early (preview runs)")
    p.add_argument("--fps", type=float, default=None,
                   help="default: the sequence's own rate, so the video plays in real time")
    p.add_argument("--scale", type=int, default=None, help="output width in px (default: native)")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--index", action="store_true",
                   help="also write <out>.index.jsonl: one line per rendered frame with its "
                        "label, so scripts/find_clips.py never has to re-read the bag")
    p.add_argument("--index-only", action="store_true",
                   help="write the index and skip encoding (no image decode for GND)")
    p.add_argument("--allow-offscreen", action="store_true",
                   help="keep frames whose goal leaves the image instead of rejecting them "
                        "(a superset of the normal labels; needed for out-of-view clips)")
    p.add_argument("--max-window-s", type=float, default=90.0,
                   help="skip windows that take longer than this (robot idling)")
    p.add_argument("--max-gap-s", type=float, default=0.5)
    p.add_argument("--max-goal-angle", type=float, default=45.0)
    p.add_argument("--max-initial-angle", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cam-height", type=float, default=None,
                   help="GND: camera height above ground (m); the bags do not record it")
    p.add_argument("--tf-static-from", type=Path, default=None,
                   help="GND: bag to borrow /tf_static from (only chunk01 of a recording "
                        "carries it, but the camera mount is static for the whole recording, "
                        "so chunk01's tree is correct for chunk02..NN)")
    p.add_argument("--depth-tol-s", type=float, default=0.0,
                   help="TUM label recovery: when a frame has no depth image inside the strict "
                        "association window, use the nearest one within this many seconds "
                        "(0 = off, and the frame is dropped as no_depth). Every TUM colour frame "
                        "has depth within 0.07 s, so 0.08 recovers all of them")
    p.add_argument("--interp-pose", action="store_true",
                   help="label recovery: interpolate the trajectory at the frame timestamp "
                        "instead of requiring a sample inside the association tolerance "
                        "(TUM mocap runs at 300 Hz, so the true pose is bracketed)")
    p.add_argument("--goal-from-path", action="store_true",
                   help="choose the goal as the farthest point of the recorded route that is "
                        "actually straight ahead (within --max-goal-angle, on screen, at least "
                        "--min-goal-dist-m along the track), instead of taking the fixed "
                        "--horizon-m endpoint wherever it lands. The training prompt says the "
                        "goal IS straight ahead, so without this the label contradicts the "
                        "prompt -- real endpoints sit a median 37 deg off axis")
    p.add_argument("--min-goal-dist-m", type=float, default=1.0,
                   help="with --goal-from-path, reject a goal nearer than this along the route")
    p.add_argument("--goal-tail-m", type=float, default=0.0,
                   help="with --goal-from-path, draw the goal uniformly from the qualifying "
                        "route points within this many metres of the last one (seeded by "
                        "--seed), instead of always taking the last one. Metres rather than "
                        "sample count because TUM mocap and GND odometry differ by ~10x in rate")
    p.add_argument("--min-horizon-m", type=float, default=None,
                   help="label recovery: keep a window that ends before --horizon-m as long as "
                        "it is at least this long, instead of dropping it as track_too_short")
    return p.parse_args()


def window_params(args):
    """WindowParams for a video render.

    `from_args` picks up every field whose name matches a CLI arg, so a new window option
    works the moment it is added. Listing the fields by hand here once meant three new
    flags were parsed and then silently dropped -- the render ran under the old rule and
    said nothing. Only the three that cannot match by name are overridden: the video always
    walks every frame, `--horizon-m 0` means "random per frame", and `--allow-offscreen` is
    spelled without the `_goal` suffix.
    """
    return replace(WindowParams.from_args(args),
                   horizon_m=args.horizon_m or None,
                   stride_s=0.0,
                   allow_offscreen_goal=args.allow_offscreen)


def open_source(dataset, src, args):
    """Adapter + its frame walk, mirroring scripts/prepare_real_eval.py::open_source."""
    if dataset == "tum":
        return TumSequence(src), iter_tum_frames
    from rover_vlm.real_data import GndBag  # rosbags import kept local

    if args.cam_height is None:
        raise SystemExit("GND needs --cam-height (the bags do not record it; the eval set used 0.45)")
    # GndBag decodes every JPEG to disk. Reuse the prepared set's cache when it already
    # holds this bag (~6 k frames per bag); otherwise decode under outputs/ rather than
    # writing new frames into a prepared eval set.
    cached = REPO_ROOT / "data" / "prepared_real" / "gnd_campus" / "images"
    images_dir = cached if (cached / src.stem).is_dir() else args.out_dir / "_frames"
    return (GndBag(src, images_dir, cam_height=args.cam_height,
                   tf_static_from=args.tf_static_from), iter_gnd_frames)


def native_fps(adapter):
    """Mean frame rate of the source, so the video lasts as long as the drive did.

    NOT the median interval: TUM's colour stream is a mix of ~30 Hz bursts and dropped
    frames, so its median rate runs 20-35 % away from its mean and a video encoded at the
    median would claim the wrong duration (pioneer_slam: 184 s for a 155 s recording).
    Frames are played back to back at a constant rate, so motion still speeds up across a
    dropped-frame gap -- but the clock is right.
    """
    t = np.array([f.t for f in adapter.frames])
    span = t[-1] - t[0] if len(t) > 1 else 0.0
    return float((len(t) - 1) / span) if span > 0 else None


def hud_lines(res, n_frames, t0):
    """(top line, bottom line, bottom colour) for one frame's status strip."""
    m, top = res.meta, f"{res.meta['sequence']}   frame {res.index + 1}/{n_frames}   " \
                       f"t={res.frame.t - t0:6.1f}s"
    if res.record is None:
        L = f"L={m['horizon_m']:.1f}m   " if "horizon_m" in m else ""
        return top, f"REJECTED: {res.reason}   {L}(no label written)", HUD_BAD
    rm = res.record["real_meta"]
    goal = json.loads(res.record["conversations"][1]["value"])["goal"]
    return top, (f"L={rm['horizon_m']:.1f}m   goal {rm['goal_dist_m']:.1f}m "
                 f"@{rm['goal_angle_deg']:+.0f}deg   h={rm['cam_height_m']:.3f}m   "
                 f"tilt={rm['plane_tilt_deg']:+.1f}deg   hidden={rm['frac_hidden']:.2f}   "
                 f"goal {'visible' if goal[2] == 1 else 'obstructed'}"), HUD_FG


def render_frame(res, n_frames, t0, font, scale=None):
    """One video frame: the source image with its label (or dimmed) plus a status strip."""
    img = Image.open(res.frame.image).convert("RGB")
    if res.record is None:
        img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), REJECT_MIX)
    else:
        draw_label(img, json.loads(res.record["conversations"][1]["value"]))
    if scale and scale != img.width:
        img = img.resize((scale, max(2, round(img.height * scale / img.width) // 2 * 2)),
                         Image.LANCZOS)
    out = Image.new("RGB", (img.width, img.height + HUD_H), HUD_BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    top, bottom, colour = hud_lines(res, n_frames, t0)
    d.text((8, img.height + 3), top, fill=HUD_DIM, font=font)
    d.text((8, img.height + 18), bottom, fill=colour, font=font)
    return out


def encode(frames, path, fps, size, crf):
    """Pipe raw RGB frames straight into ffmpeg (no intermediate PNGs on NFS)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", f"{fps:.4f}", "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    try:
        for img in frames:
            proc.stdin.write(img.tobytes())
            n += 1
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed with code {proc.returncode}")
    return n


def index_entry(res, v, fps):
    """One line of the per-frame index: enough to pick clips without re-reading the bag.

    `v` is the frame's position in the rendered video (so a clip's time range is v/fps),
    `answer` the parsed label, `meta` its real_meta. Rejected frames are kept with their
    reason and a null answer so a run of them is visible as a gap, not as missing data.
    """
    ans = json.loads(res.record["conversations"][1]["value"]) if res.record else None
    return {
        "v": v, "i": res.index, "t": res.frame.t, "reason": res.reason,
        "id": res.record["id"] if res.record else None,
        "image": str(Path(res.frame.image).resolve()),  # consumers need an absolute path
        "answer": ans or {}, "meta": res.record["real_meta"] if res.record else res.meta,
    }


def render_source(src, args, font):
    adapter, walk = open_source(args.dataset, src, args)
    print(adapter.summary(), flush=True)
    fps = args.fps or native_fps(adapter) or 15.0
    fps = fps / args.every
    params = window_params(args)
    n_frames, t0 = len(adapter.frames), adapter.frames[0].t
    want_index = args.index or args.index_only
    stats, size = Counter(), None
    out_path = args.out_dir / f"{src.stem if src.is_file() else src.name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = open(out_path.with_suffix(".index.jsonl"), "w") if want_index else None

    def frames():
        nonlocal size
        v = 0
        for i, res in enumerate(walk(adapter, params, random.Random(args.seed))):
            if i % args.every:
                continue
            stats["kept" if res.record is not None else res.reason] += 1
            if index is not None:
                index.write(json.dumps(index_entry(res, v, fps)) + "\n")
            v += 1
            if args.index_only:
                if args.max_frames and v >= args.max_frames:
                    return
                continue
            img = render_frame(res, n_frames, t0, font, args.scale)
            if size is None:
                size = img.size
            elif img.size != size:  # a sequence with mixed image sizes would corrupt the pipe
                raise SystemExit(f"frame size changed {size} -> {img.size} at {res.frame.image}")
            yield img
            written = sum(stats.values())
            if written % 250 == 0:
                print(f"  {written} frames ({stats['kept']} labelled)", flush=True)
            if args.max_frames and written >= args.max_frames:
                return

    if args.index_only:
        for _ in frames():
            pass
        index.close()
        rejected = sum(v for k, v in stats.items() if k != "kept")
        print(f"  indexed {sum(stats.values())} frames -> {out_path.with_suffix('.index.jsonl')}")
        print(f"  labelled {stats['kept']}, rejected {rejected}: "
              + ", ".join(f"{k}={v}" for k, v in stats.most_common() if k != "kept"), flush=True)
        return

    # ffmpeg needs the frame size up front, so render one frame before opening the pipe
    gen = frames()
    first = next(gen, None)
    if first is None:
        print(f"  no frames for {src}")
        return
    n = encode(itertools.chain([first], gen), out_path, fps, first.size, args.crf)
    if index is not None:
        index.close()
    rejected = sum(v for k, v in stats.items() if k != "kept")
    print(f"  {n} frames @ {fps:.2f} fps ({n / fps:.0f} s) -> {out_path}")
    print(f"  labelled {stats['kept']}, rejected {rejected}: "
          + ", ".join(f"{k}={v}" for k, v in stats.most_common() if k != "kept"), flush=True)


def main():
    args = parse_args()
    font = ImageFont.load_default(size=13)
    for src in args.source:
        render_source(src, args, font)


if __name__ == "__main__":
    main()
