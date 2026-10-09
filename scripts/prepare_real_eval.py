"""Build a real-image eval set (Habitat answer format) from a robot video + trajectory.

CPU only, login node. Windows of the recorded trajectory become path labels: for a frame
at time t, the robot's future positions until the path is L metres long are projected
into the frame (see rover_vlm.real_data). One record per frame, frames at least
--stride-s apart.

    uv run scripts/prepare_real_eval.py --dataset tum \
        --source data/real/tum/rgbd_dataset_freiburg2_pioneer_slam \
                 data/real/tum/rgbd_dataset_freiburg2_pioneer_slam2 ... \
        --name tum_pioneer
    uv run scripts/prepare_real_eval.py --dataset gnd \
        --source data/real/gnd/GMU_1_2_jcScEn_chunk01.bag --name gnd_gmu

Writes data/prepared_real/<name>/eval.json (consumed unchanged by scripts/evaluate.py
--task habitat --eval-file ...) and meta.json (parameters, kept/skipped histogram,
camera-height statistics). GND frames are decoded to data/prepared_real/<name>/images/.
Sanity-check the labels with scripts/inspect_real_eval.py before spending GPU time.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from rover_vlm.real_data import (
    TumSequence,
    WindowParams,
    iter_gnd_frames,
    iter_tum_frames,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "prepared_real"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["tum", "gnd"], required=True)
    p.add_argument("--source", nargs="+", type=Path, required=True,
                   help="TUM: sequence dirs; GND: .bag files")
    p.add_argument("--name", required=True, help="output set name -> data/prepared_real/<name>/")
    p.add_argument("--horizon-m", type=float, default=None,
                   help="fixed path length per window; default draws uniformly in [--min-len, --max-len]")
    p.add_argument("--min-len", type=float, default=3.0, help="Habitat paths are 3-12 m")
    p.add_argument("--max-len", type=float, default=12.0)
    p.add_argument("--stride-s", type=float, default=1.0, help="min time between chosen frames")
    p.add_argument("--max-window-s", type=float, default=40.0,
                   help="skip windows that take longer than this (robot idling)")
    p.add_argument("--max-gap-s", type=float, default=0.5, help="skip windows with a pose gap")
    p.add_argument("--max-goal-angle", type=float, default=45.0,
                   help="goal must lie within this many degrees of the optical axis")
    p.add_argument("--max-initial-angle", type=float, default=30.0,
                   help="first metre of travel must be within this many degrees of the optical axis")
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
    p.add_argument("--max-samples", type=int, default=None)
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
    p.add_argument("--min-horizon-m", type=float, default=None,
                   help="label recovery: keep a window that ends before --horizon-m as long as "
                        "it is at least this long, instead of dropping it as track_too_short")
    return p.parse_args()


def open_source(dataset, src, out_dir, args):
    """Build the dataset adapter for one --source and return (adapter, frame walk)."""
    if dataset == "tum":
        seq = TumSequence(src)
        return seq, iter_tum_frames
    from rover_vlm.real_data import GndBag  # rosbags import kept local

    bag = GndBag(src, out_dir / "images", cam_height=args.cam_height,
                 tf_static_from=args.tf_static_from)
    return bag, iter_gnd_frames


def main():
    args = parse_args()
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    skipped = Counter()
    records = []
    params = WindowParams.from_args(args)
    for src in args.source:
        adapter, walk = open_source(args.dataset, src, out_dir, args)
        print(adapter.summary())
        n_before = len(records)
        for res in walk(adapter, params, rng):
            if res.record is None:
                skipped[res.reason] += 1
                continue
            records.append(res.record)
            if args.max_samples and len(records) >= args.max_samples:
                break
        print(f"  kept {len(records) - n_before} from {src.name}")
        if args.max_samples and len(records) >= args.max_samples:
            break

    (out_dir / "eval.json").write_text(json.dumps(records, indent=1))
    heights = [r["real_meta"]["cam_height_m"] for r in records]
    hidden = [r["real_meta"]["frac_hidden"] for r in records]
    goal_v = [json.loads(r["conversations"][1]["value"])["goal"][2] for r in records]
    meta = {
        "dataset": args.dataset,
        "sources": [str(s) for s in args.source],
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in ("source",)},
        "kept": len(records),
        "skipped": dict(skipped),
        "cam_height_m": {"mean": float(np.mean(heights)) if heights else None,
                         "std": float(np.std(heights)) if heights else None},
        "frac_hidden_mean": float(np.mean(hidden)) if hidden else None,
        "goal_visible_frac": float(np.mean(goal_v)) if goal_v else None,
        "hfov_deg": records[0]["real_meta"]["hfov_deg"] if records else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nkept {len(records)}  skipped {sum(skipped.values())}: "
          + ", ".join(f"{k}={v}" for k, v in skipped.most_common()))
    if heights:
        print(f"camera height {meta['cam_height_m']['mean']:.3f} +- {meta['cam_height_m']['std']:.3f} m, "
              f"hidden frac {meta['frac_hidden_mean']:.2f}, goal visible {meta['goal_visible_frac']:.2f}")
    print(f"-> {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
