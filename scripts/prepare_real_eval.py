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

from rover_vlm.projection import backproject_depth, fit_floor_plane
from rover_vlm.real_data import TumSequence, build_real_record, window_end_index

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
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cam-height", type=float, default=None,
                   help="GND: camera height above ground (m); the bags do not record it")
    return p.parse_args()


def iter_tum(seq: TumSequence, args, rng, skipped: Counter):
    """Yield records from one TUM sequence."""
    last_plane = None
    next_t = -np.inf
    for k, frame in enumerate(seq.frames):
        if frame.t < next_t:
            continue
        next_t = frame.t + args.stride_s
        T_w_c = seq.camera_pose(frame.t)
        if T_w_c is None:
            skipped["no_gt"] += 1
            continue
        L = args.horizon_m if args.horizon_m else rng.uniform(args.min_len, args.max_len)
        i0 = seq.traj.index_at(frame.t, seq.pose_tol_s)
        i1 = window_end_index(seq.traj, i0, L, args.max_window_s, args.max_gap_s)
        if isinstance(i1, str):
            skipped[i1] += 1
            continue
        depth = seq.load_depth(frame)
        if depth is None:
            skipped["no_depth"] += 1
            continue
        plane = fit_floor_plane(backproject_depth(depth, seq.K), seed=args.seed + k)
        plane_source = "depth"
        if plane is None:
            if last_plane is None:
                skipped["plane_fit_failed"] += 1
                continue
            plane, plane_source = last_plane, "previous_frame"
        else:
            last_plane = plane
        future_w = seq.traj.positions[i0:i1 + 1]
        rec, reason = build_real_record(
            f"{seq.root.name}_{frame.image.stem}", frame.image, seq.K, T_w_c, future_w, plane,
            depth_m=depth, max_goal_angle_deg=args.max_goal_angle,
            max_initial_angle_deg=args.max_initial_angle,
            meta={"dataset": "tum", "sequence": seq.root.name, "timestamp": frame.t,
                  "horizon_m": round(L, 2), "plane_source": plane_source,
                  "gt_tz_m": round(float(T_w_c[2, 3]), 3)},
        )
        if rec is None:
            skipped[reason] += 1
            continue
        yield rec


def iter_gnd(bag_path: Path, out_images: Path, args, rng, skipped: Counter):
    from rover_vlm.real_data import GndBag  # rosbags import kept local

    bag = GndBag(bag_path, out_images, cam_height=args.cam_height)
    print(f"[gnd] {bag_path.name}: {len(bag.frames)} frames, {len(bag.traj.t)} odom poses, "
          f"K fx={bag.K.fx:.1f} hfov={bag.K.hfov_deg:.1f} deg, floor normal from "
          f"{bag.extrinsic_source}, cam height {bag.cam_height:.3f} m (assumed)")
    next_t = -np.inf
    for frame in bag.frames:
        if frame.t < next_t:
            continue
        next_t = frame.t + args.stride_s
        T_w_c = bag.camera_pose(frame.t)
        if T_w_c is None:
            skipped["no_odom"] += 1
            continue
        L = args.horizon_m if args.horizon_m else rng.uniform(args.min_len, args.max_len)
        i0 = bag.traj.index_at(frame.t, bag.pose_tol_s)
        i1 = window_end_index(bag.traj, i0, L, args.max_window_s, args.max_gap_s)
        if isinstance(i1, str):
            skipped[i1] += 1
            continue
        future_w = bag.ground_positions(i0, i1)
        plane = bag.floor_plane(frame.t)
        rec, reason = build_real_record(
            f"{bag_path.stem}_{frame.image.stem}", bag.materialize(frame), bag.K, T_w_c, future_w,
            plane, depth_m=None, max_goal_angle_deg=args.max_goal_angle,
            max_initial_angle_deg=args.max_initial_angle,
            meta={"dataset": "gnd", "sequence": bag_path.stem, "timestamp": frame.t,
                  "horizon_m": round(L, 2), "plane_source": bag.extrinsic_source},
        )
        if rec is None:
            skipped[reason] += 1
            continue
        yield rec


def main():
    args = parse_args()
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    skipped = Counter()
    records = []
    for src in args.source:
        if args.dataset == "tum":
            seq = TumSequence(src)
            print(f"[tum] {src.name}: {len(seq.frames)} frames, {len(seq.traj.t)} poses")
            gen = iter_tum(seq, args, rng, skipped)
        else:
            gen = iter_gnd(src, out_dir / "images", args, rng, skipped)
        n_before = len(records)
        for rec in gen:
            records.append(rec)
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
