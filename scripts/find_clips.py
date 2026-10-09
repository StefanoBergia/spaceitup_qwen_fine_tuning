"""Cut benchmark clips out of the real-image videos, and the eval set that goes with them.

CPU only, login node. Reads the per-frame index written by
`scripts/render_real_video.py --index` (so it never touches a rosbag), scores every
labelled frame against the criteria in `rover_vlm.clips`, groups qualifying frames into
contiguous runs, and for each run writes a short .mp4 cut from the full video plus the
frame records it covers.

    # 1. index every sequence (once)
    uv run scripts/render_real_video.py --dataset gnd --source data/real/gnd/NOVA_chunk01.bag \
        --cam-height 0.45 --horizon-m 5 --max-window-s 40 --allow-offscreen --index-only
    # 2. cut clips
    uv run scripts/find_clips.py --criterion curve out_of_view --per-criterion 6

Writes outputs/clips/<criterion>/<sequence>_<start>-<end>.mp4, outputs/clips/manifest.json,
and data/prepared_real/<--name>/eval.json -- the frames inside the selected clips, in the
standard record format, so `scripts/evaluate.py --task habitat --eval-file ...` runs on it
unchanged. `--criterion occluded` only ever yields TUM clips: GND has no depth and no
usable LiDAR extrinsic, so its visibility flags are all-visible by construction.
"""

import argparse
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from rover_vlm.clips import CRITERIA, CRITERIA_BY_NAME, find_runs

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO_ROOT / "outputs" / "videos"
CLIP_DIR = REPO_ROOT / "outputs" / "clips"
PREPARED = REPO_ROOT / "data" / "prepared_real"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-dir", type=Path, default=VIDEO_DIR,
                   help="where <sequence>.index.jsonl and <sequence>.mp4 live")
    p.add_argument("--sequence", nargs="*", default=None, help="default: every indexed sequence")
    p.add_argument("--criterion", nargs="+", default=[c.name for c in CRITERIA],
                   choices=[c.name for c in CRITERIA])
    p.add_argument("--per-criterion", type=int, default=6,
                   help="strongest N runs per criterion across all sequences")
    p.add_argument("--per-sequence", type=int, default=2,
                   help="cap per sequence within a criterion, so one video cannot fill the set")
    p.add_argument("--min-seconds", type=float, default=2.0, help="drop shorter runs")
    p.add_argument("--merge-gap-s", type=float, default=0.7,
                   help="join runs separated by less than this (one manoeuvre, one clip)")
    p.add_argument("--pad-s", type=float, default=1.0, help="context added each side of a run")
    p.add_argument("--threshold", type=float, default=None, help="override the criterion's own")
    p.add_argument("--name", default="real_clips", help="eval set -> data/prepared_real/<name>/")
    p.add_argument("--out-dir", type=Path, default=CLIP_DIR)
    p.add_argument("--eval-stride-s", type=float, default=0.5,
                   help="min spacing between eval records inside a clip; consecutive 15 Hz "
                        "frames are near-duplicates and inflate n without adding information "
                        "(0 = keep every frame). The clip video always stays full rate")
    p.add_argument("--no-cut", action="store_true", help="report and write the eval set, no mp4s")
    p.add_argument("--crf", type=int, default=26)
    return p.parse_args()


def resolve_image(raw):
    """Absolute path for an index entry's image.

    Eval records MUST carry absolute paths: scripts/evaluate.py opens
    `IMAGE_ROOT / rec["image"][0]` (the ShareRobot convention, where paths are relative to
    data/sharerobot/trajectory), so a relative path there silently resolves to the wrong
    place and the job dies on its first sample. The index stores whatever `--source` was
    typed, which is repo-root-relative for TUM and absolute for GND; anchoring to
    REPO_ROOT is right for both, since `Path('/a') / '/b'` is '/b'.
    """
    return str((REPO_ROOT / raw).resolve())


def load_index(path):
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in entries if e.get("answer")]  # labelled frames only


def video_fps(mp4):
    """Frames per second as ffprobe reports it, for turning frame positions into seconds."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(mp4)],
                         capture_output=True, text=True, check=True).stdout.strip().rstrip(",")
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def cut(mp4, out, t0, dur, crf):
    """Re-encode the range rather than stream-copying: a copy would snap to keyframes and
    the clip would not start on the frame the run actually starts on."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-accurate_seek",
                    "-ss", f"{t0:.3f}", "-i", str(mp4), "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)


def main():
    args = parse_args()
    indexes = sorted(args.video_dir.glob("*.index.jsonl"))
    if args.sequence:
        want = set(args.sequence)
        indexes = [p for p in indexes if p.name[: -len(".index.jsonl")] in want]
    if not indexes:
        raise SystemExit(f"no *.index.jsonl in {args.video_dir} — run render_real_video.py --index")

    picked, manifest, records = [], [], {}  # records: id -> record (a frame can be in >1 clip)
    for cname in args.criterion:
        crit = CRITERIA_BY_NAME[cname]
        if args.threshold is not None:
            crit = type(crit)(crit.name, crit.describe, crit.score, crit.eligible, args.threshold)
        found = []
        for idx_path in indexes:
            seq = idx_path.name[: -len(".index.jsonl")]
            mp4 = args.video_dir / f"{seq}.mp4"
            entries = load_index(idx_path)
            if not entries:
                continue
            fps = video_fps(mp4) if mp4.exists() else 15.0
            runs = find_runs(entries, crit,
                             min_frames=max(1, round(args.min_seconds * fps)),
                             merge_gap=round(args.merge_gap_s * fps),
                             pad=round(args.pad_s * fps))
            for r in runs[: args.per_sequence]:
                found.append((seq, mp4, fps, r))
        found.sort(key=lambda x: -x[3].peak)
        picked.extend((crit, *f) for f in found[: args.per_criterion])

    out_root = args.out_dir
    for crit, seq, mp4, fps, run in picked:
        d = out_root / crit.name
        d.mkdir(parents=True, exist_ok=True)
        t0, dur = run.start / fps, run.length / fps
        clip = d / f"{seq}_{run.start:05d}-{run.end:05d}.mp4"
        if not args.no_cut and mp4.exists():
            cut(mp4, clip, t0, dur, args.crf)
        last_t = -1e18
        for e in run.entries:
            if e["t"] - last_t < args.eval_stride_s or not e["id"]:
                continue
            last_t = e["t"]
            # One record per frame (one inference), but a frame near a corner is often
            # both `curve` and `occluded`; keep every membership so no clip loses its
            # frames to whichever criterion happened to be processed first.
            rec = records.setdefault(e["id"], {
                "id": e["id"], "image": [resolve_image(e["image"])],
                "conversations": [{"from": "human", "value": HUMAN},
                                  {"from": "gpt", "value": json.dumps(
                                      e["answer"], separators=(",", ":"))}],
                "real_meta": {**e["meta"], "clips": [], "criteria": []},
            })
            if clip.name not in rec["real_meta"]["clips"]:
                rec["real_meta"]["clips"].append(clip.name)
            if crit.name not in rec["real_meta"]["criteria"]:
                rec["real_meta"]["criteria"].append(crit.name)
        manifest.append({
            "criterion": crit.name, "sequence": seq, "clip": str(clip.relative_to(REPO_ROOT)),
            "fps": round(fps, 4),
            "start_frame": run.start, "end_frame": run.end,
            "start_s": round(t0, 2), "duration_s": round(dur, 2),
            "peak_score": round(run.peak, 4), "hits": run.n_hits,
            "labelled_frames": len(run.entries),
            "eval_records": sum(1 for r in records.values()
                                if clip.name in r["real_meta"]["clips"]),
        })

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if records:
        records = list(records.values())
        missing = [r["image"][0] for r in records if not Path(r["image"][0]).exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(records)} eval images do not exist, first:\n"
                f"  {missing[0]}\n"
                "Refusing to write the eval set — this would fail on the GPU instead. "
                "Re-render the index for that sequence, or check the frames were not deleted.")
        out_dir = PREPARED / args.name
        out_dir.mkdir(parents=True, exist_ok=True)

        def mean(key):
            return float(statistics.fmean(r["real_meta"][key] for r in records))

        def stdev(key):
            return float(statistics.pstdev(r["real_meta"][key] for r in records))

        (out_dir / "eval.json").write_text(json.dumps(records, indent=1))
        (out_dir / "meta.json").write_text(json.dumps({
            "built_by": "scripts/find_clips.py",
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "kept": len(records),
            "per_criterion": dict(Counter(c for r in records for c in r["real_meta"]["criteria"])),
            "sequences": sorted({m["sequence"] for m in manifest}),
            # same keys prepare_real_eval.py writes, so the report's set chips read a clip
            # set exactly like a full set
            "cam_height_m": {"mean": mean("cam_height_m"), "std": stdev("cam_height_m")},
            "frac_hidden_mean": mean("frac_hidden"),
            "goal_visible_frac": float(sum(
                json.loads(r["conversations"][1]["value"])["goal"][2] for r in records
            ) / len(records)),
            "hfov_deg": records[0]["real_meta"]["hfov_deg"],
        }, indent=2))

    by_c = Counter(m["criterion"] for m in manifest)
    print(f"{len(manifest)} clips ({', '.join(f'{k}={v}' for k, v in by_c.items())}), "
          f"{len(records)} labelled frames")
    for m in manifest:
        print(f"  {m['criterion']:12s} {m['sequence'][:34]:34s} "
              f"{m['start_s']:7.1f}s +{m['duration_s']:5.1f}s  peak={m['peak_score']:.3f}  "
              f"{m['labelled_frames']:4d} frames")
    print(f"-> {out_root}/manifest.json" + (f"  +  {PREPARED / args.name}/eval.json" if records else ""))


HUMAN = None  # set at import from the shared prompt


def _load_prompt():
    global HUMAN
    from rover_vlm.habitat_data import HABITAT_PROMPT
    HUMAN = HABITAT_PROMPT


_load_prompt()

if __name__ == "__main__":
    main()
