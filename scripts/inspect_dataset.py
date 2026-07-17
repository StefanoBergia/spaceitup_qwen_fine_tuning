"""Inspect the downloaded ShareRobot trajectory dataset.

Run on the login node after scripts/download.py:

    uv run scripts/inspect_dataset.py                 # full report + 16 overlay images
    uv run scripts/inspect_dataset.py --num-overlays 32 --seed 7

Writes to outputs/inspection/:
    report.txt        — sample counts, image-size / waypoint-count / coordinate stats
    overlays/*.png    — random samples with the ground-truth trajectory drawn on the image

The report checks the assumptions our data prep relies on:
  * actual PNG dimensions match meta_data.original_{width,height}
  * trajectory coordinates are pixel-space (x = width axis, y = height axis)
  * whether any waypoints fall outside image bounds
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = REPO_ROOT / "data" / "sharerobot" / "trajectory"
OUT_DIR = REPO_ROOT / "outputs" / "inspection"


def percentiles(values: list[float], name: str) -> str:
    a = np.asarray(values)
    return (
        f"{name}: min={a.min():.1f} p25={np.percentile(a, 25):.1f} "
        f"median={np.median(a):.1f} p75={np.percentile(a, 75):.1f} max={a.max():.1f}"
    )


def build_report(samples: list[dict], check_dims_n: int, rng: random.Random) -> list[str]:
    lines: list[str] = []
    add = lines.append

    add(f"Total samples: {len(samples)}")
    add("")

    add("Samples per source dataset:")
    for name, count in Counter(s["meta_data"]["original_dataset"] for s in samples).most_common():
        add(f"  {count:5d}  {name}")
    add("")

    add("Declared image sizes (meta_data original_width x original_height):")
    size_counts = Counter(
        f'{s["meta_data"]["original_width"]}x{s["meta_data"]["original_height"]}' for s in samples
    )
    for size, count in size_counts.most_common():
        add(f"  {count:5d}  {size}")
    add("")

    # Verify actual PNG dims match metadata on a random subset (opening all 6,870 is slow on NFS)
    checked = rng.sample(samples, min(check_dims_n, len(samples)))
    mismatches = []
    for s in checked:
        with Image.open(TRAJ_DIR / "images" / s["image_path"]) as im:
            if im.size != (s["meta_data"]["original_width"], s["meta_data"]["original_height"]):
                mismatches.append((s["id"], im.size, s["meta_data"]))
    add(f"PNG dims vs meta_data (random {len(checked)} samples): {len(mismatches)} mismatches")
    for mid, actual, meta in mismatches[:10]:
        add(f"  id={mid} actual={actual} declared={meta['original_width']}x{meta['original_height']}")
    add("")

    n_points = [len(s["trajectory"]) for s in samples]
    add(percentiles(n_points, "Waypoints per sample"))
    add("Waypoint count histogram:")
    for n, count in sorted(Counter(n_points).items()):
        add(f"  {n:3d} points: {count:5d}")
    add("")

    # Coordinate ranges, normalized by the declared image size — anything outside [0, 1]
    # means out-of-bounds waypoints our normalization has to clamp or drop.
    oob = 0
    xs_norm, ys_norm = [], []
    for s in samples:
        w, h = s["meta_data"]["original_width"], s["meta_data"]["original_height"]
        for x, y in s["trajectory"]:
            xs_norm.append(x / w)
            ys_norm.append(y / h)
            if not (0 <= x <= w and 0 <= y <= h):
                oob += 1
    add(percentiles(xs_norm, "x / width"))
    add(percentiles(ys_norm, "y / height"))
    total_pts = len(xs_norm)
    add(f"Out-of-bounds waypoints: {oob} / {total_pts} ({100 * oob / total_pts:.2f}%)")
    add("")

    instr_lens = [len(s["instruction"].split()) for s in samples]
    add(percentiles(instr_lens, "Instruction length (words)"))
    add("Example instructions:")
    for s in rng.sample(samples, 10):
        add(f"  [{s['meta_data']['original_dataset']}] {s['instruction']!r}")

    return lines


def draw_overlays(samples: list[dict], n: int, rng: random.Random) -> None:
    overlay_dir = OUT_DIR / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for s in rng.sample(samples, min(n, len(samples))):
        with Image.open(TRAJ_DIR / "images" / s["image_path"]) as im:
            img = im.convert("RGB")
        traj = np.asarray(s["trajectory"])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.plot(traj[:, 0], traj[:, 1], "-o", color="cyan", markersize=4, linewidth=1.5)
        ax.plot(*traj[0], "o", color="lime", markersize=9, label="start")
        ax.plot(*traj[-1], "o", color="red", markersize=9, label="end")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(s["instruction"], fontsize=9, wrap=True)
        ax.axis("off")
        fig.savefig(overlay_dir / f"sample_{s['id']:05d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"Overlay images -> {overlay_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-overlays", type=int, default=16, help="overlay images to render")
    parser.add_argument("--check-dims", type=int, default=300, help="PNGs to open for dim verification")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ann_path = TRAJ_DIR / "trajectory.json"
    if not ann_path.exists():
        raise SystemExit(f"{ann_path} not found — run scripts/download.py first")
    samples = json.loads(ann_path.read_text())

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = build_report(samples, args.check_dims, rng)
    report = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(report + "\n")
    print(report)
    print(f"\nReport -> {OUT_DIR / 'report.txt'}")

    draw_overlays(samples, args.num_overlays, rng)


if __name__ == "__main__":
    main()
