"""Overlay the prepared path + goal (colored by visibility) on fpv_enhanced.png.

Login node (CPU-only), run after prepare_habitat.py or directly on the dataset:
    uv run scripts/inspect_habitat.py --num 16

Writes outputs/inspection_habitat/overlays/*.png. Visible waypoints = green,
obstructed = red; goal ring green (visible) / red (obstructed). Sanity-checks the
coordinate + visibility convention before training.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from rover_vlm.habitat_data import DATASET_ROOT, build_record

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "inspection_habitat" / "overlays"
VIS, OBS = "#2a78d6", "#d64a2a"  # visible=blue, obstructed=red (green reserved for goal)


def draw(rec: dict, out_path: Path) -> None:
    obj = json.loads(rec["conversations"][1]["value"])
    img = Image.open(rec["image"][0]).convert("RGB")
    W, H = img.size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    pts = obj["path"]
    xs = [p[0] * W for p in pts]
    ys = [p[1] * H for p in pts]
    ax.plot(xs, ys, "-", color="#dddddd", linewidth=1.5, zorder=1)
    for (x, y, v) in pts:
        ax.plot(x * W, y * H, "o", color=(VIS if v == 1 else OBS), markersize=7, zorder=2)
    gx, gy, gv = obj["goal"]
    ax.plot(gx * W, gy * H, "*", color=("#00a000" if gv == 1 else "#d00000"),
            markersize=22, markeredgecolor="white", zorder=3)
    ax.set_title(f"{rec['id']}  (goal {'visible' if gv == 1 else 'obstructed'})", fontsize=9)
    ax.axis("off")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dirs = sorted(args.dataset_root.glob("*/samples/*/"))
    random.Random(args.seed).shuffle(dirs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for d in dirs:
        if drawn >= args.num:
            break
        rec = build_record(d)
        if rec is None:
            continue
        draw(rec, OUT_DIR / f"{rec['id']}.png")
        drawn += 1
    print(f"wrote {drawn} overlays -> {OUT_DIR}")


if __name__ == "__main__":
    main()
