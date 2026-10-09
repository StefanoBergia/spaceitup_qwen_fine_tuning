"""Overlay the prepared path + goal (colored by visibility) on fpv_enhanced.png.

Login node (CPU-only). Either draw straight from the dataset, or -- preferred after a
prep -- draw the prepared records themselves, so what you eyeball is exactly what the
model will be trained on, prompt framing included:

    uv run scripts/inspect_habitat.py --num 16
    uv run scripts/inspect_habitat.py --num 16 --split data/prepared_habitat_v3/eval.json

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

from rover_vlm.habitat_data import DATASET_ROOTS, build_record, sample_dirs_by_id

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
    # round 3 carries no "goal" key: the goal is the final waypoint (and is in the prompt)
    gx, gy, gv = obj["goal"] if "goal" in obj else pts[-1]
    ax.plot(gx * W, gy * H, "*", color=("#00a000" if gv == 1 else "#d00000"),
            markersize=22, markeredgecolor="white", zorder=3)
    given = "given" if "goal" not in obj else "predicted"
    ax.set_title(f"{rec['id']}  ({given} goal, {'visible' if gv == 1 else 'obstructed'})",
                 fontsize=9)
    ax.axis("off")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, nargs="*", default=list(DATASET_ROOTS))
    p.add_argument("--split", type=Path, default=None,
                   help="prepared split json to draw instead of scanning the dataset")
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if args.split:
        records = json.loads(args.split.read_text())
        random.Random(args.seed).shuffle(records)
        records = records[: args.num]
    else:
        dirs = sorted(sample_dirs_by_id(args.dataset_root).values())
        random.Random(args.seed).shuffle(dirs)
        records = []
        for d in dirs:
            if len(records) >= args.num:
                break
            rec = build_record(d)
            if rec is not None:
                records.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        draw(rec, args.out_dir / f"{rec['id']}.png")
    print(f"wrote {len(records)} overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
