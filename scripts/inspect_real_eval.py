"""Overlay the prepared real-image labels on their frames (CPU, login node).

    uv run scripts/inspect_real_eval.py --set tum_pioneer --num 24

Reads data/prepared_real/<set>/eval.json, writes outputs/inspection_real/<set>/*.png:
path coloured by visibility (blue visible, red obstructed), goal star. Also writes a
single contact sheet `sheet.png` for a quick look. This is the correctness gate for the
projection: the path must lie on the floor ahead of the robot, hidden runs behind
obstacles, the goal at the far end.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from rover_vlm.overlay import draw_label

REPO_ROOT = Path(__file__).resolve().parent.parent


def overlay(rec):
    """Ground truth on its own frame (grey polyline, markers coloured by visibility)."""
    return draw_label(Image.open(rec["image"][0]).convert("RGB"),
                      json.loads(rec["conversations"][1]["value"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", required=True)
    p.add_argument("--num", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    recs = json.loads((REPO_ROOT / "data" / "prepared_real" / args.set / "eval.json").read_text())
    random.Random(args.seed).shuffle(recs)
    recs = recs[:args.num]
    out = REPO_ROOT / "outputs" / "inspection_real" / args.set
    out.mkdir(parents=True, exist_ok=True)
    imgs = []
    for rec in recs:
        img = overlay(rec)
        img.save(out / f"{rec['id']}.png")
        imgs.append((rec, img))
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (rec, img) in zip(axes.flat, imgs):
        m = rec["real_meta"]
        ax.imshow(img)
        ax.set_title(f"{rec['id'][-24:]}\nL={m['horizon_m']} m  h={m['cam_height_m']} m  "
                     f"tilt={m['plane_tilt_deg']}  hid={m['frac_hidden']}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sheet.png", dpi=100)
    print(f"wrote {len(imgs)} overlays + sheet.png -> {out}")


if __name__ == "__main__":
    main()
