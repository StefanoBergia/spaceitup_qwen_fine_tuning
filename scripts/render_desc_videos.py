"""Render predicted vs ground-truth paths on d1 eval frames as one mp4 per set and tag.

CPU only, login node, after slurm/eval_descriptions.sbatch. Each video frame shows one image
(both routes of a pair side by side): ground truth thick grey, prediction orange, the
description under each tile; every frame is held HOLD_S seconds.

    uv run scripts/render_desc_videos.py --set eval_desc_pairs --tag d1
    uv run scripts/render_desc_videos.py --set eval_desc_entry --tag d1 --n 40

Writes outputs/eval_desc/<set>/<tag>/paths.mp4.
"""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from rover_vlm.overlay import GT_GREY, draw_polyline

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED = (255, 140, 0)
HOLD_S = 3
FPS = 2
TEXT_H = 90


def pick_frames(rows, n):
    """Up to n frames: pairs first (both routes together), then single routes, input order."""
    pairs, singles, seen = {}, [], []
    for r in rows:
        pid = (r.get("habitat_meta") or {}).get("pair_id")
        if pid:
            if pid not in pairs:
                seen.append(pid)
            pairs.setdefault(pid, []).append(r)
        else:
            singles.append([r])
    frames = [pairs[p] for p in seen if len(pairs[p]) == 2] + singles
    return frames[:n]


def missing_images(frames, images):
    """Ids of picked rows that have no image in the eval data file, in order."""
    return [r["id"] for f in frames for r in f if r["id"] not in images]


def _tile(image_path, row, px, font):
    img = Image.open(image_path).convert("RGB").resize((px, px), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    draw_polyline(d, [(p[0], p[1]) for p in row["gt"]["path"]], GT_GREY, px, px, width=8)
    if (row.get("parsed") or {}).get("path"):
        draw_polyline(d, [(p[0], p[1]) for p in row["parsed"]["path"]], PRED, px, px, width=4)
    tile = Image.new("RGB", (px, px + TEXT_H), (20, 20, 20))
    tile.paste(img, (0, 0))
    text = "\n".join(textwrap.wrap(row["habitat_meta"].get("instruction") or "", width=max(20, px // 8))[:5])
    ImageDraw.Draw(tile).multiline_text((6, px + 4), text, fill=(235, 235, 235), font=font, spacing=2)
    return tile


def panel(image_path, rows, px=448):
    font = ImageFont.load_default()
    tiles = [_tile(image_path, r, px, font) for r in rows]
    out = Image.new("RGB", (px * len(tiles), tiles[0].height), (0, 0, 0))
    for i, t in enumerate(tiles):
        out.paste(t, (i * px, 0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_desc")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v3")
    args = p.parse_args()
    run = args.eval_dir / args.set / args.tag
    rows = json.loads((run / "predictions.json").read_text())
    images = {r["id"]: r["image"][0] for r in json.loads((args.data_dir / f"{args.set}.json").read_text())}
    frames = pick_frames(rows, args.n)
    if not frames:
        raise SystemExit(f"no predictions in {run}")
    data_file = args.data_dir / f"{args.set}.json"
    missing = missing_images(frames, images)
    if missing:
        raise SystemExit(f"{len(missing)} prediction ids not in {data_file}, e.g. {missing[:3]}")
    width = max(len(f) for f in frames) * 448
    height = 448 + TEXT_H
    out = run / "paths.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        img = panel(images[f[0]["id"]], f)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(img, (0, 0))
        for _ in range(HOLD_S * FPS):
            proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
