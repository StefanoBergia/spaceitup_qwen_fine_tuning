"""Re-render the benchmark clips with the models' predicted paths beside the ground truth.

CPU only, login node. Runs AFTER `sbatch slurm/eval_clips.sbatch <SET>`. Joins each clip's
frames (from data/prepared_real/<SET>/eval.json, whose `real_meta.clips` lists every clip a
frame belongs to -- a corner frame is often both `curve` and `occluded`) with the
per-frame predictions each tag wrote, and encodes one video per clip:

    [ ground truth + model A ] [ ground truth + model B ] [ model B's <think> reasoning ]

Ground truth is the thick grey line in both panels, so the two models are compared against
the same reference and against each other in one glance. The reasoning panel is only drawn
for a tag evaluated with --enable-thinking; its text is also written verbatim, per frame,
to <clip>.traces.jsonl next to the video, because a trace is easier to grep than to read
off a video frame.

    uv run scripts/render_clip_predictions.py --set real_clips
    uv run scripts/render_clip_predictions.py --set real_clips --tags plain_2b traced_2b \
        --trace-tag traced_2b --criterion curve

Writes outputs/clips/predictions/<criterion>/<clip stem>.{mp4,traces.jsonl}.
"""

import argparse
import itertools
import json
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from rover_vlm.overlay import GT_GREY, WHITE, draw_goal, draw_path, draw_polyline
from rover_vlm.trace_eval import reasoning_and_answer

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_PX = 500
LINE_H = 13
MAX_TRACE_LINES = 60   # a 384-token trace wraps to ~30 lines; beyond this it is truncated
TRACE_W = 380
HUD_H = 34
BG = (18, 18, 18)
FG = (232, 232, 232)
DIM = (150, 150, 150)
# one colour per model, distinct from GT grey and from each other in greyscale too
TAG_COLOURS = ((235, 140, 40), (80, 150, 235), (150, 90, 200))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="real_clips", help="data/prepared_real/<set>/eval.json")
    p.add_argument("--eval-root", type=Path, default=REPO_ROOT / "outputs" / "eval_real")
    p.add_argument("--tags", nargs="+", default=["plain_2b", "traced_2b"])
    p.add_argument("--trace-tag", default="traced_2b",
                   help="tag whose <think> reasoning is shown and saved ('' for none)")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "outputs" / "clips" / "manifest.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "clips" / "predictions")
    p.add_argument("--criterion", nargs="*", default=None, help="default: all")
    p.add_argument("--fps", type=float, default=None, help="default: each clip's own rate")
    p.add_argument("--crf", type=int, default=26)
    return p.parse_args()


def load_predictions(eval_root, eval_set, tag):
    path = eval_root / eval_set / tag / "predictions.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — run: sbatch slurm/eval_clips.sbatch {eval_set}")
    return {r["id"]: r for r in json.loads(path.read_text())}


def wrap_trace(text, font, width_px, draw):
    """Greedy wrap by measured pixel width — the trace font is proportional."""
    lines = []
    for para in (text or "(no reasoning)").split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            if draw.textlength(trial, font=font) <= width_px or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def panel(image_path, gt, pred, colour, max_px=PANEL_PX):
    """One model's answer over the frame, with ground truth underneath as the reference."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    if gt and gt.get("path"):
        draw_polyline(d, [(p[0], p[1]) for p in gt["path"]], GT_GREY, W, H, width=9)
    if pred and pred.get("path"):
        draw_path(d, pred["path"], colour, W, H)
    if pred and pred.get("goal"):
        draw_goal(d, pred["goal"], W, H)
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    return img


def trace_lines(text, mono, scratch):
    """Wrapped reasoning lines, capped. Kept separate from `compose` because the video's
    height must be fixed for the whole clip -- ffmpeg is fed raw frames of one size -- so
    the caller measures every frame's trace first and sizes the canvas to the longest."""
    lines = wrap_trace(text, mono, TRACE_W - 20, scratch)
    return lines[:MAX_TRACE_LINES] + (["..."] if len(lines) > MAX_TRACE_LINES else [])


def compose(rec, preds_by_tag, tags, colours, lines, fonts, err_by_tag, height=None):
    """Panels side by side + a reasoning column + a status strip, as one video frame."""
    body, small, mono = fonts
    gt = rec["gt"]
    panels = [panel(rec["image"], gt, preds_by_tag.get(t), colours[t]) for t in tags]
    ph = max(p.height for p in panels)
    pw = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    width = pw + (TRACE_W if lines is not None else 0)
    width += width % 2
    height = height or (ph + HUD_H)
    height += height % 2

    out = Image.new("RGB", (width, height), BG)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 6
    d = ImageDraw.Draw(out)
    for i, t in enumerate(tags):  # label each panel with its model, in that model's colour
        lx = sum(panels[j].width + 6 for j in range(i))
        d.rectangle([lx, 0, lx + 8 + int(d.textlength(t, font=small)), 16], fill=BG)
        d.text((lx + 4, 2), t, fill=colours[t], font=small)

    if lines is not None:
        tx = pw + 10
        d.text((tx, 4), "reasoning", fill=DIM, font=small)
        y = 20
        for line in lines:
            d.text((tx, y), line, fill=FG, font=mono)
            y += LINE_H

    errs = "   ".join(f"{t} err={err_by_tag[t]:.3f}" if err_by_tag.get(t) is not None
                      else f"{t} unparsed" for t in tags)
    d.text((8, height - HUD_H + 4), f"{rec['id'][-42:]}", fill=DIM, font=small)
    d.text((8, height - HUD_H + 18), errs, fill=FG, font=small)
    return out


def encode(frames, path, fps, size, crf):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", f"{fps:.4f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    try:
        for img in frames:
            proc.stdin.write(img.tobytes())
            n += 1
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed")
    return n


def main():
    args = parse_args()
    records = json.loads((REPO_ROOT / "data" / "prepared_real" / args.set / "eval.json").read_text())
    preds = {t: load_predictions(args.eval_root, args.set, t) for t in args.tags}
    colours = {t: TAG_COLOURS[i % len(TAG_COLOURS)] for i, t in enumerate(args.tags)}
    fps_by_clip, clip_criterion = {}, {}
    if args.manifest.exists():
        for m in json.loads(args.manifest.read_text()):
            fps_by_clip[Path(m["clip"]).name] = m.get("fps", 15.0)
            clip_criterion[Path(m["clip"]).name] = m["criterion"]

    by_clip = defaultdict(list)
    for r in records:
        m = r["real_meta"]
        # a frame can sit in several clips (find_clips keeps every membership)
        for clip in m.get("clips", [m.get("clip", "unknown.mp4")]):
            crit = clip_criterion.get(clip) or (m.get("criteria") or [m.get("criterion", "other")])[0]
            if args.criterion and crit not in args.criterion:
                continue
            by_clip[(crit, clip)].append(r)

    fonts = (ImageFont.load_default(size=13), ImageFont.load_default(size=12),
             ImageFont.load_default(size=11))
    total_frames = 0
    for (criterion, clip), recs in sorted(by_clip.items()):
        recs.sort(key=lambda r: r["real_meta"]["timestamp"])
        out_dir = args.out_dir / criterion
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(clip).stem
        fps = args.fps or fps_by_clip.get(clip, 15.0)

        scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        traces, imgs, per_frame = [], [], []
        for r in recs:
            rid = r["id"]
            gt = json.loads(r["conversations"][1]["value"])
            preds_by_tag, err_by_tag, trace_text = {}, {}, None
            for t in args.tags:
                p = preds[t].get(rid)
                if p is None:
                    continue
                preds_by_tag[t] = p.get("parsed")
                err_by_tag[t] = (p.get("metrics") or {}).get("mean_point_error")
                if t == args.trace_tag:
                    trace_text, _ = reasoning_and_answer(p.get("generated", ""))
            if args.trace_tag:
                traces.append({"id": rid, "t": r["real_meta"]["timestamp"],
                               "clip": clip, "criterion": criterion,
                               "reasoning": trace_text or "",
                               "point_error": err_by_tag.get(args.trace_tag)})
            lines = trace_lines(trace_text, fonts[2], scratch) if args.trace_tag else None
            per_frame.append(({"id": rid, "image": r["image"][0], "gt": gt},
                              preds_by_tag, err_by_tag, lines))
        if not per_frame:
            continue
        probe = compose(*per_frame[0][:2], args.tags, colours, per_frame[0][3], fonts,
                        per_frame[0][2])
        tallest = max((len(f[3]) for f in per_frame if f[3]), default=0)
        height = max(probe.height, 20 + tallest * LINE_H + 8 + HUD_H)
        imgs = (compose(rec_, pb, args.tags, colours, ln, fonts, eb, height)
                for rec_, pb, eb, ln in per_frame)
        first = next(imgs)
        size = first.size
        n = encode(itertools.chain([first], imgs), out_dir / f"{stem}.mp4", fps, size, args.crf)
        if args.trace_tag:
            (out_dir / f"{stem}.traces.jsonl").write_text(
                "".join(json.dumps(t) + "\n" for t in traces))
        total_frames += n
        print(f"  {criterion:12s} {stem[:44]:44s} {n:4d} frames @ {fps:.2f} fps -> {out_dir}")
    print(f"{len(by_clip)} clips, {total_frames} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
