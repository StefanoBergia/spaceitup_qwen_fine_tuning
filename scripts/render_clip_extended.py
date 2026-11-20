"""One benchmark clip, extended past its labelled span with the raw footage.

The benchmark clips end where the labels end: once the robot pivots, no point of its upcoming
route is inside the camera's field of view, so there is no goal to hand the model and no
honest query to draw. This renders a longer video anyway -- predictions side by side while a
goal is in view (same layout as render_clip_predictions.py), then the dimmed raw frames with a
"goal out of view: no query" strip until the robot is visibly past the obstacle. CPU only.

    uv run scripts/render_clip_extended.py --sequence rgbd_dataset_freiburg2_pioneer_slam \
        --start 589 --end 780 --set real_clips_endpoints --tags plain_2b plain_2b_r4_rtx \
        --out outputs/clips/extended/pioneer_slam_00589-00780.mp4
"""
import argparse, importlib.util, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
spec = importlib.util.spec_from_file_location("rcp", REPO_ROOT / "scripts" / "render_clip_predictions.py")
rcp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rcp)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--sequence", required=True)
p.add_argument("--start", type=int, required=True, help="first video frame index (index.jsonl 'v')")
p.add_argument("--end", type=int, required=True, help="last video frame index, inclusive")
p.add_argument("--set", default="real_clips_endpoints")
p.add_argument("--tags", nargs="+", default=["plain_2b", "plain_2b_r4_rtx"])
p.add_argument("--labels", default="plain_2b_r4_rtx", help="tag whose eval dir holds the *_endpoints predictions (for GT lookup)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--fps", type=float, default=None)
p.add_argument("--no-gt", action="store_true", help="do not draw the ground-truth route")
p.add_argument("--demo-set", default=None, help="prepared_real set with hand-placed goals for the frames the labeller rejected")
p.add_argument("--demo-tag", default=None, help="tag under outputs/eval_real/<demo-set>/ holding its predictions")
p.add_argument("--crf", type=int, default=24)
args = p.parse_args()

index = [json.loads(l) for l in open(REPO_ROOT / "outputs" / "videos" / f"{args.sequence}.index.jsonl")]
fps = args.fps or 47031 / 2500  # the label video's rate for this sequence family
recs = {r["id"]: r for r in json.load(open(REPO_ROOT / "data" / "prepared_real" / args.set / "eval.json"))}
preds = {t: rcp.load_predictions(REPO_ROOT / "outputs" / "eval_real", args.set, t) for t in args.tags}
demo_recs, demo_preds = {}, {}
if args.demo_set:
    demo_recs = {r["id"]: r for r in json.load(open(REPO_ROOT / "data" / "prepared_real" / args.demo_set / "eval.json"))}
    demo_preds = rcp.load_predictions(REPO_ROOT / "outputs" / "eval_real", args.demo_set, args.demo_tag or args.tags[-1])
colours = {t: rcp.TAG_COLOURS[i % len(rcp.TAG_COLOURS)] for i, t in enumerate(args.tags)}
fonts = (ImageFont.load_default(size=13), ImageFont.load_default(size=12), ImageFont.load_default(size=11))

frames_meta = [r for r in index if args.start <= r["v"] <= args.end]
# probe size from the first labelled frame
first_lab = next(r for r in frames_meta if r.get("id") and r["id"] in recs)
def labelled_frame(r, demo=False):
    rec = (demo_recs if demo else recs)[r["id"]]
    gt = {} if (args.no_gt or demo) else json.loads(rec["conversations"][1]["value"])
    pb, eb = {}, {}
    for t in args.tags:
        q = demo_preds.get(r["id"]) if demo else preds[t].get(r["id"])
        if q is None: continue
        pb[t] = q.get("parsed"); eb[t] = None if demo else (q.get("metrics") or {}).get("mean_point_error")
    img = rcp.compose({"id": r["id"], "image": rec["image"][0], "gt": gt}, pb, args.tags, colours, None, fonts, eb)
    if demo:
        d = ImageDraw.Draw(img)
        d.rectangle([0, img.height - rcp.HUD_H + 16, img.width, img.height], fill=rcp.BG)
        d.text((8, img.height - rcp.HUD_H + 18), "hand-placed goal on the open floor (no ground truth for this frame)", fill=(240, 170, 120), font=fonts[1])
    return img
probe = labelled_frame(first_lab); size = probe.size

def raw_frame(r):
    img = Image.open(r["image"]).convert("RGB")
    img.thumbnail((rcp.PANEL_PX, rcp.PANEL_PX), Image.LANCZOS)
    dim = Image.eval(img, lambda x: int(x * 0.55))
    out = Image.new("RGB", size, rcp.BG)
    x = 0
    for _ in args.tags:
        out.paste(dim, (x, 0)); x += dim.width + 6
    d = ImageDraw.Draw(out)
    d.text((8, size[1] - rcp.HUD_H + 4), f"{args.sequence[-30:]} frame {r['v']}", fill=rcp.DIM, font=fonts[1])
    d.text((8, size[1] - rcp.HUD_H + 18), "goal out of view: robot turning, no route point inside the camera frame -> no query", fill=(240, 170, 120), font=fonts[1])
    return out

def gen():
    n_lab = n_raw = 0
    for r in frames_meta:
        rid = r.get("id") or f"{args.sequence}_{Path(r['image']).stem}"
        if rid in recs and rid in preds[args.tags[0]]:
            n_lab += 1; yield labelled_frame({**r, "id": rid})
        elif rid in demo_recs and rid in demo_preds:
            n_lab += 1; yield labelled_frame({**r, "id": rid}, demo=True)
        else:
            n_raw += 1; yield raw_frame(r)
    print(f"labelled {n_lab}, raw {n_raw}")

args.out.parent.mkdir(parents=True, exist_ok=True)
n = rcp.encode(gen(), args.out, fps, size, args.crf)
print(f"{n} frames @ {fps:.2f} fps -> {args.out}  ({n / fps:.1f} s)")
