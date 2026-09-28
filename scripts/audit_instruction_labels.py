"""Build a human audit page for v2 instruction labels, or for their grounded-landmark targets.

Draws N kept instructed records from train_general.json (fixed seed, stratified between
single-route "instructed" records and the "opposite" side of two-route pairs), renders
each frame with the labelled route in magenta (alternative in grey, as Cosmos saw it), and
writes a self-contained HTML page where each label is marked OK / wrong landmark / wrong
side / unsure. Marks are stored in the published artifact's db (collection "marks", doc
id = record id), so they can be read back with the Artifact tool's read_db.

--mode geometry (2026-09-28): the page shows each labelset record's route, its straight
entry->goal line and the target rover_vlm.landmarks chose (cyan outline), and asks whether
that target is right. 30 pair records + 70 single-route records from labelset_train.json.

Run (CPU, login node is fine):
    uv run scripts/audit_instruction_labels.py            # -> outputs/instr_label_audit.html
    uv run scripts/audit_instruction_labels.py --n 150 --seed 1
    uv run scripts/audit_instruction_labels.py --mode geometry \
        --landmarks data/prepared_habitat_instr_v2/landmarks_train.jsonl   # -> outputs/instr_geometry_audit.html
"""

import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from rover_vlm.instructions import _answer_path, render_generation_jpeg

REPO_ROOT = Path(__file__).resolve().parent.parent
V2 = REPO_ROOT / "data" / "prepared_habitat_instr_v2"
DATA = V2 / "train_general.json"
OUT = REPO_ROOT / "outputs" / "instr_label_audit.html"
TEMPLATE = Path(__file__).with_name("audit_instruction_labels.html")
IMG_PX = 640
TARGET_OUTLINE = (0, 230, 255)
VERDICTS = {
    "labels": [
        {"v": "ok", "label": "Correct", "help": "Landmark is there and the route goes that side of it"},
        {"v": "landmark", "label": "Wrong landmark", "help": "Object not in the frame, misnamed, or not what the route bends around"},
        {"v": "side", "label": "Wrong side", "help": "Route goes the other way round the named object"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous to judge from this frame"},
    ],
    "geometry": [
        {"v": "ok", "label": "Right target", "help": "The outline is the opening the route uses / the object it goes around"},
        {"v": "wrong", "label": "Wrong target", "help": "The outline is on something else"},
        {"v": "none_ok", "label": "No target, correctly", "help": "Nothing nameable; dropping this record is right"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
}


def sample(records, n, opposite_share, seed):
    by_kind = {"instructed": [], "opposite": []}
    for r in records:
        kind = r["habitat_meta"].get("kind")
        if kind in by_kind and r["habitat_meta"].get("instruction"):
            by_kind[kind].append(r)
    rng = random.Random(seed)
    n_opp = round(n * opposite_share)
    picked = rng.sample(by_kind["opposite"], n_opp) + rng.sample(by_kind["instructed"], n - n_opp)
    rng.shuffle(picked)
    pop = {k: len(v) for k, v in by_kind.items()}
    return picked, pop


def draw_reference_line(img, path, dash=12):
    """Dashed straight line entry -> goal: the side label says which way the route bows from it."""
    (x0, y0, _), (x1, y1, _) = path[0], path[-1]
    W, H = img.size
    a, b = (x0 * W, y0 * H), (x1 * W, y1 * H)
    n = max(1, int(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 // dash))
    draw = ImageDraw.Draw(img)
    for i in range(0, n, 2):
        p = [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in (i, min(i + 1, n))]
        draw.line(p, fill=(0, 0, 0), width=5)
        draw.line(p, fill=(255, 255, 255), width=3)
    for (x, y) in (a, b):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 255, 255), outline=(0, 0, 0), width=2)


def sample_geometry(records, landmarks, n_pair, n_single, seed):
    """Labelset records that have a landmark row: n_pair from pairs (both kinds), n_single others."""
    have = [r for r in records if r["id"] in landmarks]
    pairs = [r for r in have if r["habitat_meta"].get("pair_id")]
    single = [r for r in have if not r["habitat_meta"].get("pair_id")]
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n_pair, len(pairs))) + rng.sample(single, min(n_single, len(single)))
    rng.shuffle(picked)
    return picked, {"pair": len(pairs), "single": len(single)}


def item(rec, landmark=None):
    m = rec["habitat_meta"]
    img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw_reference_line(img, _answer_path(rec))
    if landmark and landmark.get("polygon"):
        ImageDraw.Draw(img).polygon([(u * IMG_PX, v * IMG_PX) for u, v in landmark["polygon"]],
                                    outline=TARGET_OUTLINE, width=4)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    out = {"id": rec["id"], "instruction": m.get("instruction"), "side": m["side"], "kind": m["kind"],
           "has_alt": bool(m.get("alt_path")), "scene": m["scene_id"],
           "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
    if landmark:
        out.update(target=landmark["target"], reason=landmark["reason"])
    return out


def load_landmarks_rows(paths):
    rows = {}
    for p in paths or []:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--opposite-share", type=float, default=0.2,
                    help="fraction drawn from the 'opposite' side of pairs (weighted back when scoring)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["labels", "geometry"], default="labels")
    ap.add_argument("--landmarks", type=Path, nargs="+", default=None,
                    help="landmarks_*.jsonl: draw each record's target (required for --mode geometry)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    landmarks = load_landmarks_rows(args.landmarks)
    if args.mode == "geometry":
        if not landmarks:
            raise SystemExit("--mode geometry needs --landmarks data/prepared_habitat_instr_v2/landmarks_train.jsonl")
        source = V2 / "labelset_train.json"
        picked, pop = sample_geometry(json.loads(source.read_text()), landmarks, 30, args.n - 30, args.seed)
        out = args.out or REPO_ROOT / "outputs" / "instr_geometry_audit.html"
    else:
        source = DATA
        picked, pop = sample(json.loads(source.read_text()), args.n, args.opposite_share, args.seed)
        out = args.out or OUT
    items = [item(r, landmarks.get(r["id"])) for r in picked]
    payload = {"items": items, "population": pop, "seed": args.seed, "mode": args.mode,
               "verdicts": VERDICTS[args.mode], "source": str(source.relative_to(REPO_ROOT))}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
