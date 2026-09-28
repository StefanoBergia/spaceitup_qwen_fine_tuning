"""Build a human audit page for a random sample of v2 Cosmos instruction labels.

Draws N kept instructed records from train_general.json (fixed seed, stratified between
single-route "instructed" records and the "opposite" side of two-route pairs), renders
each frame with the labelled route in magenta (alternative in grey, as Cosmos saw it), and
writes a self-contained HTML page where each label is marked OK / wrong landmark / wrong
side / unsure. Marks are stored in the published artifact's db (collection "marks", doc
id = record id), so they can be read back with the Artifact tool's read_db.

Run (CPU, login node is fine):
    uv run scripts/audit_instruction_labels.py            # -> outputs/instr_label_audit.html
    uv run scripts/audit_instruction_labels.py --n 150 --seed 1
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
DATA = REPO_ROOT / "data" / "prepared_habitat_instr_v2" / "train_general.json"
OUT = REPO_ROOT / "outputs" / "instr_label_audit.html"
TEMPLATE = Path(__file__).with_name("audit_instruction_labels.html")
IMG_PX = 640


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


def item(rec):
    m = rec["habitat_meta"]
    img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw_reference_line(img, _answer_path(rec))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return {"id": rec["id"], "instruction": m["instruction"], "side": m["side"], "kind": m["kind"],
            "has_alt": bool(m.get("alt_path")), "scene": m["scene_id"],
            "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--opposite-share", type=float, default=0.2,
                    help="fraction drawn from the 'opposite' side of pairs (weighted back when scoring)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    records = json.loads(DATA.read_text())
    picked, pop = sample(records, args.n, args.opposite_share, args.seed)
    items = [item(r) for r in picked]
    payload = {"items": items, "population": pop, "seed": args.seed, "source": str(DATA.relative_to(REPO_ROOT))}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{len(items)} items (population {pop}) -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
