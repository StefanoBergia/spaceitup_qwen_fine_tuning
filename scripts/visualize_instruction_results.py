"""HTML report for the instruction-conditioned comparison. CPU only.

Reads outputs/eval_instr_v1/<set>/<tag>/{metrics,predictions}.json (written by
slurm/eval_instruction.sbatch) and writes outputs/instruction_results.html: the
model x eval-set table (pair success, follow rate, point error) and a gallery of pairs —
one frame, both instructions, both predictions.

    uv run scripts/visualize_instruction_results.py
    uv run scripts/visualize_instruction_results.py --gallery-tag instr_cosmos --gallery-set cosmos --n 12
"""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from rover_vlm.overlay import GT_GREY, draw_path, embed_jpeg

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "outputs" / "eval_instr_v1"
DATA = REPO_ROOT / "data" / "prepared_habitat_instr_v1"
OUT = REPO_ROOT / "outputs" / "instruction_results.html"
TAGS = ["instr_template", "instr_cosmos", "r4_lora", "base"]
SETS = ["template", "cosmos"]
SIDE_COLORS = {"left": (42, 120, 214), "right": (214, 74, 42)}


def collect_cells(root):
    """{(tag, set): metrics} for every finished eval cell under root/<set>/<tag>/."""
    cells = {}
    for m in Path(root).glob("*/*/metrics.json"):
        cells[(m.parent.name, m.parent.parent.name)] = json.loads(m.read_text())
    return cells


def _fmt(v):
    return "–" if v is None else f"{v:.3f}"


def table(cells):
    head = "".join(f"<th>{s} eval set</th>" for s in SETS)
    rows = []
    for tag in TAGS:
        tds = []
        for s in SETS:
            c = cells.get((tag, s))
            if c is None:
                tds.append("<td>–</td>")
                continue
            ins = c.get("instruction") or {}
            tds.append(f"<td><b>{_fmt(ins.get('pair_success'))}</b> pair · "
                       f"{_fmt(ins.get('follow_rate'))} follow · "
                       f"{_fmt(c.get('mean_point_error_median'))} err · "
                       f"{_fmt(c.get('parse_rate'))} parse</td>")
        rows.append(f"<tr><th>{tag}</th>{''.join(tds)}</tr>")
    return f"<table><tr><th>model</th>{head}</tr>{''.join(rows)}</table>"


def gallery(root, tag, set_, n):
    preds = json.loads((Path(root) / set_ / tag / "predictions.json").read_text())
    evals = {r["id"]: r for r in json.loads((DATA / f"eval_{set_}.json").read_text())}
    pairs = {}
    for r in preds:
        pairs.setdefault(r["habitat_meta"]["pair_id"], {})[r["habitat_meta"]["side"]] = r
    cards = []
    for sides in list(pairs.values())[:n]:
        if len(sides) != 2:
            continue
        img = Image.open(evals[sides["left"]["id"]]["image"][0]).convert("RGB").resize((512, 512))
        d = ImageDraw.Draw(img)
        for r in sides.values():
            draw_path(d, r["gt"]["path"], GT_GREY, 512, 512, r=0, width=3)
        lines = []
        for side, r in sides.items():
            if r.get("parsed"):
                draw_path(d, r["parsed"]["path"], SIDE_COLORS[side], 512, 512, r=4, width=3)
            instr = evals[r["id"]]["habitat_meta"]["instruction"]
            color = "#%02x%02x%02x" % SIDE_COLORS[side]
            lines.append(f'<div style="color:{color}">{side}: {html.escape(instr)}</div>')
        cards.append(f'<figure><img src="{embed_jpeg(img, 420)}"><figcaption>{"".join(lines)}</figcaption></figure>')
    return "".join(cards)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--gallery-tag", default="instr_cosmos")
    p.add_argument("--gallery-set", default="cosmos")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()
    cells = collect_cells(args.root)
    if not cells:
        raise SystemExit(f"no metrics under {args.root} — run slurm/eval_instruction.sbatch first")
    body = table(cells)
    if (args.root / args.gallery_set / args.gallery_tag / "predictions.json").exists():
        body += (f"<h2>{args.gallery_tag} on the {args.gallery_set} set</h2>"
                 "<p>grey = both routes; blue / orange = prediction for the left / right instruction</p>"
                 + gallery(args.root, args.gallery_tag, args.gallery_set, args.n))
    style = ("body{font:14px system-ui;margin:24px;background:#fafafa}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:6px 10px;text-align:left}"
             "figure{display:inline-block;margin:8px;max-width:420px}img{max-width:100%}")
    args.out.write_text(f"<title>Instruction paths</title><style>{style}</style>"
                        "<h1>Instruction-conditioned paths</h1>"
                        "<p>pair success = both instructions of a frame followed "
                        f"(an instruction-blind model scores 0)</p>{body}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
