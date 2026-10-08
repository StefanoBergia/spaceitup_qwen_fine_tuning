"""Steering eval report: stop-pair success per text x prompt form x model, plus a gallery.

CPU only, login node, after `SETS=steer sbatch slurm/eval_descriptions.sbatch` (and the
round-4 control `SETS=steer_goal ... TAG=r4`). Metrics are recomputed from predictions.json
with rover_vlm.eval.aggregate_steering_metrics, so a metric fix never needs a GPU re-run.

    uv run scripts/visualize_steering.py                   # tags d1 and r4
    uv run scripts/visualize_steering.py --tags d1 --n 16

Writes outputs/steering_report.html (self-contained; images inline). Spec:
docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from rover_vlm.eval import aggregate_steering_metrics
from rover_vlm.overlay import description_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
CELLS = [("stop", t, f) for t in ("template", "composed") for f in ("entry", "goal")]
TEXT_H = 70


def metric_cells(eval_dir, tags):
    """One row per (slice, text, form, tag) with predictions on disk."""
    out = []
    for s, t, f in CELLS:
        for tag in tags:
            p = Path(eval_dir) / f"eval_steer_{s}_{t}_{f}" / tag / "predictions.json"
            if not p.exists():
                continue
            recs = json.loads(p.read_text())
            out.append({"slice": s, "text": t, "form": f, "tag": tag, "n": len(recs),
                        "parse_rate": sum(bool((r.get("parsed") or {}).get("path")) for r in recs) / max(1, len(recs)),
                        "steering": aggregate_steering_metrics(recs) or {}})
    return out


def pair_panel(image_path, a, b, px=360):
    """Both routes of a pair side by side: GT grey, prediction orange, each text underneath."""
    return description_panel(image_path, [a, b], px, TEXT_H, 4)


def _uri(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def gallery(eval_dir, data_dir, set_name, tag, n):
    p = Path(eval_dir) / set_name / tag / "predictions.json"
    if not p.exists():
        return []
    recs = json.loads(p.read_text())
    images = {r["id"]: r["image"][0] for r in json.loads((Path(data_dir) / f"{set_name}.json").read_text())}
    by_id, seen, out = {r["id"]: r for r in recs}, set(), []
    for r in recs:
        sid = r["habitat_meta"].get("sibling_id")
        if r["id"] in seen or sid not in by_id:
            continue
        a, b = sorted((r, by_id[sid]), key=lambda x: x["id"])
        seen |= {a["id"], b["id"]}
        out.append((a["id"].rsplit("_", 1)[0], _uri(pair_panel(images[a["id"]], a, b))))
        if len(out) >= n:
            break
    return out


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Steering Eval Results</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddddd6;--card:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ececea;--muted:#a3a39c;--line:#33332f;--card:#1f1f1d}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ececea;--muted:#a3a39c;--line:#33332f;--card:#1f1f1d}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}
th:nth-child(-n+4),td:nth-child(-n+4){text-align:left}.muted{color:var(--muted)}
.wrap{overflow-x:auto}figure{margin:16px 0}img{max-width:100%;height:auto;border-radius:6px}
</style></head><body>
<h1>Steering eval</h1>
<p class="muted">Pair success = both texts of a pair steered their own way: each prediction ends closer to its own
stop than to the other text's stop. In the entry form a text-blind model scores 0; goal-form rows hand each text its own stop, so they measure
goal-following (a ceiling), not steering. Swap = both predictions followed the
<em>other</em> text. Strict = every event and the stop (0.03 / 0.05).</p>
<div class="wrap"><table><thead><tr><th>slice</th><th>text</th><th>form</th><th>model</th><th>pairs</th>
<th>pair success [95% CI]</th><th>pure stop pairs</th><th>swap</th><th>closer to own GT</th><th>strict .03</th><th>strict .05</th><th>parse</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
__GALLERY__
</body></html>"""


def render(cells, galleries):
    rows = []
    for c in cells:
        s = c["steering"]
        ci = s.get("pair_success_ci")
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            c["slice"], c["text"], c["form"], c["tag"], s.get("n_pairs", 0),
            f"{_pct(s.get('pair_success'))} [{_pct(ci[0]) if ci else '—'}, {_pct(ci[1]) if ci else '—'}]",
            f"{_pct(s.get('pair_success_prefix_events'))} (n={s.get('n_prefix_events', 0)})",
            _pct(s.get("swap_pair_rate")), _pct(s.get("closer_to_own_gt")),
            _pct(s.get("strict_pair_success_03")), _pct(s.get("strict_pair_success_05")), _pct(c["parse_rate"]))) + "</tr>")
    gal = "".join(f"<h2>{html.escape(title)}</h2>" + "".join(
        f'<figure><img src="{uri}" alt="{html.escape(pid)}"><figcaption class="muted">{html.escape(pid)}</figcaption></figure>'
        for pid, uri in items) for title, items in galleries)
    return PAGE.replace("__ROWS__", "".join(rows)).replace("__GALLERY__", gal)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=["d1", "r4"])
    p.add_argument("--n", type=int, default=16, help="gallery pairs")
    p.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_desc")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v3")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "steering_report.html")
    args = p.parse_args()
    cells = metric_cells(args.eval_dir, args.tags)
    if not cells:
        raise SystemExit(f"no eval_steer_* predictions under {args.eval_dir}")
    galleries = [("stop pairs — d1, template text, entry form",
                  gallery(args.eval_dir, args.data_dir, "eval_steer_stop_template_entry", "d1", args.n))]
    args.out.write_text(render(cells, galleries))
    for c in cells:
        print(f"{c['slice']:5s} {c['text']:9s} {c['form']:5s} {c['tag']:3s} pairs {c['steering'].get('n_pairs', 0):4d} "
              f"success {_pct(c['steering'].get('pair_success'))}")
    print(f"-> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
