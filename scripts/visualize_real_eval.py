"""Habitat-vs-real comparison for the path models: table + overlay gallery.

Login node (CPU-only), after slurm/run_all_real.sbatch <SET> has produced
outputs/eval_real/<SET>/<tag>/{metrics,predictions}.json:

    uv run scripts/visualize_real_eval.py                      # all sets under outputs/eval_real
    uv run scripts/visualize_real_eval.py --sets tum_pioneer    # one set

Writes outputs/eval_real/comparison.md and outputs/eval_real/real_eval.html (self-contained).
Each model's real-set numbers sit next to its own Habitat v2 eval numbers so the
render-to-real drop is read per model, not against a moving baseline. Visibility
metrics are only meaningful where the labels have depth-derived flags (TUM); for sets
without depth every label is "visible" and those columns are marked as such.
"""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from rover_vlm.eval import goal_visibility_confusion, shape_correlation, trivial_baselines
from rover_vlm.overlay import GT_GREY, draw_goal, draw_path, draw_polyline, embed_jpeg, side_by_side

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "outputs" / "eval_real"

# tag on the real sets -> (label, Habitat v2 eval metrics for the same model)
MODELS = [
    ("base_2b", "2B base", "outputs/eval_habitat_v2/habitat_base"),
    ("plain_2b", "2B plain SFT", "outputs/eval_habitat_v2/habitat_train_full"),
    ("plain_0.8b", "0.8B plain SFT", "outputs/eval_habitat_v2_0.8b/habitat_train_full"),
    ("traced_2b", "2B traced", "outputs/eval_habitat_v2_traced/habitat_train_full_traced"),
    ("traced_0.8b", "0.8B traced", "outputs/eval_habitat_v2_traced_0.8b/habitat_train_full_traced"),
]
COLORS = {"base_2b": (213, 94, 0), "plain_2b": (42, 120, 214), "plain_0.8b": (0, 158, 115),
          "traced_2b": (204, 121, 167), "traced_0.8b": (86, 180, 233)}

COLS = [
    ("parse_rate", "Parse rate", "{:.3f}"),
    ("mean_point_error_median", "Median point err", "{:.3f}"),
    ("shape_rho", "Path\u2194image \u03c1", "{:+.2f}"),
    ("frechet_median", "Median Fréchet", "{:.3f}"),
    ("goal_point_error_median", "Median goal err", "{:.3f}"),
    ("path_visibility_acc_mean", "Waypoint vis. acc", "{:.3f}"),
    ("goal_balanced", "Goal vis. (balanced)", "{:.3f}"),
]


def load_run(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "metrics.json").exists():
        return None
    m = json.loads((run_dir / "metrics.json").read_text())
    preds = json.loads((run_dir / "predictions.json").read_text())
    conf = goal_visibility_confusion(preds)
    m["goal_balanced"] = conf["balanced_accuracy"] if conf["n_visible"] and conf["n_obstructed"] else None
    rho = shape_correlation(preds)
    m["shape_rho"] = None if rho != rho else rho  # NaN -> the model gave one shape every frame
    m["_preds"] = preds
    return m


def set_has_visibility(preds):
    """True when the ground truth carries both visibility classes (depth-derived labels)."""
    flags = {v for r in preds for *_, v in r["gt"]["path"]} | {r["gt"]["goal"][2] for r in preds}
    return len(flags) > 1


def fmt(m, key, spec, vis_ok=True):
    if m is None or m.get(key) is None:
        return "–"
    if key in ("path_visibility_acc_mean", "goal_balanced") and not vis_ok:
        return "n/a"
    return spec.format(m[key])


def build_tables(eval_root, sets):
    """-> (markdown, html) with one block per set; rows = model x {Habitat, real}."""
    md, htm = [], []
    for s in sets:
        set_meta = json.loads((REPO_ROOT / "data" / "prepared_real" / s / "meta.json").read_text()) \
            if (REPO_ROOT / "data" / "prepared_real" / s / "meta.json").exists() else {}
        head = ["Model", "Eval set"] + [c[1] for c in COLS]
        rows, base_rows = [], []
        any_run = False
        for tag, label, hab_dir in MODELS:
            hab = load_run(REPO_ROOT / hab_dir)
            real = load_run(eval_root / s / tag)
            if real is None:
                continue
            any_run = True
            vis_ok = set_has_visibility(real["_preds"])
            rows.append([label, "Habitat v2 (1,000)"] + [fmt(hab, k, f) for k, _, f in COLS])
            rows.append(["", f"{s} ({real['num_samples']})"] + [fmt(real, k, f, vis_ok) for k, _, f in COLS])
        if not any_run:
            continue
        first = next(load_run(eval_root / s / t) for t, _, _ in MODELS
                     if (eval_root / s / t / "metrics.json").exists())
        base = trivial_baselines([r["gt"] for r in first["_preds"]])
        for key, label in (("straight", "straight line up the middle"),
                           ("constant", "set-mean constant path")):
            base_rows.append([label, "image-blind baseline"]
                             + [fmt(base[key], k, f) if k == "mean_point_error_median" else "–"
                                for k, _, f in COLS])
        rows = base_rows + rows
        title = f"{s}: {set_meta.get('kept', '?')} frames, HFOV {set_meta.get('hfov_deg', '?')}°, " \
                f"camera {set_meta.get('cam_height_m', {}).get('mean', float('nan')):.2f} m"
        md.append(f"### {title}\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
                  + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")
        htm.append(f"<h2>{html.escape(title)}</h2><table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head)
                   + "</tr>" + "".join("<tr class='" + ("base" if r[1].startswith("image-blind")
                                                        else "hab" if r[1].startswith("Habitat") else "real") + "'>"
                                       + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
                   + "</table>")
    return "\n".join(md), "\n".join(htm)


def render_row(image_path, gt, preds_by_tag, max_px=360):
    """GT (grey) plus one panel per model prediction, joined horizontally."""
    panels = []
    for tag, _, _ in MODELS:
        if tag not in preds_by_tag:
            continue
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        if gt.get("path"):
            draw_polyline(d, [(p[0], p[1]) for p in gt["path"]], GT_GREY, W, H, width=8)
        pred = preds_by_tag[tag]
        if pred and pred.get("path"):
            draw_path(d, pred["path"], COLORS[tag], W, H, r=5, width=3)
        if pred and pred.get("goal"):
            draw_goal(d, pred["goal"], W, H, r=7)
        d.text((6, 6), tag, fill=(255, 255, 255))
        d.text((5, 5), tag, fill=(0, 0, 0))
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        panels.append(img)
    out = panels[0]
    for p in panels[1:]:
        out = side_by_side(out, p, gap=4)
    return out


def build_gallery(eval_root, s, n=12):
    runs = {tag: load_run(eval_root / s / tag) for tag, _, _ in MODELS}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return ""
    ref_tag = "plain_2b" if "plain_2b" in runs else next(iter(runs))
    by_id = {tag: {r["id"]: r for r in run["_preds"]} for tag, run in runs.items()}
    recs = {r["id"]: r for r in json.loads((REPO_ROOT / "data" / "prepared_real" / s / "eval.json").read_text())}
    ref = [r for r in runs[ref_tag]["_preds"] if r.get("metrics")]
    ref.sort(key=lambda r: r["metrics"]["mean_point_error"])
    picks = [ref[int(i * (len(ref) - 1) / max(n - 1, 1))] for i in range(min(n, len(ref)))]
    cards = []
    for r in picks:
        rec = recs.get(r["id"])
        if rec is None:
            continue
        preds = {tag: by_id[tag].get(r["id"], {}).get("parsed") for tag in runs}
        img = render_row(rec["image"][0], r["gt"], preds)
        errs = ", ".join(f"{tag} {by_id[tag][r['id']]['metrics']['mean_point_error']:.3f}"
                         for tag in runs if by_id[tag].get(r["id"], {}).get("metrics"))
        cards.append(f"<figure><img src='{embed_jpeg(img, max_px=1800, quality=78)}'>"
                     f"<figcaption>{html.escape(r['id'])} — point err: {html.escape(errs)}</figcaption></figure>")
    return (f"<h3>{html.escape(s)}: frames ordered easy → hard for {ref_tag} "
            f"(grey = ground truth; filled marker = predicted visible, hollow = obstructed)</h3>"
            + "".join(cards))


STYLE = """
<style>
:root{--bg:#fbfaf7;--fg:#1f1f1f;--muted:#666;--line:#ddd;--hab:#f3f1ea;color-scheme:light}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px;max-width:1400px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}h3{font-size:14px;color:var(--muted);margin:18px 0 8px}
p.lead{color:var(--muted);margin:0 0 12px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin-bottom:8px}
th,td{border-bottom:1px solid var(--line);padding:5px 12px;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tr.hab td{background:var(--hab);color:var(--muted)}tr.real td{font-weight:600}\ntr.base td{background:#fdf3e7;color:#8a5a1a;font-style:italic}
figure{margin:0 0 14px}figure img{max-width:100%;display:block;border:1px solid var(--line)}
figcaption{font-size:12px;color:var(--muted);margin-top:3px}
</style>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    p.add_argument("--sets", nargs="*", default=None, help="default: every dir under --eval-root")
    p.add_argument("--gallery", type=int, default=12, help="frames per set in the gallery")
    args = p.parse_args()
    sets = args.sets or sorted(d.name for d in args.eval_root.iterdir() if d.is_dir())
    md, tables = build_tables(args.eval_root, sets)
    if not md:
        raise SystemExit(f"no metrics.json under {args.eval_root} for sets {sets}")
    galleries = "".join(build_gallery(args.eval_root, s, args.gallery) for s in sets)
    (args.eval_root / "comparison.md").write_text("# Habitat vs real-image eval\n\n" + md)
    page = ("<title>Real-image path eval</title>" + STYLE + "<h1>Habitat-trained path models on real robot frames</h1>"
            "<p class='lead'>Each model's real-set row sits under its own Habitat v2 eval row. Labels come from the "
            "robot's own future trajectory projected into the frame. Two readings guard against a flattering number. "
            "The <b>image-blind baselines</b> at the top of each table are what a model scores by ignoring the picture "
            "entirely, so a row worse than those has not transferred. <b>Path\u2194image \u03c1</b> is the rank "
            "correlation between how the predicted path bends and how the true route bends; it is positive only if the "
            "answer depends on the image, and \u2013 means the model gave the same shape on every frame. Habitat's own "
            "goal sits at x = 0.5 in 100% of its samples by construction, so its goal-error column never tested lateral "
            "placement.</p>"
            + tables + "<h2>Gallery</h2>" + galleries)
    (args.eval_root / "real_eval.html").write_text(page)
    print(md)
    print(f"-> {args.eval_root / 'comparison.md'}\n-> {args.eval_root / 'real_eval.html'}")


if __name__ == "__main__":
    main()
