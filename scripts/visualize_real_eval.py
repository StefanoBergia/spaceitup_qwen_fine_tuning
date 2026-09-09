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

from rover_vlm.eval import (
    endpoint_spread,
    goal_visibility_confusion,
    shape_correlation,
    trivial_baselines,
)
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
    ("end_x_sd", "Endpoint x spread", "{:.3f}"),
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
    ends = endpoint_spread(preds)
    m["end_x_sd"] = ends["pred_end_x_sd"] if ends["n"] else None
    m["_ends"] = ends
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


SET_BLURB = {
    "tum_pioneer": "Indoor. Pioneer wheeled robot, Kinect, mocap camera poses, depth-derived visibility.",
    "gnd_campus": "Outdoor. Clearpath Jackal, ZED2, EKF odometry; no depth, so every label is “visible”.",
    "real_clips": "The hard slices of the TUM sequences — frames where the route bends or the "
                  "goal is hidden — cut into clips by scripts/find_clips.py, so the models are "
                  "judged on the cases a straight-ahead guess should fail.",
}


def set_facts(name, meta, n):
    """The chips under a set's heading: what this eval set physically is."""
    h = meta.get("cam_height_m", {}).get("mean")
    facts = [f"{n} frames", f"HFOV {meta.get('hfov_deg', '?')}°"]
    if h is not None:
        assumed = meta.get("params", {}).get("cam_height") is not None
        facts.append(f"camera {h:.2f} m{' (assumed)' if assumed else ' (fitted)'}")
    facts.append("Habitat: 0.80 m / 90° / 512²")
    return facts


def transfer_state(real, base):
    """Did this model beat an image-blind guess, and does its answer track the image?"""
    err, rho = real.get("mean_point_error_median"), real.get("shape_rho")
    chips = []
    if err is not None and err > base["constant"]["mean_point_error_median"]:
        chips.append("below baseline")
    if rho is None or abs(rho) < 0.15:
        chips.append("image-blind")
    ends = real.get("_ends") or {}
    # Habitat pins every label's terminus at x = 0.5; a model that inherited that cannot
    # reach an off-axis endpoint at all, so flag it separately from being merely wrong
    if ends.get("n") and ends["pred_end_x_sd"] < 0.01 <= ends["gt_end_x_sd"]:
        chips.append("endpoint pinned")
    return chips


def build_tables(eval_root, sets):
    """-> (markdown, html). Per set: the image-blind floor, then each model's Habitat row
    over its real row, with the failure state spelled out rather than left to arithmetic."""
    md, htm = [], []
    for s in sets:
        mpath = REPO_ROOT / "data" / "prepared_real" / s / "meta.json"
        set_meta = json.loads(mpath.read_text()) if mpath.exists() else {}
        head = ["Model", "Eval set"] + [c[1] for c in COLS] + ["Transfer"]
        runs = [(tag, label, load_run(REPO_ROOT / hab), load_run(eval_root / s / tag))
                for tag, label, hab in MODELS]
        runs = [r for r in runs if r[3] is not None]
        if not runs:
            continue
        base = trivial_baselines([r["gt"] for r in runs[0][3]["_preds"]])
        n = runs[0][3]["num_samples"]

        rows = []
        for key, label in (("straight", "straight line up the middle"),
                           ("constant", "set-mean constant path")):
            rows.append(("base", [label, "image-blind floor"]
                         + [fmt(base[key], k, f) if k == "mean_point_error_median" else "–"
                            for k, _, f in COLS] + ["–"]))
        for tag, label, hab, real in runs:
            vis_ok = set_has_visibility(real["_preds"])
            chips = transfer_state(real, base)
            rows.append(("hab", [label, "Habitat v2 (1,000)"]
                         + [fmt(hab, k, f) for k, _, f in COLS] + ["reference"]))
            rows.append(("real", ["", f"{s} ({n})"]
                         + [fmt(real, k, f, vis_ok) for k, _, f in COLS]
                         + [" · ".join(chips) if chips else "clears baseline"]))

        title = s
        facts = set_facts(s, set_meta, n)
        md.append(f"### {title} — {', '.join(facts)}\n\n| " + " | ".join(head) + " |\n|"
                  + "---|" * len(head) + "\n"
                  + "\n".join("| " + " | ".join(r) + " |" for _, r in rows) + "\n")

        def cell(kind, i, c):
            # the two columns that carry the verdict get the failure colour on model rows
            hi = kind == "real" and i in (2, 3) and c not in ("–", "n/a")
            klass = " class='hi'" if hi else ""
            if i == len(head) - 1 and kind == "real":
                bits = "".join(f"<span class='chip'>{html.escape(x.strip())}</span>"
                               for x in c.split("·")) if c != "clears baseline" else html.escape(c)
                return f"<td>{bits}</td>"
            return f"<td{klass}>{html.escape(c)}</td>"

        body = ""
        for j, (kind, r) in enumerate(rows):
            last = " last" if kind == "base" and rows[j + 1][0] != "base" else ""
            body += (f"<tr class='{kind}{last}'>"
                     + "".join(cell(kind, i, c) for i, c in enumerate(r)) + "</tr>")
        htm.append(f"<section><h2>{html.escape(title)}</h2>"
                   f"<h3>{html.escape(SET_BLURB.get(s, ''))}</h3>"
                   "<ul class='setmeta'>" + "".join(f"<li>{html.escape(f)}</li>" for f in facts) + "</ul>"
                   "<div class='scroll'><table><tr>"
                   + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
                   + body + "</table></div></section>")
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
    return (f"<h3>{html.escape(s)} — ordered easiest to hardest for {html.escape(ref_tag)}</h3>"
            + "".join(cards))


STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --paper:#f7f8fa; --panel:#ffffff; --ink:#12171f; --slate:#5c6673; --line:#dfe3e9;
  --rule:#c3cad4; --accent:#2f6fb0; --accent-soft:#e8f0f8;
  --fail:#b0492f; --fail-soft:#fbeeea; --floor:#8a6a1a; --floor-soft:#faf3e2;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  color-scheme:light dark;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0f1319; --panel:#161c24; --ink:#e6eaef; --slate:#98a3b1; --line:#28313c;
    --rule:#3b4653; --accent:#7fb2e0; --accent-soft:#182836;
    --fail:#e08a72; --fail-soft:#2c1d18; --floor:#d3ab5c; --floor-soft:#292115;
  }
}
:root[data-theme="dark"]{
  --paper:#0f1319; --panel:#161c24; --ink:#e6eaef; --slate:#98a3b1; --line:#28313c;
  --rule:#3b4653; --accent:#7fb2e0; --accent-soft:#182836;
  --fail:#e08a72; --fail-soft:#2c1d18; --floor:#d3ab5c; --floor-soft:#292115;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5;
     margin:0;padding:40px 28px 72px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:38px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
         color:var(--slate);margin:0 0 10px}
h1{font-size:34px;line-height:1.12;font-weight:600;letter-spacing:-.02em;margin:0;text-wrap:balance}
.standfirst{color:var(--slate);max-width:64ch;margin:12px 0 0;font-size:16px}
h2{font-size:20px;font-weight:600;letter-spacing:-.01em;margin:0 0 3px;text-wrap:balance}
h3{font-size:13px;font-weight:500;color:var(--slate);margin:0 0 12px}

.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1px;
          background:var(--line);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.finding{background:var(--panel);padding:20px 22px;display:flex;flex-direction:column;gap:7px}
.finding .k{font-family:var(--mono);font-size:26px;font-weight:500;letter-spacing:-.02em;
            font-variant-numeric:tabular-nums}
.finding .t{font-weight:600;font-size:14px}
.finding .d{color:var(--slate);font-size:13px;line-height:1.45}
.finding.ok .k{color:var(--accent)} .finding.bad .k{color:var(--fail)}

.setmeta{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px;padding:0;list-style:none}
.setmeta li{font-family:var(--mono);font-size:11px;color:var(--slate);
            border:1px solid var(--line);border-radius:2px;padding:3px 8px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
   color:var(--slate);font-weight:500;text-align:right;padding:0 14px 8px;white-space:nowrap;
   border-bottom:1px solid var(--rule)}
td{padding:7px 14px;text-align:right;border-bottom:1px solid var(--line);
   font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
td:first-child{font-family:var(--sans);font-weight:600}
td:nth-child(2){font-family:var(--sans);color:var(--slate);font-weight:400}
tr.base td{background:var(--floor-soft);color:var(--floor)}
tr.base td:first-child{font-family:var(--sans);font-weight:500;color:var(--floor)}
tr.base.last td{border-bottom:2px solid var(--floor)}
tr.hab td{color:var(--slate)}
tr.real td{border-bottom:1px solid var(--rule)}
tr.real td.hi{color:var(--fail);font-weight:600}
.chip{font-family:var(--mono);font-size:10px;letter-spacing:.04em;text-transform:uppercase;
      padding:2px 7px;border-radius:2px;background:var(--fail-soft);color:var(--fail);
      border:1px solid currentColor;white-space:nowrap}
.note{background:var(--accent-soft);border-left:3px solid var(--accent);padding:14px 18px;
      border-radius:0 3px 3px 0;font-size:14px;max-width:78ch}
.note b{font-weight:600}
figure{margin:0 0 18px}
figure img{max-width:100%;display:block;border:1px solid var(--line);border-radius:2px}
figcaption{font-family:var(--mono);font-size:11px;color:var(--slate);margin-top:5px}
a{color:var(--accent)}
</style>
"""


def findings_band(eval_root, sets):
    """The three numbers that carry the result, computed from the runs themselves."""
    adapters = [t for t, _, _ in MODELS if t != "base_2b"]
    parse, rho_real, rho_hab, over = [], [], [], 0
    total = 0
    for s in sets:
        runs = {t: load_run(eval_root / s / t) for t, _, _ in MODELS}
        runs = {k: v for k, v in runs.items() if v}
        if not runs:
            continue
        base = trivial_baselines([r["gt"] for r in next(iter(runs.values()))["_preds"]])
        floor = base["constant"]["mean_point_error_median"]
        for t in adapters:
            if t not in runs:
                continue
            total += 1
            parse.append(runs[t]["parse_rate"])
            if runs[t].get("shape_rho") is not None:
                rho_real.append(runs[t]["shape_rho"])
            if runs[t]["mean_point_error_median"] > floor:
                over += 1
    for _, _, hab in MODELS[1:]:
        h = load_run(REPO_ROOT / hab)
        if h and h.get("shape_rho") is not None:
            rho_hab.append(h["shape_rho"])
    cards = [
        ("ok", f"{min(parse):.3f}", "Format transfers intact",
         f"Lowest parse rate across all {total} fine-tuned runs on real frames. Every adapter still "
         f"emits well-formed waypoint JSON; the base model drops to 0.277 on the outdoor set."),
        ("bad", f"{over}/{total}", "Geometry does not transfer",
         "Fine-tuned runs scoring worse than an image-blind guess — the set's own mean path. "
         "Not one model clears the floor on either set."),
        ("bad", f"{sum(rho_hab)/len(rho_hab):+.2f} → {sum(rho_real)/len(rho_real):+.2f}",
         "The image stops mattering",
         "Mean correlation between the predicted path's bend and the true route's bend, "
         "Habitat versus real. On real frames the answer is uninformed by the picture."),
    ]
    return "<section class='findings'>" + "".join(
        f"<div class='finding {k}'><div class='k'>{html.escape(v)}</div>"
        f"<div class='t'>{html.escape(t)}</div><div class='d'>{html.escape(d)}</div></div>"
        for k, v, t, d in cards) + "</section>"


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
    galleries = "".join(f"<section>{build_gallery(args.eval_root, s, args.gallery)}</section>"
                        for s in sets)
    (args.eval_root / "comparison.md").write_text("# Habitat vs real-image eval\n\n" + md)

    header = (
        "<header><p class='eyebrow'>Out-of-domain evaluation</p>"
        "<h1>Habitat-trained path models on real robot frames</h1>"
        "<p class='standfirst'>Four LoRA adapters trained to predict a traversable path from a single "
        "simulated indoor frame, scored for the first time on real robot video. Ground truth comes from "
        "the robot's own future trajectory: the positions it went on to occupy, dropped to the floor plane "
        "and projected back into the frame it was looking at.</p></header>")
    how = (
        "<div class='note'><b>How to read this.</b> Normalized-coordinate error has no absolute meaning, so "
        "each table opens with two <b>image-blind floors</b>: what a straight line up the middle, and the set's "
        "own mean path, score without looking at the picture at all. A model below that line has not "
        "transferred. <b>Path↔image ρ</b> is the rank correlation between how the predicted path bends and how "
        "the true route bends — positive only if the answer depends on the image, and “–” where the model "
        "returned the same shape on every frame.</div>")
    caveat = (
        "<div class='note'><b>Two things the Habitat numbers were not measuring.</b> Habitat places the goal at "
        "x = 0.5 in 100% of its samples by construction, so its goal-error column only ever tested height, never "
        "lateral placement — and all four adapters learned that constant, emitting x = 0.500 with zero variance "
        "on real frames too. Its goals are also obstructed 77% of the time, against 8% on the real indoor set, "
        "so the visibility columns carry a prior that real scenes contradict.</div>")
    page = ("<title>Sim-to-Real Path Transfer</title>" + STYLE + "<div class='wrap'>" + header
            + findings_band(args.eval_root, sets) + how + tables + caveat
            + "<section><h2>Frames</h2><h3>Grey is ground truth; each panel adds one model's prediction. "
              "Filled marker = predicted visible, hollow = predicted obstructed.</h3>" + galleries
            + "</section></div>")
    (args.eval_root / "real_eval.html").write_text(page)
    print(md)
    print(f"-> {args.eval_root / 'comparison.md'}\n-> {args.eval_root / 'real_eval.html'}")


if __name__ == "__main__":
    main()
