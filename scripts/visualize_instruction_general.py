"""Results page for the general instruction-conditioned model (v2). CPU only.

Reads outputs/eval_instr_v2/<set>/<tag>/{metrics,predictions}.json (written by
slurm/eval_instruction_v2.sbatch), the round-4 reference metrics and the v2 dataset meta,
and writes one self-contained page (outputs/instruction_general.html): the verdict against
the spec's pass criteria, pair success by frame source with viewpoint-clustered intervals,
the blocked-side and hand-written probes, a gallery with waypoint occlusion markers, and
every cell's numbers. Drawing and page style are shared with the v1 page
(scripts/visualize_instruction_results.py).

    uv run scripts/visualize_instruction_general.py
"""

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from rover_vlm.eval import bend_side, clustered_bootstrap_ci  # noqa: E402
from rover_vlm.overlay import draw_path, embed_jpeg  # noqa: E402

_spec = importlib.util.spec_from_file_location("vir", REPO_ROOT / "scripts" / "visualize_instruction_results.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

ROOT = REPO_ROOT / "outputs" / "eval_instr_v2"
DATA = REPO_ROOT / "data" / "prepared_habitat_instr_v2"
R4_METRICS = REPO_ROOT / "outputs" / "eval_habitat_v4" / "habitat_train_full" / "metrics.json"
OUT = REPO_ROOT / "outputs" / "instruction_general.html"
SET_NAMES = {"eval_plain": "plain prompt, all eval frames", "eval_instructed": "instruction for the correct route",
             "eval_pairs": "two-route pairs", "probe_wrong_side": "told the blocked side",
             "probe_handwritten": "hand-written instructions"}
TAG_NAMES = {"general": "General model", "v1_cosmos": "v1 Cosmos model"}
# spec pass criteria (2026-09-25 design, section 4)
MAX_PLAIN_ERR, MIN_PLAIN_WIN, MIN_PAIR = 0.013, 0.69, 0.75


def cells_table(root):
    """One row per finished eval cell under root/<set>/<tag>/metrics.json."""
    rows = []
    for m in sorted(Path(root).glob("*/*/metrics.json")):
        d = json.loads(m.read_text())
        ins, ws = d.get("instruction") or {}, d.get("wrong_side") or {}
        rows.append({"tag": m.parent.name, "set": m.parent.parent.name,
                     "parse_rate": d.get("parse_rate"), "median_err": d.get("mean_point_error_median"),
                     "win": d.get("win_vs_to_goal"), "pair_success": ins.get("pair_success"),
                     "follow_rate": ins.get("follow_rate"), "obeyed": ws.get("obeyed"),
                     "kept_feasible": ws.get("kept_feasible")})
    return rows


def pair_vectors(root, tag, set_):
    """(success per pair, viewpoint per pair, source per pair, pair outcomes) for one cell."""
    p = Path(root) / set_ / tag / "predictions.json"
    if not p.exists():
        return None
    pairs = {k: s for k, s in v1.pair_outcomes(json.loads(p.read_text())).items() if len(s) == 2}
    keys = sorted(pairs)
    success = np.array([all(ok for _, ok in pairs[k].values()) for k in keys], float)
    view = [pairs[k]["left"][0]["habitat_meta"].get("parent_sample_id") or k for k in keys]
    source = [pairs[k]["left"][0]["habitat_meta"].get("source") or "unknown" for k in keys]
    return {"success": success, "view": view, "source": source, "pairs": pairs, "keys": keys}


def fmt(v, pct=True, digits=1):
    if v is None:
        return "–"
    return f"{100 * v:.{digits}f}%" if pct else f"{v:.3f}"


def dot_chart(rows):
    """Horizontal dot plot of pair success with intervals. rows: label, sub, color, v, lo, hi, n."""
    W, left, right, row_h, top = 720, 260, 28, 44, 30
    H = top + row_h * len(rows) + 30
    x = lambda v: left + v * (W - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Pair success with 95% intervals" class="chart">']
    for t in range(0, 101, 20):
        xv = x(t / 100)
        out.append(f'<line x1="{xv:.1f}" x2="{xv:.1f}" y1="{top - 8}" y2="{H - 26}" class="grid"/>'
                   f'<text x="{xv:.1f}" y="{H - 8}" class="tick" text-anchor="middle">{t}%</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + row_h / 2
        tip = html.escape(f'{r["label"]} · {r["sub"]}: {fmt(r["v"])} (95% CI {fmt(r["lo"])}–{fmt(r["hi"])}), {r["n"]} pairs')
        out.append(f'<text x="0" y="{y - 3:.1f}" class="row-label">{html.escape(r["label"])}</text>'
                   f'<text x="0" y="{y + 13:.1f}" class="row-sub">{html.escape(r["sub"])} · {r["n"]} pairs</text>'
                   f'<g class="mark" tabindex="0" data-tip="{tip}">'
                   f'<rect x="{left - 6}" y="{y - row_h / 2 + 2:.1f}" width="{W - left - right + 12}" height="{row_h - 4}" class="hit"/>')
        if r["hi"] > r["lo"]:
            out.append(f'<line x1="{x(r["lo"]):.1f}" x2="{x(r["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                       f'stroke="var({r["color"]})" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(r["v"]):.1f}" cy="{y:.1f}" r="6" fill="var({r["color"]})" class="dot"/>')
        lx, anchor = (x(r["lo"]) - 10, "end") if r["v"] >= 0.5 else (x(r["hi"]) + 10, "start")
        out.append(f'<text x="{lx:.1f}" y="{y + 4:.1f}" class="value" text-anchor="{anchor}">{fmt(r["v"])}</text></g>')
    out.append("</svg>")
    return "".join(out)


def pair_cards(root, tag, set_, keys, data_file, label):
    """Gallery cards for chosen pairs, via the v1 figure (waypoint occlusion markers)."""
    vec = pair_vectors(root, tag, set_)
    if vec is None:
        return ""
    recs = {r["id"]: r for r in json.loads((DATA / data_file).read_text())}
    cards = []
    for k in keys:
        sides = vec["pairs"][k]
        src = recs[sides["left"][0]["id"]]["image"][0]
        lines = []
        for side, cls in (("left", "k-left"), ("right", "k-right")):
            r, ok = sides[side]
            vis = (r.get("metrics") or {}).get("path_visibility_acc")
            vis_txt = f" · visibility {100 * vis:.0f}% match" if vis is not None else ""
            lines.append(f'<li class="{cls}"><span class="swatch" aria-hidden="true"></span>'
                         f'<span class="instr">“{html.escape(recs[r["id"]]["habitat_meta"]["instruction"])}”</span>'
                         f'<span class="outcome {"ok" if ok else "miss"}">{"followed" if ok else "not followed"}{vis_txt}</span></li>')
        both = all(ok for _, ok in sides.values())
        cards.append(f'<figure class="pair"><img src="{v1.pair_figure(src, sides)}" alt="{html.escape(label)} frame {html.escape(k)}" '
                     f'loading="lazy" width="460" height="460"><figcaption><div class="pair-head"><span class="pid">'
                     f'{html.escape(label)} · {html.escape(k)}</span><span class="status {"ok" if both else "miss"}">'
                     f'{"both followed" if both else "pair failed"}</span></div><ul class="instr-list">{"".join(lines)}</ul>'
                     f'</figcaption></figure>')
    return "".join(cards)


def wrong_side_card(root):
    p = Path(root) / "probe_wrong_side" / "general" / "predictions.json"
    if not p.exists():
        return ""
    preds = json.loads(p.read_text())
    recs = {r["id"]: r for r in json.loads((DATA / "probe_wrong_side.json").read_text())}
    for r in preds:                      # first one whose prediction obeyed, to show what that looks like
        if r.get("parsed") and bend_side(r["parsed"]["path"]) == r["habitat_meta"]["side"]:
            break
    else:
        r = preds[0]
    img = Image.open(recs[r["id"]]["image"][0]).convert("RGB").resize((460, 460), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    draw_path(d, r["gt"]["path"], v1.GT_RGB, 460, 460, r=4, width=3)
    if r.get("parsed"):
        draw_path(d, r["parsed"]["path"], v1.RIGHT_RGB if r["habitat_meta"]["side"] == "right" else v1.LEFT_RGB,
                  460, 460, r=7, width=3)
    got = bend_side(r["parsed"]["path"]) if r.get("parsed") else "unparsed"
    hm = r["habitat_meta"]
    return (f'<figure class="pair blind"><img src="{embed_jpeg(img, 460, 85)}" alt="Blocked-side probe frame {html.escape(r["id"])}" '
            f'loading="lazy" width="460" height="460"><figcaption><div class="pair-head"><span class="pid">blocked-side probe · '
            f'{html.escape(r["id"])}</span><span class="status neutral">bends {got}</span></div>'
            f'<ul class="instr-list"><li class="{"k-right" if hm["side"] == "right" else "k-left"}"><span class="swatch" aria-hidden="true"></span>'
            f'<span class="instr">“{html.escape(recs[r["id"]]["habitat_meta"]["instruction"])}” — the only feasible route goes '
            f'{html.escape(hm["feasible_side"])} (grey)</span></li></ul></figcaption></figure>')


def build_page(root):
    cells = {(r["tag"], r["set"]): r for r in cells_table(root)}
    r4 = json.loads(R4_METRICS.read_text()) if R4_METRICS.exists() else {}
    plain, instr = cells.get(("general", "eval_plain"), {}), cells.get(("general", "eval_instructed"), {})
    gen_v, v1_v = pair_vectors(root, "general", "eval_pairs"), pair_vectors(root, "v1_cosmos", "eval_pairs")
    hw_v = pair_vectors(root, "general", "probe_handwritten")
    ws = cells.get(("general", "probe_wrong_side"), {})

    def ci(vec, mask=None):
        s = vec["success"] if mask is None else vec["success"][mask]
        g = np.array(vec["view"], dtype=object) if mask is None else np.array(vec["view"], dtype=object)[mask]
        return (float(s.mean()), *clustered_bootstrap_ci(s, list(g))) if len(s) else (None, 0.0, 0.0)

    chart_rows, verdict = [], []
    for name, vec, color in (("General model", gen_v, "--arm-cosmos"), ("v1 Cosmos model", v1_v, "--arm-template")):
        if vec is None:
            continue
        src = np.array(vec["source"])
        for sub, mask in (("all eval pairs", None), ("original frames", src == "original"),
                          ("yaw-augmented frames", src == "augmented")):
            v, lo, hi = ci(vec, mask)
            if v is not None:
                chart_rows.append({"label": name, "sub": sub, "color": color, "v": v, "lo": lo, "hi": hi,
                                   "n": int(len(vec["success"]) if mask is None else mask.sum())})
    if hw_v is not None:
        v, lo, hi = ci(hw_v)
        chart_rows.append({"label": "General model", "sub": "hand-written instructions", "color": "--ref",
                           "v": v, "lo": lo, "hi": hi, "n": len(hw_v["success"])})

    # pass / fail against the spec
    diff_txt, pair_ok = "–", None
    if gen_v is not None and v1_v is not None:
        common = sorted(set(gen_v["keys"]) & set(v1_v["keys"]))
        gi, vi = {k: i for i, k in enumerate(gen_v["keys"])}, {k: i for i, k in enumerate(v1_v["keys"])}
        d = np.array([gen_v["success"][gi[k]] - v1_v["success"][vi[k]] for k in common])
        lo, hi = clustered_bootstrap_ci(d, [gen_v["view"][gi[k]] for k in common])
        diff_txt = f"{100 * d.mean():+.1f} pts (95% CI {100 * lo:+.0f} to {100 * hi:+.0f})"
        pair_ok = gen_v["success"].mean() >= MIN_PAIR and hi >= 0
    checks = [
        ("Plain prompt still does round 4's job",
         f'median error {fmt(plain.get("median_err"), False)} (round 4 {fmt(r4.get("mean_point_error_median"), False)}), '
         f'beats to_goal {fmt(plain.get("win"))} (round 4 {fmt(r4.get("win_vs_to_goal"))})',
         f"≤ {MAX_PLAIN_ERR} and ≥ {100 * MIN_PLAIN_WIN:.0f}%",
         None if not plain else (plain["median_err"] <= MAX_PLAIN_ERR and plain["win"] >= MIN_PLAIN_WIN)),
        ("Instruction on ordinary frames does not hurt",
         f'median error {fmt(instr.get("median_err"), False)}, beats to_goal {fmt(instr.get("win"))}',
         "not worse than the plain prompt (±0.002 error, −2 pts win)",
         None if not (plain and instr) else (instr["median_err"] <= plain["median_err"] + 0.002
                                             and instr["win"] >= plain["win"] - 0.02)),
        ("Still steerable on two-route frames",
         f'pair success {fmt(gen_v["success"].mean()) if gen_v else "–"}; vs v1 {diff_txt}',
         f"≥ {100 * MIN_PAIR:.0f}% and not significantly below v1", pair_ok),
    ]
    for name, got, need, ok in checks:
        state = "pending" if ok is None else ("pass" if ok else "fail")
        verdict.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(got)}</td><td>{html.escape(need)}</td>'
                       f'<td><span class="status {"ok" if state == "pass" else "miss"}">{state}</span></td></tr>')

    table = "".join(
        f'<tr><td>{TAG_NAMES.get(r["tag"], r["tag"])}</td><td>{SET_NAMES.get(r["set"], r["set"])}</td>'
        f'<td class="num">{fmt(r["parse_rate"])}</td><td class="num">{fmt(r["median_err"], False)}</td>'
        f'<td class="num">{fmt(r["win"])}</td><td class="num">{fmt(r["pair_success"])}</td>'
        f'<td class="num">{fmt(r["follow_rate"])}</td><td class="num">{fmt(r["obeyed"])}</td>'
        f'<td class="num">{fmt(r["kept_feasible"])}</td></tr>' for r in cells.values())

    gallery = ""
    if gen_v is not None:
        # one pair per viewpoint: yaw copies of the same frame look alike
        seen, ok_keys, bad_keys = set(), [], []
        for k, s_, view in zip(gen_v["keys"], gen_v["success"], gen_v["view"]):
            if view in seen:
                continue
            seen.add(view)
            (ok_keys if s_ else bad_keys).append(k)
        gallery += pair_cards(root, "general", "eval_pairs", ok_keys[:3] + bad_keys[:1], "eval_pairs.json", "eval pair")
    if hw_v is not None:
        gallery += pair_cards(root, "general", "probe_handwritten", hw_v["keys"][:2], "probe_handwritten.json", "hand-written")
    gallery += wrong_side_card(root)

    gm = json.loads((DATA / "general_meta.json").read_text()) if (DATA / "general_meta.json").exists() else {}
    mv = json.loads((DATA / "meta_v2.json").read_text()) if (DATA / "meta_v2.json").exists() else {}
    tr, ev = mv.get("train", {}), mv.get("eval", {})
    gen_ps = fmt(gen_v["success"].mean()) if gen_v else "–"
    return f"""<title>One Model, Both Prompts</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{v1.CSS}
table.wrap-cells td{{white-space:normal;vertical-align:top}}
.status.neutral{{color:var(--ink-2);background:var(--grid)}}</style>
<div class="wrap">
<header>
  <span class="eyebrow">Rover path VLM · Qwen3.5-2B LoRA · every Habitat round-4 frame</span>
  <h1>One Model, Both Prompts</h1>
  <p class="lede">One adapter trained on every round-4 frame twice: once with round 4's plain prompt, once with a
  route instruction (plus the opposite instruction on frames with a route on each side). It should plan as well as
  round 4 when told nothing, and follow the instruction when told something.</p>
  <div class="verdict">
    <div><span class="big">{fmt(plain.get("median_err"), False)}</span><span class="small">median point error with the plain prompt (round 4: {fmt(r4.get("mean_point_error_median"), False)})</span></div>
    <div><span class="big">{gen_ps}</span><span class="small">pair success on {len(gen_v["success"]) if gen_v else "–"} held-out two-route pairs</span></div>
    <div><span class="big">{fmt(ws.get("obeyed"))}</span><span class="small">of blocked-side instructions obeyed ({fmt(ws.get("kept_feasible"))} kept to the feasible side)</span></div>
  </div>
</header>

<section>
  <div class="stack"><span class="eyebrow">Verdict</span><h2>Against the pass criteria</h2></div>
  <div class="table-scroll"><table class="wrap-cells"><thead><tr><th>Question</th><th>Result</th><th>Needed</th><th></th></tr></thead>
  <tbody>{"".join(verdict)}</tbody></table></div>
</section>

<section>
  <div class="stack"><span class="eyebrow">Steering</span><h2>Pair success by frame source</h2>
  <p>A pair succeeds when the model follows both opposite instructions on one frame. Yaw-augmented frames are small
  camera rotations of original viewpoints, so intervals resample whole viewpoints rather than single pairs.</p></div>
  <div class="chart-box"><div class="legend" aria-hidden="true"><span><i style="background:var(--arm-cosmos)"></i>General model</span>
  <span><i style="background:var(--arm-template)"></i>v1 Cosmos model (1.4k pair records only)</span>
  <span><i style="background:var(--ref)"></i>hand-written probe</span></div>
  <div class="chart-scroll">{dot_chart(chart_rows)}</div></div>
</section>

<section>
  <div class="stack"><span class="eyebrow">Examples</span><h2>Frames, instructions and paths</h2>
  <p>Grey: the labelled routes. Blue / orange: the prediction for the left / right instruction. Filled waypoints are
  visible ground, hollow ones are hidden behind an obstacle.</p></div>
  <div class="gallery">{gallery}</div>
</section>

<section>
  <div class="stack"><span class="eyebrow">Every cell</span><h2>All numbers</h2></div>
  <div class="table-scroll"><table><thead><tr><th>Model</th><th>Eval set</th><th>Parse</th><th>Median err</th><th>Beats to_goal</th>
  <th>Pair success</th><th>Follow rate</th><th>Obeyed</th><th>Kept feasible</th></tr></thead><tbody>{table}</tbody></table></div>
  <p class="small">Training set: {tr.get("records", "–")} records ({gm.get("stats", {}).get("train", {}).get("plain", "–")} plain,
  the rest instructed after Cosmos labelling; {tr.get("pairs_kept", "–")} pair frames kept). Eval: {ev.get("eval_plain", "–")} plain,
  {ev.get("eval_instructed", "–")} instructed, {ev.get("eval_pairs", "–")} pair records.</p>
</section>

<section>
  <div class="stack"><span class="eyebrow">Limits</span><h2>What this does not show</h2>
  <ul class="plain">
    <li>The eval pairs come from few independent viewpoints (yaw copies of the same frames), hence the wide intervals.</li>
    <li>Habitat renders only; real camera frames are untested.</li>
    <li>The blocked-side probe reads which way the prediction bends; Habitat draws no route on the blocked side, so there is no path to compare against.</li>
    <li>The hand-written instructions were written by Claude from the frames, not by an operator.</li>
  </ul></div>
  <pre>uv run scripts/prepare_habitat_instr.py general
sbatch slurm/label_instructions.sbatch
uv run scripts/prepare_habitat_instr.py finalize-general --instructions outputs/instr_labels_v2/train_i3.jsonl outputs/instr_labels_v2/eval_i3.jsonl
sbatch --time=14:00:00 slurm/train_habitat.sbatch data/prepared_habitat_instr_v2/train_general.json outputs/runs_instr_v2/general --epochs 1 --save-steps 500 --resume
sbatch slurm/eval_instruction_v2.sbatch
uv run scripts/visualize_instruction_general.py</pre>
</section>
</div>
<script>{v1.JS}</script>"""


def main() -> None:
    global DATA
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--data-dir", type=Path, default=DATA, help="the v2 dataset dir (eval files, meta)")
    args = p.parse_args()
    DATA = args.data_dir
    if not cells_table(args.root):
        raise SystemExit(f"no metrics under {args.root} — run slurm/eval_instruction_v2.sbatch first")
    args.out.write_text(build_page(args.root))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
