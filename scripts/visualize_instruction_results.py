"""Results page for the instruction-conditioned path experiment. CPU only.

Reads outputs/eval_instr_v1/<set>/<tag>/{metrics,predictions}.json (written by
slurm/eval_instruction.sbatch), the dataset meta under data/prepared_habitat_instr_v1/ and the
label files under outputs/instr_labels/, and writes one self-contained HTML page
(outputs/instruction_results.html): the verdict, pair success per model x wording with
bootstrap intervals, a gallery of held-out pairs (one frame, both instructions, both
predictions), how the Cosmos instructions were made (and why the first prompt failed), and
the dataset funnel.

    uv run scripts/visualize_instruction_results.py
    uv run scripts/visualize_instruction_results.py --gallery-tag instr_cosmos --gallery-set cosmos --n 5
"""

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from rover_vlm.eval import instruction_follow  # noqa: E402
from rover_vlm.overlay import draw_polyline, embed_jpeg  # noqa: E402

ROOT = REPO_ROOT / "outputs" / "eval_instr_v1"
DATA = REPO_ROOT / "data" / "prepared_habitat_instr_v1"
LABELS = REPO_ROOT / "outputs" / "instr_labels"
OUT = REPO_ROOT / "outputs" / "instruction_results.html"
TAGS = ["instr_template", "instr_cosmos", "r4_lora", "base"]
SETS = ["template", "cosmos"]
MODEL_NAMES = {
    "instr_template": "Template-trained",
    "instr_cosmos": "Cosmos-trained",
    "r4_lora": "Round-4 model",
    "base": "Base Qwen3.5-2B",
}
MODEL_NOTES = {
    "instr_template": "fine-tuned on 2 fixed sentences",
    "instr_cosmos": "fine-tuned on 1,086 distinct Cosmos sentences",
    "r4_lora": "never trained on instructions",
    "base": "no fine-tuning",
}
SET_NAMES = {"template": "template wording", "cosmos": "Cosmos wording"}
# chart identity: one hue per training arm (validated pair), neutral grey for references
ARM_VAR = {"instr_template": "--arm-template", "instr_cosmos": "--arm-cosmos",
           "r4_lora": "--ref", "base": "--ref"}
# prediction overlay colours (image content, fixed): left instruction / right instruction
LEFT_RGB, RIGHT_RGB, ROUTE_RGB = (42, 120, 214), (235, 104, 52), (255, 255, 255)

# Measured on the 100-pair i1 gate (2026-09-25, serve job 91865); the control run used a
# throwaway script, so its numbers are recorded here rather than re-read from disk.
I1_GATE = {"pair_survival": 0.25, "verify_acc": 0.59, "control_verify_acc": 0.633,
           "control_pair_survival": 0.37, "no_side_word": 0.48, "side_correct": 0.67}


def collect_cells(root):
    """{(tag, set): metrics} for every finished eval cell under root/<set>/<tag>/."""
    cells = {}
    for m in Path(root).glob("*/*/metrics.json"):
        cells[(m.parent.name, m.parent.parent.name)] = json.loads(m.read_text())
    return cells


def pair_outcomes(preds):
    """{pair_id: {side: (record, follows)}} with the same follow rule as the metric."""
    out = {}
    for r in preds:
        hm = r["habitat_meta"]
        ok = 0
        if r.get("parsed") and r["parsed"].get("path"):
            ok = instruction_follow(r["parsed"]["path"], r["gt"]["path"], hm["alt_path"])["follows"]
        out.setdefault(hm["pair_id"], {})[hm["side"]] = (r, ok)
    return out


def bootstrap_ci(values, n=4000, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pair_success_vector(root, tag, set_):
    preds = json.loads((Path(root) / set_ / tag / "predictions.json").read_text())
    pairs = pair_outcomes(preds)
    return np.array([all(ok for _, ok in sides.values()) for _, sides in sorted(pairs.items())
                     if len(sides) == 2], float)


def paired_diff(a, b):
    d = a - b
    lo, hi = bootstrap_ci(d)
    return float(d.mean()), lo, hi


# --- rendering ------------------------------------------------------------------------

def _pct(v):
    return f"{100 * v:.1f}%"


def chart_svg(rows):
    """Horizontal dot plot, pair success 0-100% with 95% intervals; one row per cell."""
    W, left, right, row_h, top = 720, 250, 28, 44, 34
    H = top + row_h * len(rows) + 30
    x = lambda v: left + v * (W - left - right)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="chart-title chart-desc" class="chart">',
             '<desc id="chart-desc">Pair success for each model and instruction wording, with 95% bootstrap intervals.</desc>']
    for t in range(0, 101, 20):
        xv = x(t / 100)
        parts.append(f'<line x1="{xv:.1f}" x2="{xv:.1f}" y1="{top - 8}" y2="{H - 26}" class="grid"/>')
        parts.append(f'<text x="{xv:.1f}" y="{H - 8}" class="tick" text-anchor="middle">{t}%</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + row_h / 2
        parts.append(f'<text x="0" y="{y - 3:.1f}" class="row-label">{html.escape(MODEL_NAMES[r["tag"]])}</text>')
        parts.append(f'<text x="0" y="{y + 13:.1f}" class="row-sub">{html.escape(SET_NAMES[r["set"]])}</text>')
        color = f"var({ARM_VAR[r['tag']]})"
        lo, hi = r["ci"]
        tip = html.escape(f'{MODEL_NAMES[r["tag"]]} · {SET_NAMES[r["set"]]}: {_pct(r["ps"])} '
                          f'(95% CI {_pct(lo)}–{_pct(hi)}), follow rate {_pct(r["follow"])}')
        parts.append(f'<g class="mark" tabindex="0" data-tip="{tip}">')
        parts.append(f'<rect x="{left - 6}" y="{y - row_h / 2 + 2:.1f}" width="{W - left - right + 12}" '
                     f'height="{row_h - 4}" class="hit"/>')
        if hi > lo:
            parts.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                         f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x(r["ps"]):.1f}" cy="{y:.1f}" r="6" fill="{color}" class="dot"/>')
        anchor, dx = ("start", 12) if r["ps"] < 0.75 else ("end", -12)
        lx = x(hi) + 10 if anchor == "start" and hi > lo else x(r["ps"]) + dx
        if anchor == "end":
            lx = x(lo) - 10 if hi > lo else x(r["ps"]) + dx
        parts.append(f'<text x="{lx:.1f}" y="{y + 4:.1f}" class="value" text-anchor="{anchor}">{_pct(r["ps"])}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def pair_figure(image_path, sides, size=460):
    """One frame: both candidate routes in white, the two predictions in blue / orange."""
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for side in ("left", "right"):
        r, _ = sides[side]
        draw_polyline(d, [(p[0], p[1]) for p in r["gt"]["path"]], (150, 150, 150), size, size, width=3)
    for side, rgb in (("left", LEFT_RGB), ("right", RIGHT_RGB)):
        r, _ = sides[side]
        if r.get("parsed") and r["parsed"].get("path"):
            draw_polyline(d, [(p[0], p[1]) for p in r["parsed"]["path"]], rgb, size, size, width=4)
    return embed_jpeg(img, max_px=size, quality=82)


def gallery(root, tag, set_, n):
    preds = json.loads((Path(root) / set_ / tag / "predictions.json").read_text())
    evals = {r["id"]: r for r in json.loads((DATA / f"eval_{set_}.json").read_text())}
    pairs = [(pid, s) for pid, s in sorted(pair_outcomes(preds).items()) if len(s) == 2]

    def separation(item):
        s = item[1]
        return instruction_follow(s["left"][0]["gt"]["path"], s["left"][0]["gt"]["path"],
                                  s["right"][0]["gt"]["path"])["frechet_alt"]

    good = sorted([p for p in pairs if all(ok for _, ok in p[1].values())], key=separation, reverse=True)
    bad = sorted([p for p in pairs if not all(ok for _, ok in p[1].values())], key=separation, reverse=True)
    n_bad = max(1, n // 3)
    chosen = good[: n - n_bad] + bad[:n_bad]
    cards = []
    for pid, sides in chosen:
        src = evals[sides["left"][0]["id"]]["image"][0]
        lines = []
        for side, cls in (("left", "k-left"), ("right", "k-right")):
            r, ok = sides[side]
            instr = evals[r["id"]]["habitat_meta"]["instruction"]
            verdict = "followed" if ok else "not followed"
            lines.append(f'<li class="{cls}"><span class="swatch" aria-hidden="true"></span>'
                         f'<span class="instr">“{html.escape(instr)}”</span>'
                         f'<span class="outcome {"ok" if ok else "miss"}">{verdict}</span></li>')
        status = "both followed" if all(ok for _, ok in sides.values()) else "pair failed"
        cards.append(
            f'<figure class="pair"><img src="{pair_figure(src, sides)}" alt="Held-out frame {html.escape(pid)} '
            f'with both candidate routes and the model\'s two predicted paths" loading="lazy" width="460" height="460">'
            f'<figcaption><div class="pair-head"><span class="pid">{html.escape(pid)}</span>'
            f'<span class="status {"ok" if status == "both followed" else "miss"}">{status}</span></div>'
            f'<ul class="instr-list">{"".join(lines)}</ul></figcaption></figure>')
    return "".join(cards)


def blind_example(root, set_):
    """The round-4 model on one frame: two opposite instructions, one route."""
    p = Path(root) / set_ / "r4_lora" / "predictions.json"
    if not p.exists():
        return ""
    preds = json.loads(p.read_text())
    evals = {r["id"]: r for r in json.loads((DATA / f"eval_{set_}.json").read_text())}
    for pid, sides in sorted(pair_outcomes(preds).items()):
        if len(sides) == 2 and sum(ok for _, ok in sides.values()) == 1:
            src = evals[sides["left"][0]["id"]]["image"][0]
            li = evals[sides["left"][0]["id"]]["habitat_meta"]["instruction"]
            ri = evals[sides["right"][0]["id"]]["habitat_meta"]["instruction"]
            return (f'<figure class="pair blind"><img src="{pair_figure(src, sides)}" alt="Round-4 model on frame '
                    f'{html.escape(pid)}: both predictions take the same side" loading="lazy" width="460" height="460">'
                    f'<figcaption><div class="pair-head"><span class="pid">Round-4 model · {html.escape(pid)}</span>'
                    f'<span class="status miss">same route twice</span></div>'
                    f'<ul class="instr-list"><li class="k-left"><span class="swatch" aria-hidden="true"></span>'
                    f'<span class="instr">“{html.escape(li)}”</span></li><li class="k-right"><span class="swatch" '
                    f'aria-hidden="true"></span><span class="instr">“{html.escape(ri)}”</span></li></ul>'
                    f'</figcaption></figure>')
    return ""


def label_stats():
    out = {}
    for name in ("gate_i2", "train_i2", "eval_i2"):
        f = LABELS / f"{name}.jsonl"
        if f.exists():
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            ok = [r for r in rows if r.get("passed")]
            out[name] = {"n": len(rows), "passed": len(ok),
                         "distinct": len({r["instruction"] for r in ok}),
                         "median_words": int(np.median([len(r["instruction"].split()) for r in ok])) if ok else 0}
    return out


CSS = """
:root{
  --bg:#f5f6f4; --panel:#ffffff; --ink:#141817; --ink-2:#4d5553; --ink-3:#6d7573;
  --rule:#dfe3e1; --grid:#e7eae8; --accent:#b0006e; --accent-soft:#f6e3ef;
  --arm-template:#2a78d6; --arm-cosmos:#eb6834; --ref:#8a918f;
  --ok:#0a7d33; --miss:#b3261e; --code-bg:#eef1ef;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#131615; --panel:#1b1f1e; --ink:#eef1ef; --ink-2:#b9c0be; --ink-3:#8f9895;
    --rule:#2c3331; --grid:#262c2a; --accent:#ff5cb8; --accent-soft:#3a1a2e;
    --arm-template:#3987e5; --arm-cosmos:#d95926; --ref:#7f8785;
    --ok:#4cc27a; --miss:#ff7a70; --code-bg:#20262a;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#131615; --panel:#1b1f1e; --ink:#eef1ef; --ink-2:#b9c0be; --ink-3:#8f9895;
  --rule:#2c3331; --grid:#262c2a; --accent:#ff5cb8; --accent-soft:#3a1a2e;
  --arm-template:#3987e5; --arm-cosmos:#d95926; --ref:#7f8785;
  --ok:#4cc27a; --miss:#ff7a70; --code-bg:#20262a;
}
body{background:var(--bg);color:var(--ink);font:16px/1.6 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1040px;margin:0 auto;padding-inline:20px;padding-block:40px 72px;}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;line-height:1.15;text-wrap:balance;margin:0}
h1{font-size:clamp(2.1rem,5vw,3.2rem);font-weight:600;letter-spacing:-.01em}
h2{font-size:1.55rem;font-weight:600;margin-bottom:.5rem}
h3{font-size:1.1rem;font-weight:600}
p{margin:0;max-width:68ch}
.eyebrow{font:500 .75rem/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.09em;text-transform:uppercase;color:var(--accent)}
.lede{font-size:1.15rem;color:var(--ink-2);max-width:62ch}
.stack{display:flex;flex-direction:column;gap:14px}
section{display:flex;flex-direction:column;gap:18px;padding-block:36px;border-top:1px solid var(--rule)}
header{display:flex;flex-direction:column;gap:16px;padding-bottom:32px}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.verdict div{background:var(--panel);padding:16px 18px;display:flex;flex-direction:column;gap:4px}
.big{font:600 2rem/1.1 "IBM Plex Sans Condensed",system-ui,sans-serif;font-variant-numeric:tabular-nums}
.small{font-size:.88rem;color:var(--ink-2)}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86em}
.chart-box{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:18px 18px 8px;position:relative}
.chart-scroll{overflow-x:auto}
.chart{width:100%;min-width:560px;height:auto;display:block}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .tick{fill:var(--ink-3);font:12px "IBM Plex Mono",ui-monospace,monospace}
.chart .row-label{fill:var(--ink);font:600 14px "IBM Plex Sans",system-ui,sans-serif}
.chart .row-sub{fill:var(--ink-3);font:13px "IBM Plex Sans",system-ui,sans-serif}
.chart .value{fill:var(--ink);font:600 13px "IBM Plex Mono",ui-monospace,monospace}
.chart .hit{fill:transparent}
.chart .dot{stroke:var(--panel);stroke-width:2}
.chart .mark:hover .hit,.chart .mark:focus .hit{fill:var(--accent-soft)}
.chart .mark:focus{outline:none}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:.88rem;color:var(--ink-2);padding:6px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:8px}
.legend i{width:11px;height:11px;border-radius:50%;display:inline-block}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:.82rem;padding:6px 9px;border-radius:4px;max-width:320px;line-height:1.35}
.table-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-weight:600;color:var(--ink-2);font-size:.8rem;letter-spacing:.04em;text-transform:uppercase}
td.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86rem}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}
.pair{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:6px;overflow:hidden;display:flex;flex-direction:column}
.pair img{width:100%;height:auto;display:block;aspect-ratio:1/1}
.pair figcaption{padding:12px 14px 14px;display:flex;flex-direction:column;gap:10px}
.pair-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap}
.pid{font:12px "IBM Plex Mono",ui-monospace,monospace;color:var(--ink-3)}
.status{font:600 .72rem/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;padding:4px 7px;border-radius:3px}
.status.ok,.outcome.ok{color:var(--ok)} .status.miss,.outcome.miss{color:var(--miss)}
.status.ok{background:color-mix(in srgb,var(--ok) 12%,transparent)} .status.miss{background:color-mix(in srgb,var(--miss) 12%,transparent)}
.instr-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.instr-list li{display:grid;grid-template-columns:14px 1fr;column-gap:8px;font-size:.92rem;line-height:1.4}
.instr-list .swatch{width:14px;height:4px;border-radius:2px;margin-top:.6em}
.k-left .swatch{background:rgb(42,120,214)} .k-right .swatch{background:rgb(235,104,52)}
.instr-list .outcome{grid-column:2;font:500 .75rem "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.04em}
.key{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:.88rem;color:var(--ink-2)}
.key span{display:inline-flex;align-items:center;gap:8px}
.key i{width:18px;height:4px;border-radius:2px;display:inline-block}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px;align-items:start}
.note{border-left:3px solid var(--accent);padding:4px 0 4px 16px;color:var(--ink-2)}
.funnel{display:flex;flex-direction:column;gap:8px;font-variant-numeric:tabular-nums}
.funnel .row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;font-size:.92rem}
.funnel .bar{grid-column:1/-1;height:6px;border-radius:3px;background:var(--grid);overflow:hidden}
.funnel .bar i{display:block;height:100%;background:var(--accent)}
pre{background:var(--code-bg);border-radius:6px;padding:14px 16px;overflow-x:auto;font:13px/1.55 "IBM Plex Mono",ui-monospace,monospace;margin:0}
ul.plain{margin:0;padding-left:1.1rem;display:flex;flex-direction:column;gap:6px;max-width:70ch}
a{color:var(--accent)}
@media (prefers-reduced-motion:no-preference){.chart .dot{transition:r .15s}.chart .mark:hover .dot{r:7.5}}
"""

JS = """
(function(){
  var box=document.querySelector('.chart-box'); if(!box) return;
  var tip=document.createElement('div'); tip.className='tip'; tip.hidden=true; box.appendChild(tip);
  function show(el,x,y){tip.textContent=el.getAttribute('data-tip'); tip.hidden=false;
    var b=box.getBoundingClientRect(); var left=Math.min(x-b.left+14,b.width-tip.offsetWidth-8);
    tip.style.left=Math.max(8,left)+'px'; tip.style.top=(y-b.top+14)+'px';}
  box.querySelectorAll('.mark').forEach(function(el){
    el.addEventListener('mousemove',function(e){show(el,e.clientX,e.clientY)});
    el.addEventListener('mouseleave',function(){tip.hidden=true});
    el.addEventListener('focus',function(){var r=el.getBoundingClientRect(); show(el,r.left+r.width/2,r.top+r.height/2)});
    el.addEventListener('blur',function(){tip.hidden=true});
  });
})();
"""


def build_page(root, gallery_tag, gallery_set, n):
    cells = collect_cells(root)
    order = [("instr_cosmos", "cosmos"), ("instr_cosmos", "template"), ("instr_template", "template"),
             ("instr_template", "cosmos"), ("r4_lora", "cosmos"), ("base", "cosmos")]
    rows, vecs = [], {}
    for tag, s in order:
        if (tag, s) not in cells:
            continue
        v = pair_success_vector(root, tag, s)
        vecs[(tag, s)] = v
        ins = cells[(tag, s)]["instruction"]
        rows.append({"tag": tag, "set": s, "ps": ins["pair_success"], "ci": bootstrap_ci(v),
                     "follow": ins["follow_rate"], "n_pairs": ins["n_pairs"], "m": cells[(tag, s)], "ins": ins})
    get = {(r["tag"], r["set"]): r for r in rows}
    d_cos = paired_diff(vecs[("instr_cosmos", "cosmos")], vecs[("instr_template", "cosmos")])
    d_tpl = paired_diff(vecs[("instr_cosmos", "template")], vecs[("instr_template", "template")])
    n_pairs = get[("instr_cosmos", "cosmos")]["n_pairs"]
    pm = json.loads((DATA / "pairs_meta.json").read_text())
    fm = json.loads((DATA / "meta.json").read_text())
    ls = label_stats()

    table_rows = "".join(
        f'<tr><td>{MODEL_NAMES[r["tag"]]}</td><td>{SET_NAMES[r["set"]]}</td>'
        f'<td class="num">{_pct(r["ps"])}</td><td class="num">{_pct(r["ci"][0])}–{_pct(r["ci"][1])}</td>'
        f'<td class="num">{_pct(r["follow"])}</td>'
        f'<td class="num">{_pct(r["ins"]["follow_rate_by_side"]["left"])} / {_pct(r["ins"]["follow_rate_by_side"]["right"])}</td>'
        f'<td class="num">{r["m"]["mean_point_error_median"]:.3f}</td><td class="num">{_pct(r["m"]["parse_rate"])}</td></tr>'
        for r in rows)

    total_frames = pm["stats"]["train"]["pairs"] + pm["stats"]["eval"]["pairs"]
    skipped = pm["skipped"]
    two_sided = total_frames + skipped["entry_differs"] + skipped["routes_too_similar"]
    scanned = two_sided + skipped["no_two_sided"]
    kept = fm["splits"]["train"]["pairs_kept"] + fm["splits"]["eval"]["pairs_kept"]
    funnel = [
        ("Original round-4 frames scanned", scanned),
        ("with a feasible route on each side", two_sided),
        ("…both entering the frame at the same point", two_sided - skipped["entry_differs"]),
        ("…and clearly different routes", total_frames),
        ("…with both Cosmos instructions accepted", kept),
    ]
    funnel_html = "".join(
        f'<div class="row"><span>{label}</span><span class="mono">{v:,}</span>'
        f'<div class="bar"><i style="width:{100 * v / scanned:.1f}%"></i></div></div>' for label, v in funnel)

    g = ls.get("gate_i2", {})
    tr, ev = ls.get("train_i2", {}), ls.get("eval_i2", {})
    cos_c, cos_t = get[("instr_cosmos", "cosmos")], get[("instr_cosmos", "template")]
    tpl_t, tpl_c = get[("instr_template", "template")], get[("instr_template", "cosmos")]
    r4 = get[("r4_lora", "cosmos")]

    return f"""<title>Steering by Instruction</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <span class="eyebrow">Rover path VLM · Qwen3.5-2B LoRA · Habitat round 4 frames</span>
  <h1>Steering by Instruction</h1>
  <p class="lede">The route instruction becomes an input. The model sees the frame, where the route enters and
  where the goal is, plus one sentence such as “Go left around the stone pillar”, and must draw the route that
  obeys it. Every test frame has a feasible route on each side of an obstacle, so the instruction alone decides
  which one is right.</p>
  <div class="verdict">
    <div><span class="big">{_pct(cos_c["ps"])}</span><span class="small">Cosmos-trained model, pair success on Cosmos wording ({n_pairs} held-out frames)</span></div>
    <div><span class="big">{_pct(r4["ps"])}</span><span class="small">round-4 model on the same frames: it draws the same route whatever it is told</span></div>
    <div><span class="big">+{100 * d_cos[0]:.0f} pts</span><span class="small">Cosmos-trained over template-trained on Cosmos wording (95% CI +{100 * d_cos[1]:.0f} to +{100 * d_cos[2]:.0f})</span></div>
  </div>
</header>

<section>
  <div class="stack">
    <span class="eyebrow">Result</span>
    <h2 id="chart-title">Pair success by model and instruction wording</h2>
    <p>A pair succeeds when the model follows <em>both</em> instructions on a frame: its path for “left” lies closer
    to the left route than to the right one, and vice versa (Fréchet distance). A model that ignores the
    instruction returns one route twice and wins at most one of the two, so it scores 0% by construction.
    Bars are 95% bootstrap intervals over the {n_pairs} frames.</p>
  </div>
  <div class="chart-box">
    <div class="legend" aria-hidden="true"><span><i style="background:var(--arm-cosmos)"></i>Cosmos-trained</span>
      <span><i style="background:var(--arm-template)"></i>Template-trained</span>
      <span><i style="background:var(--ref)"></i>References</span></div>
    <div class="chart-scroll">{chart_svg(rows)}</div>
  </div>
  <div class="two">
    <p class="note">The template-trained model saw only two sentences in training (“Go around to the left/right.”).
    It reaches {_pct(tpl_t["ps"])} on that wording but falls to {_pct(tpl_c["ps"])} on Cosmos's varied sentences.
    The Cosmos-trained model holds {_pct(cos_t["ps"])} and {_pct(cos_c["ps"])} on the two. On template wording its
    lead of {100 * d_tpl[0]:+.1f} points is within noise (CI {100 * d_tpl[1]:+.0f} to {100 * d_tpl[2]:+.0f}); on Cosmos wording
    the +{100 * d_cos[0]:.1f} lead is not.</p>
    <p class="note">Following the instruction costs no geometry. On these same frames the round-4 model's median
    point error is {r4["m"]["mean_point_error_median"]:.3f}; the instruction-trained models reach
    {min(cos_c["m"]["mean_point_error_median"], cos_t["m"]["mean_point_error_median"]):.3f}–{max(tpl_t["m"]["mean_point_error_median"], tpl_c["m"]["mean_point_error_median"]):.3f}.
    These two-route frames are harder than round 4's average frame (0.011 there).</p>
  </div>
  <div class="table-scroll"><table>
    <thead><tr><th>Model</th><th>Instructions</th><th>Pair success</th><th>95% CI</th><th>Follow rate</th><th>Left / right</th><th>Median point err</th><th>Parse</th></tr></thead>
    <tbody>{table_rows}</tbody></table></div>
</section>

<section>
  <div class="stack">
    <span class="eyebrow">Held-out examples · {MODEL_NAMES[gallery_tag]} · {SET_NAMES[gallery_set]}</span>
    <h2>One frame, two instructions, two paths</h2>
    <p>Both candidate routes in grey with a white outline; the model's path for the left instruction in blue and for the right
    instruction in orange. Frames with the most clearly separated routes first; the last cards are failures.</p>
    <div class="key"><span><i style="background:rgb(150,150,150);outline:1px solid #fff"></i>both candidate routes (grey, white outline)</span>
      <span><i style="background:rgb(42,120,214)"></i>prediction for the left instruction</span>
      <span><i style="background:rgb(235,104,52)"></i>prediction for the right instruction</span></div>
  </div>
  <div class="gallery">{gallery(root, gallery_tag, gallery_set, n)}{blind_example(root, gallery_set)}</div>
</section>

<section>
  <div class="stack">
    <span class="eyebrow">Method</span>
    <h2>How the instructions were made</h2>
  </div>
  <div class="two">
    <div class="stack">
      <h3>First attempt failed at chance</h3>
      <p>Prompt i1 showed Cosmos3-Nano the frame with the target route in magenta and the alternative in grey, asked
      it to describe the magenta route, and had a second Cosmos call pick which of two identically drawn routes
      the instruction described. On 100 pairs only {_pct(I1_GATE["pair_survival"])} survived, which is what coin flips
      give (0.5 × 0.5).</p>
      <p>The control settled it: handed the correct template instruction, the verifier still chose the right route
      only {_pct(I1_GATE["control_verify_acc"])} of the time. Cosmos3-Nano cannot tell left from right on drawn
      routes. {_pct(I1_GATE["no_side_word"])} of its i1 instructions named no side at all.</p>
    </div>
    <div class="stack">
      <h3>Prompt i2: hand over the side</h3>
      <p>The side of each route is known exactly from the Habitat metadata, so i2 tells Cosmos the side and asks
      only for the landmark the route passes. The Cosmos verify call is gone; a text check replaces it: the
      correct side word is present, the opposite one absent, something is named, and nothing refers to the drawing.</p>
      <p>Gate: {_pct(g.get("passed", 0) / max(1, g.get("n", 1)))} of records accepted. Full run:
      {tr.get("passed", 0):,} of {tr.get("n", 0):,} train and {ev.get("passed", 0)} of {ev.get("n", 0)} eval records,
      {tr.get("distinct", 0):,} distinct train sentences, median {tr.get("median_words", 0)} words. A named landmark
      that is not in the frame is the one error no check catches; a spot check found it rare.</p>
    </div>
  </div>
</section>

<section>
  <div class="stack">
    <span class="eyebrow">Data</span>
    <h2>From round-4 frames to test pairs</h2>
    <p>Only frames where the instruction genuinely decides the route are usable. When the two routes enter the
    image far apart, the entry point in the prompt already gives the side away; when they nearly coincide, there
    is nothing to follow. The split is round 4's, scene-disjoint.</p>
  </div>
  <div class="funnel">{funnel_html}</div>
  <p class="small">Kept: {fm["splits"]["train"]["pairs_kept"]} train / {fm["splits"]["eval"]["pairs_kept"]} eval pairs
  ({fm["splits"]["train"]["records"]:,} / {fm["splits"]["eval"]["records"]} records). Training: fresh LoRA on
  Qwen3.5-2B, 3 epochs, 273 steps, 36 min per arm on a 3g.40gb A100 slice; both arms use identical frames,
  answers and settings, and differ only in the instruction text.</p>
</section>

<section>
  <div class="stack">
    <span class="eyebrow">Limits</span>
    <h2>What this does not show yet</h2>
    <ul class="plain">
      <li>{n_pairs} held-out pairs is small; the intervals above are wide.</li>
      <li>Habitat renders only. Earlier real-image evals found Habitat-trained path models do not transfer to real
      robot frames, so instruction following on real images is untested.</li>
      <li>Only frames with two clearly different, shared-entry routes are in the set; how instructions behave on
      every other frame (approach A) is still to be measured.</li>
      <li>In these instructions “pass X on the left” always means the route goes to the left of X. The convention
      is consistent in training and test, but it is the model's convention, not necessarily an operator's.</li>
    </ul>
  </div>
  <pre>uv run scripts/prepare_habitat_instr.py pairs
uv run scripts/label_instructions.py --split data/prepared_habitat_instr_v1/pairs_train.json --out outputs/instr_labels/train_i2.jsonl --concurrency 8 --resume
uv run scripts/prepare_habitat_instr.py finalize --instructions outputs/instr_labels/train_i2.jsonl outputs/instr_labels/eval_i2.jsonl
sbatch slurm/train_habitat.sbatch data/prepared_habitat_instr_v1/train_cosmos.json outputs/runs_instr_v1/cosmos --epochs 3
sbatch slurm/eval_instruction.sbatch
uv run scripts/visualize_instruction_results.py</pre>
</section>
</div>
<script>{JS}</script>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--gallery-tag", default="instr_cosmos")
    p.add_argument("--gallery-set", default="cosmos")
    p.add_argument("--n", type=int, default=5, help="gallery pairs (about a third are failures)")
    args = p.parse_args()
    if not collect_cells(args.root):
        raise SystemExit(f"no metrics under {args.root} — run slurm/eval_instruction.sbatch first")
    args.out.write_text(build_page(args.root, args.gallery_tag, args.gallery_set, args.n))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
