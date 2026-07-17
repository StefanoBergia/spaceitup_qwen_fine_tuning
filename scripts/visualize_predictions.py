"""Build an interactive HTML explorer of model predictions vs. ground truth.

Qualitative companion to scripts/compare_evals.py: instead of the aggregate
scaling curve, this renders the *actual* predicted trajectories of every model
(base + each LoRA size) overlaid on the source image, so you can see how the
output improves as the training set grows.

Run on the login node after the evals exist (CPU-only, no GPU/SLURM):

    uv run scripts/visualize_predictions.py
    uv run scripts/visualize_predictions.py --per-bucket 4 --max-image-px 640

Reads data/prepared/eval.json (image + instruction) and every
outputs/eval/<tag>/predictions.json. Samples are picked to span difficulty:
they are ranked by the *base* model's waypoint error, split into
easy/medium/hard terciles, and --per-bucket samples are drawn from each.
Selected images are downscaled and base64-embedded so the output is a single
self-contained file (no external requests).

Writes:
    outputs/eval/prediction_explorer.html  — open in a browser, or publish as an artifact
"""

import argparse
import base64
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from rover_vlm.eval import in_unit_range, normalize_prediction

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = REPO_ROOT / "data" / "sharerobot" / "trajectory"
TRAJ_JSON = IMAGE_ROOT / "trajectory.json"
DEFAULT_EVAL_DIR = REPO_ROOT / "outputs" / "eval"
EVAL_FILE = REPO_ROOT / "data" / "prepared" / "eval.json"
META_FILE = REPO_ROOT / "data" / "prepared" / "meta.json"

# model panels, in scaling order (base first, then increasing LoRA train size)
MODEL_TAGS = ["base", "lora_train_500", "lora_train_1000", "lora_train_2000", "lora_train_5000", "lora_train_full"]


def model_label(tag: str, full_size: int) -> str:
    if tag == "base":
        return "base (zero-shot)"
    n = tag.removeprefix("lora_train_")
    if n == "full":
        return f"LoRA · full ({full_size / 1000:.1f}K)".replace(".0K", "K")
    if n.isdigit() and int(n) >= 1000:
        return f"LoRA · {int(n) // 1000}K"
    return f"LoRA · {n}"


def load_predictions(eval_dir: Path, tag: str) -> dict[int, dict]:
    """tag -> {sample id: prediction record}."""
    path = eval_dir / tag / "predictions.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — run the eval for tag {tag!r} first")
    return {rec["id"]: rec for rec in json.loads(path.read_text())}


def task_text(prompt: str) -> str:
    """Pull the human-readable task out of the templated prompt."""
    m = re.search(r'task is "([^"]+)"', prompt)
    return m.group(1) if m else prompt.replace("<image>", "").strip()


def embed_image(rel_path: str, max_px: int) -> str:
    """Downscale to max_px on the long side and return a base64 PNG data URI."""
    with Image.open(IMAGE_ROOT / rel_path) as im:
        img = im.convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_prediction(rec: dict) -> dict:
    """Display-ready prediction: coords in [0,1] + status + per-sample metrics."""
    parsed = rec.get("parsed")
    if not parsed:
        return {"status": "no-parse", "pts": None, "rescaled": False, "point_error": None, "frechet": None}
    pts, rescaled = normalize_prediction(parsed)
    status = "ok" if in_unit_range(pts) else "off-image"
    metrics = rec.get("metrics") or {}
    return {
        "status": status,
        "pts": [[round(x, 4), round(y, 4)] for x, y in pts] if status == "ok" else None,
        "rescaled": bool(rescaled),
        "point_error": metrics.get("mean_point_error"),
        "frechet": metrics.get("frechet"),
    }


def base_difficulty(rec: dict | None) -> float:
    """Ranking key: base-model waypoint error; unparseable/off-image sort hardest."""
    if rec is None:
        return float("inf")
    metrics = rec.get("metrics")
    pts = rec.get("parsed")
    if not metrics or not pts:
        return float("inf")
    norm, _ = normalize_prediction(pts)
    if not in_unit_range(norm):
        return float("inf")
    return metrics.get("mean_point_error", float("inf"))


def select_spread(ranked_ids: list[int], per_bucket: int) -> list[tuple[int, str]]:
    """Split difficulty-ranked ids into easy/medium/hard terciles, pick evenly from each."""
    n = len(ranked_ids)
    third = n // 3
    buckets = [
        ("easy", ranked_ids[:third]),
        ("medium", ranked_ids[third : 2 * third]),
        ("hard", ranked_ids[2 * third :]),
    ]
    picks: list[tuple[int, str]] = []
    for label, bucket in buckets:
        if not bucket:
            continue
        # evenly spaced representatives (deterministic), e.g. quartile positions for per_bucket=3
        idxs = sorted({min(len(bucket) - 1, round((k + 0.5) * len(bucket) / per_bucket)) for k in range(per_bucket)})
        picks.extend((bucket[i], label) for i in idxs)
    return picks


def dataset_stats(splits: dict) -> dict:
    """Whole-dataset summary (from the raw trajectory.json) + split sizes."""
    traj = json.loads(TRAJ_JSON.read_text())
    sources = Counter(s["meta_data"]["original_dataset"] for s in traj)
    sizes = Counter(f'{s["meta_data"]["original_width"]}×{s["meta_data"]["original_height"]}' for s in traj)
    npts = sorted(len(s["trajectory"]) for s in traj)
    return {
        "total": len(traj),
        "n_sources": len(sources),
        "top_sources": [name for name, _ in sources.most_common(3)],
        "n_sizes": len(sizes),
        "top_size": sizes.most_common(1)[0][0],
        "wp_min": npts[0],
        "wp_median": npts[len(npts) // 2],
        "wp_max": npts[-1],
        "eval": splits["eval"],
        "full": splits["train_full"],
    }


def overall_results(eval_dir: Path, full_size: int) -> list[dict]:
    """Aggregate eval metrics per model (mirrors outputs/eval/comparison.md)."""
    rows = []
    for tag in MODEL_TAGS:
        m = json.loads((eval_dir / tag / "metrics.json").read_text())
        suffix = tag.removeprefix("lora_train_")
        train = None if tag == "base" else (full_size if suffix == "full" else int(suffix))
        rows.append(
            {
                "label": model_label(tag, full_size),
                "is_base": tag == "base",
                "is_full": tag == "lora_train_full",
                "train": train,
                "parse": m["parse_rate"],
                "in_range": m["in_range_rate"],
                "pe_med": m["mean_point_error_median"],
                "pe_mean": m["mean_point_error_mean"],
                "fr_med": m["frechet_median"],
                "fr_mean": m["frechet_mean"],
            }
        )
    return rows


def build_data(eval_dir: Path, per_bucket: int, max_px: int) -> dict:
    eval_records = {rec["id"]: rec for rec in json.loads(EVAL_FILE.read_text())}
    splits = json.loads(META_FILE.read_text())["splits"]
    full_size = splits["train_full"]
    preds = {tag: load_predictions(eval_dir, tag) for tag in MODEL_TAGS}

    ranked = sorted(eval_records, key=lambda i: base_difficulty(preds["base"].get(i)))
    selected = select_spread(ranked, per_bucket)

    samples = []
    for sid, difficulty in selected:
        rec = eval_records[sid]
        base_err = base_difficulty(preds["base"].get(sid))
        samples.append(
            {
                "id": sid,
                "task": task_text(rec["conversations"][0]["value"]),
                "difficulty": difficulty,
                "base_error": None if base_err == float("inf") else round(base_err, 4),
                "image": embed_image(rec["image"][0], max_px),
                "gt": [[round(x, 4), round(y, 4)] for x, y in json.loads(rec["conversations"][1]["value"])],
                "preds": {tag: build_prediction(preds[tag].get(sid, {})) for tag in MODEL_TAGS},
            }
        )

    return {
        "models": [{"tag": tag, "label": model_label(tag, full_size)} for tag in MODEL_TAGS],
        "dataset": dataset_stats(splits),
        "results": overall_results(eval_dir, full_size),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# self-contained HTML (no <html>/<head>/<body> so it also publishes as an artifact;
# overlays are drawn client-side as SVG from the embedded normalized coordinates)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<title>Trajectory Prediction Explorer — Qwen3.5-2B LoRA scaling</title>
<style>
  :root {
    --bg: #f7f7f4; --panel: #ffffff; --ink: #16171a; --ink2: #4d4f55; --muted: #8a8c93;
    --line: #e4e4dd; --line2: #d3d3c9; --rail: #efefe9;
    --accent: #2a6ad6; --pred: #2a78d6; --gt: #008300;
    --easy: #2f8f57; --medium: #c08a1e; --hard: #c0522a;
    --shadow: 0 1px 2px rgba(20,20,25,.05), 0 6px 20px rgba(20,20,25,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14151a; --panel: #1c1e24; --ink: #f0f1f4; --ink2: #b6b9c2; --muted: #7f828c;
      --line: #2b2e36; --line2: #363a44; --rail: #191b21;
      --accent: #6fa8ff; --pred: #63a0ff; --gt: #37c76a;
      --easy: #4fbf7d; --medium: #e0b046; --hard: #ef7d54;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="light"] {
    --bg: #f7f7f4; --panel: #ffffff; --ink: #16171a; --ink2: #4d4f55; --muted: #8a8c93;
    --line: #e4e4dd; --line2: #d3d3c9; --rail: #efefe9;
    --accent: #2a6ad6; --pred: #2a78d6; --gt: #008300;
    --easy: #2f8f57; --medium: #c08a1e; --hard: #c0522a;
    --shadow: 0 1px 2px rgba(20,20,25,.05), 0 6px 20px rgba(20,20,25,.05);
  }
  :root[data-theme="dark"] {
    --bg: #14151a; --panel: #1c1e24; --ink: #f0f1f4; --ink2: #b6b9c2; --muted: #7f828c;
    --line: #2b2e36; --line2: #363a44; --rail: #191b21;
    --accent: #6fa8ff; --pred: #63a0ff; --gt: #37c76a;
    --easy: #4fbf7d; --medium: #e0b046; --hard: #ef7d54;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
  }

  * { box-sizing: border-box; }
  .wrap {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--ink); background: var(--bg); min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }

  header.top {
    padding: 22px 26px 18px; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; align-items: flex-end; gap: 16px 28px;
  }
  header.top h1 { margin: 0; font-size: 19px; letter-spacing: -0.01em; font-weight: 650; text-wrap: balance; }
  header.top .sub { margin: 3px 0 0; color: var(--ink2); font-size: 13px; }
  .spacer { flex: 1 1 auto; }
  .legend { display: flex; gap: 16px; align-items: center; font-size: 12px; color: var(--ink2); }
  .legend .k { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 22px; height: 0; border-top-width: 3px; border-top-style: solid; border-radius: 2px; }
  .swatch.gt { border-top-style: dashed; border-color: var(--gt); }
  .swatch.pred { border-color: var(--pred); }
  .toggle {
    display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: var(--ink2);
    cursor: pointer; user-select: none; border: 1px solid var(--line2); border-radius: 999px;
    padding: 5px 11px; background: var(--panel);
  }
  .toggle input { accent-color: var(--gt); margin: 0; }

  .layout { display: grid; grid-template-columns: 264px 1fr; align-items: start; }
  @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }

  aside.rail {
    border-right: 1px solid var(--line); background: var(--rail);
    position: sticky; top: 0; max-height: 100vh; overflow-y: auto; padding: 14px 12px 40px;
  }
  @media (max-width: 820px) { aside.rail { position: static; max-height: none; border-right: none; border-bottom: 1px solid var(--line); } }
  .rail h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .09em; color: var(--muted);
    margin: 16px 6px 8px; font-weight: 600;
  }
  .rail h2:first-child { margin-top: 4px; }
  .item {
    display: grid; grid-template-columns: 46px 1fr; gap: 10px; align-items: center;
    width: 100%; text-align: left; cursor: pointer; padding: 7px 8px; border-radius: 9px;
    border: 1px solid transparent; background: transparent; color: inherit; margin-bottom: 2px;
  }
  .item:hover { background: var(--panel); }
  .item.active { background: var(--panel); border-color: var(--line2); box-shadow: var(--shadow); }
  .item img { width: 46px; height: 46px; object-fit: cover; border-radius: 6px; display: block; background: #0002; }
  .item .meta { min-width: 0; }
  .item .tk { font-size: 12.5px; line-height: 1.25; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .item .err { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  .d-easy { background: var(--easy); } .d-medium { background: var(--medium); } .d-hard { background: var(--hard); }

  main { padding: 22px 26px 60px; min-width: 0; }
  .taskbar { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 4px; }
  .taskbar .lab { font-size: 11px; text-transform: uppercase; letter-spacing: .09em; color: var(--muted); }
  .taskbar h2 { margin: 0; font-size: 21px; font-weight: 640; letter-spacing: -0.01em; text-wrap: balance; }
  .taskbar h2::before { content: "\201C"; color: var(--muted); }
  .taskbar h2::after { content: "\201D"; color: var(--muted); }
  .chip {
    font-size: 11px; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--line2);
    color: var(--ink2); display: inline-flex; align-items: center;
  }
  .subline { color: var(--ink2); font-size: 12.5px; margin: 6px 0 20px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(184px, 1fr)); gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); }
  .card.best { border-color: color-mix(in srgb, var(--pred) 45%, var(--line)); }
  .card .hd { padding: 9px 12px 8px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .card .hd .name { font-size: 12.5px; font-weight: 600; }
  .card .hd .badge { font-size: 10px; padding: 1px 7px; border-radius: 999px; border: 1px solid var(--line2); color: var(--muted); white-space: nowrap; }
  .card .hd .badge.warn { color: var(--hard); border-color: color-mix(in srgb, var(--hard) 45%, var(--line2)); }
  .frame { position: relative; line-height: 0; background: #00000010; }
  .frame img { width: 100%; height: auto; display: block; }
  .frame svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .card .ft { padding: 8px 12px 10px; display: flex; gap: 14px; font-size: 12px; color: var(--ink2); }
  .card .ft .n { color: var(--muted); font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em; }
  .card .ft .v { display: block; margin-top: 1px; font-size: 13px; color: var(--ink); }
  .card .ft.none { color: var(--muted); font-size: 12px; }

  .nav { display: flex; gap: 10px; margin: 22px 0 0; }
  .nav button {
    font: inherit; font-size: 13px; padding: 8px 16px; border-radius: 9px; cursor: pointer;
    border: 1px solid var(--line2); background: var(--panel); color: var(--ink);
  }
  .nav button:hover { border-color: var(--accent); }
  .nav button:disabled { opacity: .4; cursor: default; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  /* overview band (dataset stats + overall results) */
  .overview { padding: 16px 26px 2px; }
  .overview details { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); box-shadow: var(--shadow); }
  .overview summary { cursor: pointer; padding: 13px 16px; font-size: 13px; font-weight: 600; list-style: none; display: flex; align-items: center; gap: 9px; }
  .overview summary::-webkit-details-marker { display: none; }
  .overview summary::before { content: "\25B8"; color: var(--muted); font-size: 11px; transition: transform .15s ease; }
  .overview details[open] summary::before { transform: rotate(90deg); }
  .overview summary .hint { font-weight: 400; color: var(--muted); font-size: 12px; }
  .ov-grid { display: grid; grid-template-columns: minmax(230px, 320px) 1fr; gap: 6px 30px; padding: 2px 18px 18px; align-items: start; }
  @media (max-width: 780px) { .ov-grid { grid-template-columns: 1fr; } }
  .ov-block h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 6px 0 10px; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.kv td { padding: 6px 0; vertical-align: top; border-bottom: 1px solid var(--line); }
  table.kv tr:last-child td { border-bottom: none; }
  table.kv td.k { color: var(--ink2); white-space: nowrap; padding-right: 18px; }
  table.kv td.v { text-align: right; font-variant-numeric: tabular-nums; }
  .res-wrap { overflow-x: auto; }
  table.res th, table.res td { padding: 7px 11px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; font-variant-numeric: tabular-nums; }
  table.res th { font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); font-weight: 600; border-bottom-color: var(--line2); }
  table.res th:first-child, table.res td:first-child { text-align: left; }
  table.res tbody tr:last-child td { border-bottom: none; }
  table.res tr.base td { color: var(--ink2); }
  table.res tr.full td { background: color-mix(in srgb, var(--pred) 9%, transparent); font-weight: 600; }
  table.res .paren { color: var(--muted); font-weight: 400; font-size: 11px; }
  td.barcell { width: 84px; }
  .bar { display: block; position: relative; height: 8px; background: var(--line); border-radius: 4px; }
  .bar .fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 4px; }
  .foot { font-size: 11px; color: var(--muted); margin: 12px 2px 0; line-height: 1.55; max-width: 62ch; }
</style>

<div class="wrap">
  <header class="top">
    <div>
      <h1>Trajectory Prediction Explorer</h1>
      <p class="sub">Qwen3.5-2B — base vs. LoRA by training-set size, on the held-out ShareRobot eval split</p>
    </div>
    <div class="spacer"></div>
    <div class="legend">
      <span class="k"><span class="swatch gt"></span>ground truth</span>
      <span class="k"><span class="swatch pred"></span>prediction</span>
      <label class="toggle"><input type="checkbox" id="gt-toggle" checked> show ground truth</label>
    </div>
  </header>

  <section class="overview">
    <details open>
      <summary>Dataset &amp; overall results <span class="hint">— ShareRobot trajectory subset · aggregate metrics on the held-out eval split</span></summary>
      <div class="ov-grid">
        <div class="ov-block">
          <h3>Dataset</h3>
          <table class="kv"><tbody id="kv"></tbody></table>
        </div>
        <div class="ov-block">
          <h3>Held-out eval results (500 samples)</h3>
          <div class="res-wrap">
            <table class="res">
              <thead><tr>
                <th>Model</th><th>Train</th><th>Parse</th><th>In-range</th>
                <th>Point err</th><th>Fréchet</th><th></th>
              </tr></thead>
              <tbody id="res-body"></tbody>
            </table>
          </div>
          <p class="foot" id="res-foot"></p>
        </div>
      </div>
    </details>
  </section>

  <div class="layout">
    <aside class="rail" id="rail"></aside>
    <main>
      <div class="taskbar">
        <span class="lab">task</span>
        <h2 id="task">—</h2>
        <span class="chip" id="diff">—</span>
      </div>
      <div class="subline" id="subline"></div>
      <div class="grid" id="grid"></div>
      <div class="nav">
        <button id="prev">← Previous sample</button>
        <button id="next">Next sample →</button>
      </div>
    </main>
  </div>
</div>

<script>
const DATA = __DATA__;
const GT_COLOR = getComputedStyle(document.documentElement).getPropertyValue('--gt') || '#008300';
let current = 0;
let showGT = true;

const diffColor = { easy: 'd-easy', medium: 'd-medium', hard: 'd-hard' };

function fmt(v, dp = 3) { return v == null ? '—' : Number(v).toFixed(dp); }

// Build an SVG overlay (viewBox 0..1) with ground-truth + prediction polylines.
function overlay(sample, pred) {
  const parts = [`<svg viewBox="0 0 1 1" preserveAspectRatio="none">`];
  const path = (pts, color, dash, r) => {
    if (!pts || pts.length === 0) return '';
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(4) + ' ' + p[1].toFixed(4)).join(' ');
    let s = `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.4" vector-effect="non-scaling-stroke"`
          + ` stroke-linejoin="round" stroke-linecap="round"${dash ? ' stroke-dasharray="6 4"' : ''}/>`;
    // start = ring, end = filled dot
    const a = pts[0], b = pts[pts.length - 1];
    s += `<circle cx="${a[0]}" cy="${a[1]}" r="${r}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    s += `<circle cx="${b[0]}" cy="${b[1]}" r="${r}" fill="${color}"/>`;
    return s;
  };
  if (showGT) parts.push(path(sample.gt, GT_COLOR.trim(), true, 0.022));
  const predColor = getComputedStyle(document.documentElement).getPropertyValue('--pred').trim();
  if (pred.status === 'ok') parts.push(path(pred.pts, predColor, false, 0.02));
  parts.push('</svg>');
  return parts.join('');
}

function card(sample, model, isLast) {
  const pred = sample.preds[model.tag];
  let badge = '';
  if (pred.status === 'no-parse') badge = '<span class="badge warn">unparseable</span>';
  else if (pred.status === 'off-image') badge = '<span class="badge warn">off-image</span>';
  else if (pred.rescaled) badge = '<span class="badge">rescaled ÷1000</span>';

  let footer;
  if (pred.status === 'ok') {
    footer = `<div class="ft mono">
        <div><span class="n">point err</span><span class="v">${fmt(pred.point_error)}</span></div>
        <div><span class="n">Fréchet</span><span class="v">${fmt(pred.frechet)}</span></div>
      </div>`;
  } else {
    const why = pred.status === 'no-parse' ? 'no coordinate list in output' : 'coordinates outside image';
    footer = `<div class="ft none">${why}</div>`;
  }

  return `<div class="card${isLast ? ' best' : ''}">
      <div class="hd"><span class="name">${model.label}</span>${badge}</div>
      <div class="frame"><img src="${sample.image}" alt="" loading="lazy">${overlay(sample, pred)}</div>
      ${footer}
    </div>`;
}

function renderMain() {
  const s = DATA.samples[current];
  document.getElementById('task').textContent = s.task;
  const diff = document.getElementById('diff');
  diff.innerHTML = `<span class="dot ${diffColor[s.difficulty]}"></span>${s.difficulty}`;
  const be = s.base_error == null ? 'unparseable / off-image' : s.base_error.toFixed(3);
  document.getElementById('subline').innerHTML =
    `sample #${s.id} · base-model waypoint error <span class="mono">${be}</span> · ${DATA.models.length} models below, left→right = base then increasing LoRA data`;
  document.getElementById('grid').innerHTML =
    DATA.models.map((m, i) => card(s, m, i === DATA.models.length - 1)).join('');
  document.querySelectorAll('.item').forEach((el, i) => el.classList.toggle('active', i === current));
  document.getElementById('prev').disabled = current === 0;
  document.getElementById('next').disabled = current === DATA.samples.length - 1;
}

function renderRail() {
  const groups = { easy: [], medium: [], hard: [] };
  DATA.samples.forEach((s, i) => groups[s.difficulty].push(i));
  const titles = { easy: 'Easy — base already close', medium: 'Medium', hard: 'Hard — base struggles' };
  let html = '';
  for (const key of ['easy', 'medium', 'hard']) {
    if (!groups[key].length) continue;
    html += `<h2>${titles[key]}</h2>`;
    for (const i of groups[key]) {
      const s = DATA.samples[i];
      const be = s.base_error == null ? 'n/a' : s.base_error.toFixed(3);
      html += `<button class="item" data-i="${i}">
          <img src="${s.image}" alt="">
          <span class="meta">
            <span class="tk"><span class="dot ${diffColor[key]}"></span>${s.task}</span>
            <span class="err mono">base err ${be}</span>
          </span>
        </button>`;
    }
  }
  const rail = document.getElementById('rail');
  rail.innerHTML = html;
  rail.querySelectorAll('.item').forEach(el => {
    el.addEventListener('click', () => { current = +el.dataset.i; renderMain(); });
  });
}

document.getElementById('prev').addEventListener('click', () => { if (current > 0) { current--; renderMain(); } });
document.getElementById('next').addEventListener('click', () => { if (current < DATA.samples.length - 1) { current++; renderMain(); } });
document.getElementById('gt-toggle').addEventListener('change', e => { showGT = e.target.checked; renderMain(); });
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SUMMARY') return;
  if (e.key === 'ArrowLeft' && current > 0) { current--; renderMain(); }
  if (e.key === 'ArrowRight' && current < DATA.samples.length - 1) { current++; renderMain(); }
});

function renderOverview() {
  const d = DATA.dataset;
  const pct1 = v => (v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0) + '%';
  const kv = [
    ['Total samples', d.total.toLocaleString()],
    ['Source datasets', `${d.n_sources} (${d.top_sources.join(', ')}, …)`],
    ['Image sizes', `${d.n_sizes} distinct · mostly ${d.top_size}`],
    ['Waypoints / sample', `${d.wp_median} median (${d.wp_min}–${d.wp_max})`],
    ['Answer format', 'normalized [0,1], ≤10 points'],
    ['Held-out eval', `${d.eval.toLocaleString()} (fixed across runs)`],
    ['Train subsets', `500 · 1K · 2K · 5K · full (${d.full.toLocaleString()})`],
  ];
  document.getElementById('kv').innerHTML =
    kv.map(([k, v]) => `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`).join('');

  const rows = DATA.results;
  const maxPe = Math.max(...rows.map(r => r.pe_med));
  const num = v => v.toFixed(v >= 10 ? 1 : 3);
  document.getElementById('res-body').innerHTML = rows.map(r => {
    const cls = r.is_base ? 'base' : (r.is_full ? 'full' : '');
    const fill = r.is_base ? 'var(--hard)' : 'var(--pred)';
    return `<tr class="${cls}">
        <td>${r.label}</td>
        <td>${r.train == null ? '—' : r.train.toLocaleString()}</td>
        <td>${pct1(r.parse)}</td>
        <td>${pct1(r.in_range)}</td>
        <td>${num(r.pe_med)} <span class="paren">(${num(r.pe_mean)})</span></td>
        <td>${num(r.fr_med)} <span class="paren">(${num(r.fr_mean)})</span></td>
        <td class="barcell"><span class="bar"><span class="fill" style="width:${(100 * r.pe_med / maxPe).toFixed(1)}%;background:${fill}"></span></span></td>
      </tr>`;
  }).join('');

  document.getElementById('res-foot').innerHTML =
    'Point error and Fréchet are in normalized [0,1] image units, shown as <b>median (mean)</b>. '
    + 'The base model answers on a 0–1000 scale and is charitably rescaled; its means are still inflated by a few out-of-range outputs, so its <b>median</b> is the fair baseline. '
    + 'Bar = median point error relative to the base model (shorter is better).';
}

renderOverview();
renderRail();
renderMain();
</script>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--per-bucket", type=int, default=3, help="samples per difficulty tercile")
    p.add_argument("--max-image-px", type=int, default=512, help="downscale embedded images to this long side")
    p.add_argument("--out", type=Path, default=DEFAULT_EVAL_DIR / "prediction_explorer.html")
    args = p.parse_args()

    data = build_data(args.eval_dir, args.per_bucket, args.max_image_px)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    kb = len(html.encode()) / 1024
    print(f"{len(data['samples'])} samples × {len(data['models'])} models")
    print(f"wrote {args.out}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
