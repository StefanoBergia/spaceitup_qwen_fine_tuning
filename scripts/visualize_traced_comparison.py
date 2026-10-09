"""Self-contained HTML report for the trace-augmented fine-tune: 2B vs 0.8B, base vs LoRA,
with per-sample reasoning inspection.

Login node (CPU-only), after slurm/train_traced_both.sbatch has produced the eval trees:
    uv run scripts/visualize_traced_comparison.py
    uv run scripts/visualize_traced_comparison.py --per-bucket 4

Reads the two traced eval trees (each with a `habitat_base` and a `habitat_train_full_traced`
tag) plus the eval split for the id->image lookup, and writes one HTML file — no network.

Unlike the scaling report (scripts/visualize_habitat_results.py), this is a single-size
comparison of the two model sizes AND it surfaces the model's generated <think> reasoning:
each gallery card shows the frame with both models' predicted paths over the ground truth,
next to the reasoning each size produced, its waypoints, and its per-sample metrics.

It also carries the **plain (no-reasoning) LoRA** as a baseline in every comparison, because
the decisive question is not traced-vs-zero-shot-base but traced-vs-plain-SFT: on this task
the plain fine-tune is significantly BETTER, so the report shows base / plain / traced side by
side, tests plain-vs-traced per size, and annotates each gallery frame with the plain error.

Why a new script: the tag `habitat_train_full_traced` is not on the size-scaling curve, so
compare_evals.py skips it and visualize_habitat_results.py (which keys off `...train_full`)
does not pick it up; and no existing report renders the `generated` reasoning text.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from rover_vlm.trace_eval import reasoning_and_answer
from rover_vlm.compare import paired_bootstrap
from rover_vlm.consistency import aggregate_consistency, per_image_consistency
from rover_vlm.overlay import embed_jpeg, render_pair
from rover_vlm.trace_eval import (
    aggregate_trace_quality,
    claims_occlusion,
    goal_occluded,
    rouge_l,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
A_DIR = REPO_ROOT / "outputs" / "eval_habitat_v2_traced"
B_DIR = REPO_ROOT / "outputs" / "eval_habitat_v2_traced_0.8b"
A_PLAIN_DIR = REPO_ROOT / "outputs" / "eval_habitat_v2"        # plain (no-reasoning) LoRA, 2B
B_PLAIN_DIR = REPO_ROOT / "outputs" / "eval_habitat_v2_0.8b"   # plain (no-reasoning) LoRA, 0.8B
EVAL_FILE = REPO_ROOT / "data" / "prepared_habitat_v2" / "eval.json"
# Cosmos3 teacher traces for the eval frames — the reference for distillation fidelity.
REF_TRACES = REPO_ROOT / "outputs" / "traces_full" / "path_v6_eval.filtered.jsonl"

BASE_TAG = "habitat_base"
PLAIN_TAG = "habitat_train_full"          # plain answer-only LoRA on the same v2 split
TRACED_TAG = "habitat_train_full_traced"

COLOR_A, COLOR_B = (42, 120, 214), (230, 97, 0)  # 2B blue, 0.8B orange

# key, display name, one-line definition (rendered as the column infobox), better direction
TABLE_COLS = [
    ("parse_rate", "Parse rate",
     "Fraction of outputs that parse into a valid {path, goal}.", "up"),
    ("mean_point_error_median", "Median waypoint err",
     "Median distance from predicted to ground-truth waypoints, normalized image coords.", "down"),
    ("frechet_median", "Fréchet (median)",
     "Median Fréchet distance — overall path-shape mismatch, not just point spacing.", "down"),
    ("path_visibility_acc_mean", "Waypoint vis. acc",
     "Mean per-waypoint accuracy of the visible/obstructed flag.", "up"),
    ("goal_point_error_median", "Goal err (median)",
     "Median distance from the predicted goal to the ground-truth goal.", "down"),
    ("goal_visibility_accuracy", "Goal vis. acc",
     "Accuracy of the goal's visible-vs-obstructed call.", "up"),
]


def load_run(eval_dir, tag):
    """Return (metrics dict or None, {id: prediction record})."""
    mpath, ppath = eval_dir / tag / "metrics.json", eval_dir / tag / "predictions.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else None
    preds = {}
    if ppath.exists():
        preds = {r["id"]: r for r in json.loads(ppath.read_text())}
    return metrics, preds


def load_judges(eval_dir, tag):
    """All VLM-judge aggregates for this run, keyed by judge_model. scripts/judge_traces.py now
    writes <tag>/judge_<slug>/judge_metrics.json (one dir per judge); older runs wrote a single
    <tag>/judge_metrics.json — both are picked up, so the pre-namespacing Cosmos run still shows
    up with zero re-run. Returns {judge_model: aggregate dict}, deduped by judge_model."""
    out = {}
    base = eval_dir / tag
    paths = sorted(base.glob("judge_*/judge_metrics.json"))
    legacy = base / "judge_metrics.json"
    if legacy.exists():
        paths.append(legacy)
    for p in paths:
        agg = json.loads(p.read_text())
        model = agg.get("judge_model") or "unknown"
        out.setdefault(model, agg)  # namespaced dirs (sorted first) win over the legacy file
    return out


def collect_judges(a_dir, b_dir, traced_tag, base_tag):
    """One entry per distinct judge model, each carrying its four aggregates (a/b × traced/base)
    so the report can render a separate table per judge. Sorted by model id for a stable order."""
    at, bt = load_judges(a_dir, traced_tag), load_judges(b_dir, traced_tag)
    ab, bb = load_judges(a_dir, base_tag), load_judges(b_dir, base_tag)
    models = sorted(set(at) | set(bt) | set(ab) | set(bb))
    return [{"model": m, "aJudge": at.get(m), "bJudge": bt.get(m),
             "aJudgeBase": ab.get(m), "bJudgeBase": bb.get(m)} for m in models]


def load_seeds(eval_dir, tag):
    """Sampled draws for a run: every <tag>/seeds/seed<k>/predictions.json written by
    scripts/evaluate.py --seed k, as [{"seed": k, "preds": {id: rec}, "decoding": {...}}]
    in numeric seed order. Empty list when the run was never sampled."""
    out = []
    for sd in (eval_dir / tag / "seeds").glob("seed*"):
        ppath = sd / "predictions.json"
        m = re.fullmatch(r"seed(\d+)", sd.name)
        if not (m and ppath.exists()):
            continue
        mpath = sd / "metrics.json"
        decoding = (json.loads(mpath.read_text()).get("decoding") or {}) if mpath.exists() else {}
        out.append({"seed": int(m.group(1)),
                    "preds": {r["id"]: r for r in json.loads(ppath.read_text())},
                    "decoding": decoding})
    return sorted(out, key=lambda s: s["seed"])


def _consistency_for(eval_dir, tag):
    """(aggregate, per-image, n_seeds, decoding) for one run, or None if it has no seeds."""
    seeds = load_seeds(eval_dir, tag)
    if not seeds:
        return None
    _, greedy = load_run(eval_dir, tag)
    per_image = per_image_consistency([s["preds"] for s in seeds], greedy)
    return aggregate_consistency(per_image), per_image, len(seeds), seeds[0]["decoding"]


def consistency_block(plain_dir, plain_tag, traced_dir, traced_tag):
    """Sampling consistency for one model size: plain vs traced aggregates plus a paired
    bootstrap on per-image path spread (lower = more stable), over the images where both
    have a spread. None when neither run was sampled; a missing side is None and the
    test is skipped. `perImagePlain` / `perImageTraced` are {id: per-image row} for the
    gallery, and are stripped before the payload is written."""
    plain, traced = _consistency_for(plain_dir, plain_tag), _consistency_for(traced_dir, traced_tag)
    if plain is None and traced is None:
        return None
    test = None
    if plain and traced:
        as_recs = lambda per: {i: {"metrics": {"path_spread": r["path_spread"]}}
                               for i, r in per.items() if r["path_spread"] is not None}
        s = paired_bootstrap(as_recs(plain[1]), as_recs(traced[1]), "path_spread")
        if s:
            test = {**s, "aName": "plain SFT", "bName": "traced",
                    "significant": s["lo"] > 0 or s["hi"] < 0,
                    "better": "plain SFT" if s["diff"] < 0 else "traced"}
    src = traced or plain
    return {"nSeeds": src[2], "decoding": src[3],
            "plain": plain[0] if plain else None, "traced": traced[0] if traced else None,
            "test": test,
            "perImagePlain": plain[1] if plain else None,
            "perImageTraced": traced[1] if traced else None}


def load_ref_traces(path):
    """{id: Cosmos3 teacher trace} from a filtered labelling file, kept rows only."""
    out = {}
    if path and Path(path).exists():
        for line in Path(path).open():
            if line.strip():
                r = json.loads(line)
                if r.get("keep") and r.get("trace"):
                    out[r["id"]] = r["trace"]
    return out


def trace_quality(preds, refs):
    """Reasoning-trace quality for one model vs the Cosmos reference. Returns (aggregate,
    {id: rouge_f1}). ROUGE-L is distillation fidelity; occlusion grounding is scored against
    the GT goal-visibility flag; the ROUGE-vs-error correlation says whether more teacher-like
    reasoning tracks a better answer (near 0 => the reasoning is decorative)."""
    recs, per_id, pairs = [], {}, []
    for sid, p in preds.items():
        ref = refs.get(sid)
        if not ref:
            continue
        reasoning, _ = reasoning_and_answer(p.get("generated", ""))
        f1 = rouge_l(reasoning, ref)["f1"]
        per_id[sid] = f1
        recs.append({"rouge_f1": f1, "occ_claim": claims_occlusion(reasoning),
                     "gt_occ": goal_occluded(p.get("gt"))})
        if p.get("metrics"):
            pairs.append((f1, p["metrics"]["mean_point_error"]))
    agg = aggregate_trace_quality(recs)
    if len(pairs) > 2:
        agg["corr_rouge_err"] = float(
            np.corrcoef([x for x, _ in pairs], [y for _, y in pairs])[0, 1])
    return agg, per_id


VARIANTS = ("base", "plain", "traced")


def build_table(a_label, b_label, a_metrics, b_metrics):
    """Six rows — each model's {base, plain, traced} — one cell per TABLE_COLS key.

    `*_metrics` is a dict {variant: metrics.json dict or None}. base = zero-shot,
    plain = the no-reasoning LoRA, traced = the reasoning LoRA."""
    def row(model, variant, metrics):
        return {"model": model, "variant": variant, "isBase": variant == "base",
                **{k: (metrics.get(k) if metrics else None) for k, _, _, _ in TABLE_COLS}}
    rows = []
    for model, mset in ((a_label, a_metrics), (b_label, b_metrics)):
        rows += [row(model, v, mset.get(v)) for v in VARIANTS]
    return rows


def paired_test(title, a_pred, b_pred, a_name, b_name, lower_better=True):
    """One plain-language paired-bootstrap verdict on mean waypoint error, or None if the
    two runs share no jointly-parsed frames. `lower_better` flips which side a negative
    (a-b) difference favours; for error, lower a means a wins."""
    s = paired_bootstrap(a_pred, b_pred, "mean_point_error")
    if not s:
        return None
    a_wins = (s["diff"] < 0) if lower_better else (s["diff"] > 0)
    return {"title": title, "aName": a_name, "bName": b_name,
            "a": s["a"], "b": s["b"], "diff": s["diff"], "lo": s["lo"], "hi": s["hi"],
            "n": s["n"], "significant": s["lo"] > 0 or s["hi"] < 0,
            "better": a_name if a_wins else b_name}


def _pred_summary(rec):
    """Compact view of a prediction record for a gallery card, or None if unparsed."""
    if not rec or not rec.get("parsed"):
        return None
    reasoning, answer = reasoning_and_answer(rec.get("generated", ""))
    m = rec.get("metrics") or {}
    return {
        "reasoning": reasoning,
        "answer": answer,
        "nWaypoints": len(rec["parsed"].get("path", [])),
        "goal": rec["parsed"].get("goal"),
        "err": round(m["mean_point_error"], 3) if "mean_point_error" in m else None,
        "frechet": round(m["frechet"], 3) if "frechet" in m else None,
        "vis": round(m["path_visibility_acc"], 2) if "path_visibility_acc" in m else None,
        "goalOk": bool(m.get("goal_visibility_correct")),
    }


def _plain_err(preds, sid):
    r = preds.get(sid)
    if r and r.get("metrics") and "mean_point_error" in r["metrics"]:
        return round(r["metrics"]["mean_point_error"], 3)
    return None


def _draws(seeds, sid):
    """Parsed prediction per sampled draw for one image (None for unparseable draws)."""
    return [(s["preds"].get(sid) or {}).get("parsed") for s in seeds]


def _spread_fields(per_image, sid, n_seeds):
    row = (per_image or {}).get(sid)
    return {"spread": (round(row["path_spread"], 3)
                       if row and row["path_spread"] is not None else None),
            "nDraws": n_seeds if per_image else None}


def build_gallery(a_traced, b_traced, a_plain, b_plain, refs, a_rouge, b_rouge,
                  id_to_image, a_label, per_bucket, max_px,
                  a_seeds=(), b_seeds=(), a_per_image=None, b_per_image=None):
    """Rank eval ids into easy/medium/hard terciles by the A traced waypoint error, pick
    `per_bucket` from each, and render both models' reasoning + paths for the same frame. Each
    model's per-sample PLAIN error and ROUGE-L vs the Cosmos reference are attached, and the
    reference trace itself is shown, so a card is a full student-vs-teacher-vs-truth view.

    `*_seeds` (load_seeds output for the traced runs) draw each sampled path faintly under
    the greedy one, and `*_per_image` supplies that frame's spread number."""
    scored = sorted((r["metrics"]["mean_point_error"], sid)
                    for sid, r in a_traced.items()
                    if r.get("metrics") and sid in id_to_image)
    n = len(scored)
    third = n // 3
    buckets = [("easy", f"Easy — lowest {a_label} traced waypoint error", scored[:third]),
               ("medium", "Medium", scored[third:2 * third]),
               ("hard", f"Hard — highest {a_label} traced waypoint error", scored[2 * third:])]

    out = []
    for key, title, pool in buckets:
        if not pool:
            continue
        step = max(1, len(pool) // per_bucket)
        for err, sid in pool[::step][:per_bucket]:
            path = Path(id_to_image[sid])
            if not path.exists():
                continue
            ar, br = a_traced.get(sid), b_traced.get(sid)
            gt = (ar or br or {}).get("gt")
            img = render_pair(path, gt,
                              ar.get("parsed") if ar else None,
                              br.get("parsed") if br else None,
                              COLOR_A, COLOR_B,
                              samples_left=_draws(a_seeds, sid), samples_right=_draws(b_seeds, sid))
            sa, sb = _pred_summary(ar), _pred_summary(br)
            if sa:
                sa["plainErr"] = _plain_err(a_plain, sid)
                sa["rouge"] = round(a_rouge[sid], 3) if sid in a_rouge else None
                sa.update(_spread_fields(a_per_image, sid, len(a_seeds)))
            if sb:
                sb["plainErr"] = _plain_err(b_plain, sid)
                sb["rouge"] = round(b_rouge[sid], 3) if sid in b_rouge else None
                sb.update(_spread_fields(b_per_image, sid, len(b_seeds)))
            out.append({
                "id": sid, "bucket": key, "bucketTitle": title,
                "img": embed_jpeg(img, max_px * 2),
                "gtGoalVisible": bool(gt and gt.get("goal") and gt["goal"][2] == 1),
                "ref": refs.get(sid),         # Cosmos teacher trace, the ROUGE reference
                "a": sa, "b": sb,
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-dir", type=Path, default=A_DIR)
    p.add_argument("--b-dir", type=Path, default=B_DIR)
    p.add_argument("--a-plain-dir", type=Path, default=A_PLAIN_DIR,
                   help="plain (no-reasoning) LoRA eval tree for model A")
    p.add_argument("--b-plain-dir", type=Path, default=B_PLAIN_DIR)
    p.add_argument("--a-label", default="2B")
    p.add_argument("--b-label", default="0.8B")
    p.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    p.add_argument("--ref-traces", type=Path, default=REF_TRACES,
                   help="Cosmos3 teacher traces for the eval frames (ROUGE-L reference)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--per-bucket", type=int, default=3)
    p.add_argument("--max-image-px", type=int, default=480)
    args = p.parse_args()

    a_base_m, _ = load_run(args.a_dir, BASE_TAG)
    a_traced_m, a_traced_p = load_run(args.a_dir, TRACED_TAG)
    b_base_m, _ = load_run(args.b_dir, BASE_TAG)
    b_traced_m, b_traced_p = load_run(args.b_dir, TRACED_TAG)
    a_plain_m, a_plain_p = load_run(args.a_plain_dir, PLAIN_TAG)
    b_plain_m, b_plain_p = load_run(args.b_plain_dir, PLAIN_TAG)
    if not (a_traced_m and b_traced_m):
        raise SystemExit(f"missing traced metrics under {args.a_dir} / {args.b_dir} — "
                         "run slurm/train_traced_both.sbatch first")
    for label, m, d in ((args.a_label, a_plain_m, args.a_plain_dir),
                        (args.b_label, b_plain_m, args.b_plain_dir)):
        if m is None:
            print(f"  note: no plain (no-reasoning) run for {label} under {d} — "
                  "its baseline column and plain-vs-traced test will be blank")

    eval_records = json.loads(args.eval_file.read_text())
    id_to_image = {r["id"]: r["image"][0] for r in eval_records}

    # Reasoning-trace quality (cheap tier): distillation fidelity (ROUGE-L vs the Cosmos
    # teacher) + occlusion grounding vs the GT flag, per traced model.
    refs = load_ref_traces(args.ref_traces)
    a_tq, a_rouge = trace_quality(a_traced_p, refs)
    b_tq, b_rouge = trace_quality(b_traced_p, refs)
    if not refs:
        print(f"  note: no reference traces at {args.ref_traces} — ROUGE section will be blank")

    # The comparisons that matter, each a paired bootstrap on waypoint error over jointly
    # parsed frames. The headline is reasoning-vs-plain PER SIZE (does the trace help?);
    # then the cross-size check on the traced adapters.
    tests = [t for t in (
        paired_test(f"{args.a_label}: does reasoning help? (plain SFT vs traced)",
                    a_plain_p, a_traced_p, "plain SFT", "traced"),
        paired_test(f"{args.b_label}: does reasoning help? (plain SFT vs traced)",
                    b_plain_p, b_traced_p, "plain SFT", "traced"),
        paired_test(f"Model size, both traced ({args.a_label} vs {args.b_label})",
                    a_traced_p, b_traced_p, args.a_label, args.b_label),
    ) if t]

    # Sampling consistency: how much each model's answer moves between sampled draws of the
    # same image (slurm/eval_seeds.sbatch). Optional — None per size until seeds exist.
    a_cons = consistency_block(args.a_plain_dir, PLAIN_TAG, args.a_dir, TRACED_TAG)
    b_cons = consistency_block(args.b_plain_dir, PLAIN_TAG, args.b_dir, TRACED_TAG)
    a_seeds, b_seeds = load_seeds(args.a_dir, TRACED_TAG), load_seeds(args.b_dir, TRACED_TAG)
    strip = lambda blk: ({k: v for k, v in blk.items() if not k.startswith("perImage")}
                         if blk else None)

    data = {
        "aLabel": args.a_label, "bLabel": args.b_label,
        "nEval": a_traced_m["num_samples"],
        "aModelId": a_traced_m.get("model_id"), "bModelId": b_traced_m.get("model_id"),
        "tableCols": [{"key": k, "name": nm, "def": d, "better": bt}
                      for k, nm, d, bt in TABLE_COLS],
        "table": build_table(args.a_label, args.b_label,
                             {"base": a_base_m, "plain": a_plain_m, "traced": a_traced_m},
                             {"base": b_base_m, "plain": b_plain_m, "traced": b_traced_m}),
        "tests": tests,
        "traceQuality": {"hasRef": bool(refs), "a": a_tq, "b": b_tq,
                         "judges": collect_judges(args.a_dir, args.b_dir, TRACED_TAG, BASE_TAG)},
        "consistency": {"a": strip(a_cons), "b": strip(b_cons)},
        "gallery": build_gallery(a_traced_p, b_traced_p, a_plain_p, b_plain_p,
                                 refs, a_rouge, b_rouge,
                                 id_to_image, args.a_label, args.per_bucket, args.max_image_px,
                                 a_seeds, b_seeds,
                                 a_cons and a_cons["perImageTraced"],
                                 b_cons and b_cons["perImageTraced"]),
    }

    out = args.out or (args.a_dir / "traced_comparison.html")
    template = (Path(__file__).parent / "_traced_comparison_report.html").read_text()
    # Escape </ inside the injected JSON: the payload embeds model-generated reasoning, which
    # can contain a literal </script> that would otherwise close the block early.
    payload = json.dumps(data).replace("</", "<\\/")
    html = template.replace("/*__DATA__*/null", payload)
    html = re.sub(r"<title>.*?</title>",
                  f"<title>Traced fine-tune — {args.a_label} vs {args.b_label} "
                  f"({data['nEval']:,} eval)</title>", html, count=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(data['gallery'])} gallery frames)")


if __name__ == "__main__":
    main()
