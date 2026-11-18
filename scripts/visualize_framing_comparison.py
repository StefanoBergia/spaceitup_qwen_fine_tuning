"""What did telling the model where it is, and where it is going, buy? Three prompt
framings side by side: legacy (round 2, "the goal is straight ahead"), goal (round 3,
the goal named in the prompt) and endpoints (round 4, entry point + goal named).

Login node (CPU-only), after the three rounds' eval sweeps have run:
    uv run scripts/visualize_framing_comparison.py
    uv run scripts/visualize_framing_comparison.py --out outputs/framing_comparison.html

Reads outputs/eval_habitat_v{2,3,4}/{habitat_base,habitat_train_full}/{metrics,predictions}.json
and the matching data/prepared_habitat_v*/{meta,eval}.json, and writes one self-contained
HTML page (inline CSS/JS, base64 images, no network requests).

The three arms were NOT measured on the same frames, and the page says so up front:
rounds 3 and 4 share one scene-disjoint 2,591-frame eval split (only the prompt differs,
so everything between them is paired), while round 2 was scored on its own 1,000-frame,
scene-leaked split whose goal is pinned at x = 0.5. Legacy numbers are therefore shown
against their own floors, never subtracted from the other two.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rover_vlm.consistency import start_edge
from rover_vlm.eval import (endpoint_spread, shape_correlation, to_goal_error,
                            trivial_baselines)
from rover_vlm.habitat_data import (HABITAT_PROMPT, HABITAT_PROMPT_ENDPOINTS,
                                    HABITAT_PROMPT_GOAL)
from rover_vlm.overlay import embed_jpeg, render_pair

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "framing_comparison.html"

# key, title, eval dir, data dir, prompt template, what the prompt hands over
ARMS = [
    ("legacy", "No goal given", "outputs/eval_habitat_v2", "data/prepared_habitat_v2",
     HABITAT_PROMPT, "nothing — the prompt asserts the goal is straight ahead"),
    ("goal", "Goal given", "outputs/eval_habitat_v3", "data/prepared_habitat_v3",
     HABITAT_PROMPT_GOAL, "the goal [x, y, v]"),
    ("endpoints", "Start and goal given", "outputs/eval_habitat_v4",
     "data/prepared_habitat_v4", HABITAT_PROMPT_ENDPOINTS,
     "the entry point [x, y, v] and the goal [x, y, v]"),
]
EDGE_ORDER = ["bottom", "left", "right"]
DETOUR_BUCKETS = [(0.0, 1.05, "near-straight"), (1.05, 1.15, "slight"),
                  (1.15, 1.35, "moderate"), (1.35, 1e9, "strong detour")]
COLORS = {"legacy": (213, 94, 0), "goal": (42, 120, 214), "endpoints": (0, 131, 0)}
GALLERY_COLORS = {"base": (137, 135, 129), "lora": (213, 94, 0)}  # arm colour = legacy orange
QUANTILES = np.linspace(0, 1, 201)


def load_run(eval_dir, tag):
    preds = json.loads((eval_dir / tag / "predictions.json").read_text())
    metrics = json.loads((eval_dir / tag / "metrics.json").read_text())
    return {"tag": tag, "metrics": metrics, "preds": {r["id"]: r for r in preds},
            "records": preds}


def scored(run):
    return [r for r in run["records"] if r.get("metrics") and "mean_point_error" in r["metrics"]]


def error_summary(run, n_total):
    """Distribution of per-sample waypoint error, plus the two 'did it use the image' tests.

    Frames the model failed to parse carry no error, so the mean here is over parsed
    frames only; the parse rate is reported next to it rather than folded in.
    """
    rs = scored(run)
    errs = np.array([r["metrics"]["mean_point_error"] for r in rs])
    srt = np.sort(errs)[::-1]
    k = max(1, len(errs) // 10)
    wins = [r["metrics"]["mean_point_error"] < to_goal_error(r["gt"]) for r in rs]
    m = run["metrics"]
    return {
        "n": n_total, "scored": len(rs), "parseRate": m["parse_rate"],
        "mean": float(errs.mean()), "median": float(np.median(errs)),
        "p90": float(np.percentile(errs, 90)), "p99": float(np.percentile(errs, 99)),
        "max": float(errs.max()), "worst10Share": float(srt[:k].sum() / srt.sum()),
        "winVsToGoal": float(np.mean(wins)),
        "frechetMedian": m["frechet_median"],
        "visAcc": m["path_visibility_acc_mean"],
        "goalErrMean": m["goal_point_error_mean"],
        "goalVisAcc": m["goal_visibility_accuracy"],
        "goalCopy": m.get("goal_copy_rate"), "startCopy": m.get("start_copy_rate"),
        "shapeCorr": shape_correlation(run["records"]),
        "endpoint": endpoint_spread(run["records"]),
        "quantiles": [float(q) for q in np.quantile(errs, QUANTILES)],
    }


def floors(run):
    gts = [r["gt"] for r in run["records"] if r.get("gt")]
    b = trivial_baselines(gts)
    return {k: {"mean": b[k]["mean_point_error_mean"], "median": b[k]["mean_point_error_median"]}
            for k in ("to_goal", "straight", "constant")}


def edge_slice(run):
    """Error by which border the ground-truth route enters from (from the GT path, so it
    works for the round-3 predictions that predate the entry_edge sidecar)."""
    groups = {}
    for r in scored(run):
        e = start_edge(r["gt"])
        if e not in EDGE_ORDER:
            continue
        g = groups.setdefault(e, {"err": [], "tg": [], "wrong": 0})
        g["err"].append(r["metrics"]["mean_point_error"])
        g["tg"].append(to_goal_error(r["gt"]))
        if r.get("parsed") and start_edge(r["parsed"]) != e:
            g["wrong"] += 1
    total = sum(len(g["err"]) for g in groups.values())
    out = []
    for key in EDGE_ORDER:
        g = groups.get(key)
        if not g:
            continue
        n = len(g["err"])
        out.append({"edge": key, "n": n, "share": n / total,
                    "mean": float(np.mean(g["err"])), "median": float(np.median(g["err"])),
                    "toGoal": float(np.mean(g["tg"])), "wrongEdge": g["wrong"] / n,
                    "winRate": float(np.mean([a < b for a, b in zip(g["err"], g["tg"])]))})
    return out


def detour_slice(run):
    """Error by how far the true route bows off the straight shot, in the same buckets as
    scripts/compare_evals.py (detour_ratio in the sidecar; absent from round 2, whose
    records carry no habitat_meta). Each bucket carries its own to_goal floor."""
    groups = {}
    for r in scored(run):
        d = (r.get("habitat_meta") or {}).get("detour_ratio")
        if d is None:
            continue
        for lo, hi, name in DETOUR_BUCKETS:
            if lo <= d < hi:
                groups.setdefault(name, []).append(r)
    if not groups:
        return None
    out = []
    for _, _, name in DETOUR_BUCKETS:
        rs = groups.get(name)
        if not rs:
            continue
        errs = [r["metrics"]["mean_point_error"] for r in rs]
        tg = [to_goal_error(r["gt"]) for r in rs]
        out.append({"detour": name, "n": len(rs), "mean": float(np.mean(errs)),
                    "median": float(np.median(errs)), "toGoal": float(np.mean(tg)),
                    "winRate": float(np.mean([a < b for a, b in zip(errs, tg)]))})
    return out


def paired_gallery(goal_run, end_run, eval_records, max_px, n_fixed=3, n_typical=3, n_worse=1):
    """The same frame under the goal prompt and the endpoints prompt — rounds 3 and 4
    share every eval frame, so this is the one place the comparison is literally paired."""
    by_id = {r["id"]: r for r in eval_records}
    rows = []
    for sid, r4 in end_run["preds"].items():
        r3 = goal_run["preds"].get(sid)
        if not (r3 and r3.get("metrics") and r4.get("metrics") and sid in by_id):
            continue
        rows.append((r3["metrics"]["mean_point_error"] - r4["metrics"]["mean_point_error"], sid))
    rows.sort(reverse=True)
    picks = [("fixed", "Largest gain from naming the start", rows[:n_fixed])]
    mid = len(rows) // 2
    picks.append(("typical", "Typical frame (median gain)", rows[mid - n_typical // 2: mid - n_typical // 2 + n_typical]))
    picks.append(("worse", "Where the endpoints prompt did worse", rows[-n_worse:]))
    out = []
    for key, title, pool in picks:
        for delta, sid in pool:
            path = Path(by_id[sid]["image"][0])
            if not path.exists():
                continue
            r3, r4 = goal_run["preds"][sid], end_run["preds"][sid]
            img = render_pair(path, r4["gt"], r3.get("parsed"), r4.get("parsed"),
                              COLORS["goal"], COLORS["endpoints"])
            out.append({"id": sid, "bucket": key, "bucketTitle": title,
                        "img": embed_jpeg(img, max_px * 2),
                        "errGoal": round(r3["metrics"]["mean_point_error"], 3),
                        "errEnd": round(r4["metrics"]["mean_point_error"], 3),
                        "toGoal": round(to_goal_error(r4["gt"]), 3),
                        "gtEdge": start_edge(r4["gt"]),
                        "goalEdge": start_edge(r3["parsed"]) if r3.get("parsed") else "none"})
    return out


def legacy_gallery(base_run, lora_run, eval_records, max_px, per_bucket=2):
    """Round 2 frames, base vs LoRA, at the LoRA model's easy / median / hard terciles."""
    by_id = {r["id"]: r for r in eval_records}
    rows = sorted((r["metrics"]["mean_point_error"], sid) for sid, r in lora_run["preds"].items()
                  if r.get("metrics") and sid in by_id)
    n = len(rows)
    third = n // 3
    buckets = [("easy", "Easy", rows[:third]), ("medium", "Medium", rows[third:2 * third]),
               ("hard", "Hard", rows[2 * third:])]
    out = []
    for key, title, pool in buckets:
        step = max(1, len(pool) // per_bucket)
        for err, sid in pool[::step][:per_bucket]:
            path = Path(by_id[sid]["image"][0])
            if not path.exists():
                continue
            lr, br = lora_run["preds"][sid], base_run["preds"].get(sid)
            img = render_pair(path, lr["gt"], br.get("parsed") if br else None, lr.get("parsed"),
                              GALLERY_COLORS["base"], GALLERY_COLORS["lora"])
            out.append({"id": sid, "bucket": key, "bucketTitle": title,
                        "img": embed_jpeg(img, max_px * 2), "err": round(err, 3),
                        "baseParsed": bool(br and br.get("parsed"))})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-image-px", type=int, default=380)
    args = p.parse_args()

    arms = []
    runs = {}
    for key, title, eval_dir, data_dir, prompt, given in ARMS:
        eval_dir, data_dir = REPO_ROOT / eval_dir, REPO_ROOT / data_dir
        meta = json.loads((data_dir / "meta.json").read_text())
        base = load_run(eval_dir, "habitat_base")
        lora = load_run(eval_dir, "habitat_train_full")
        n_eval = meta["splits"]["eval"]
        runs[key] = (base, lora, data_dir)
        arms.append({
            "key": key, "title": title, "given": given,
            "prompt": prompt.replace("<image>\n", ""),
            "trainFull": meta["splits"]["train_full"], "nEval": n_eval,
            "sceneDisjoint": "manifests" in meta,
            "yawAugmented": "manifests" in meta,
            "base": error_summary(base, n_eval), "lora": error_summary(lora, n_eval),
            "floors": floors(lora),
            "edges": edge_slice(lora), "baseEdges": edge_slice(base),
            "detour": detour_slice(lora),
        })

    goal_base, goal_lora, _ = runs["goal"]
    end_base, end_lora, end_data = runs["endpoints"]
    leg_base, leg_lora, leg_data = runs["legacy"]
    eval_end = json.loads((end_data / "eval.json").read_text())
    eval_leg = json.loads((leg_data / "eval.json").read_text())
    shared = set(goal_lora["preds"]) & set(end_lora["preds"])

    data = {
        "arms": arms,
        "sharedFrames": len(shared),
        "quantileGrid": [float(q) for q in QUANTILES],
        "pairedGallery": paired_gallery(goal_lora, end_lora, eval_end, args.max_image_px),
        "legacyGallery": legacy_gallery(leg_base, leg_lora, eval_leg, args.max_image_px),
    }

    html = (Path(__file__).parent / "_framing_report.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB, "
          f"{len(data['pairedGallery'])} paired + {len(data['legacyGallery'])} legacy frames)")


if __name__ == "__main__":
    main()
