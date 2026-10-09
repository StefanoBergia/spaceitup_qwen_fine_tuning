"""Build a self-contained HTML report for the Habitat path + visibility results.

Login node (CPU-only), after the eval sweep has run:
    uv run scripts/visualize_habitat_results.py
    uv run scripts/visualize_habitat_results.py \\
        --eval-dir outputs/eval_habitat_0.8b --label Qwen3.5-0.8B

Reads outputs/eval_habitat/<tag>/{metrics,predictions}.json plus the prepared splits and
writes <eval-dir>/habitat_results.html — one file, no network requests.

Sections: headline tiles, scaling curves per metric, the metrics table, the
goal-visibility class-imbalance callout, and a base-vs-LoRA gallery grouped by how hard
the frame was for the fine-tuned model.

Replaces a pair of throwaway scratchpad scripts that produced the original version of
this page and were never committed, so it could not be regenerated.
"""

import argparse
import json
import re
from pathlib import Path

from rover_vlm.consistency import start_edge
from rover_vlm.eval import goal_visibility_confusion, to_goal_error, trivial_baselines
from rover_vlm.overlay import embed_jpeg, render_pair

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "outputs" / "eval_habitat"
DATA_DIR = REPO_ROOT / "data" / "prepared_habitat"

COLOR_BASE, COLOR_LORA = (213, 94, 0), (42, 120, 214)

SIZES = [("habitat_base", "base", 0), ("habitat_train_500", "500", 500),
         ("habitat_train_1000", "1K", 1000), ("habitat_train_2000", "2K", 2000),
         ("habitat_train_full", "full", None)]

METRICS = [
    ("mean_point_error_median", "Waypoint position error", "median normalized distance", "down"),
    ("frechet_median", "Trajectory shape error (Fréchet)", "median normalized distance", "down"),
    ("path_visibility_acc_mean", "Per-waypoint visibility accuracy", "accuracy", "up"),
    ("goal_balanced", "Goal visibility (balanced accuracy)", "balanced accuracy", "up"),
]

TABLE_COLS = [
    ("parse_rate", "Parse rate"),
    ("mean_point_error_mean", "Mean point err"),
    ("mean_point_error_median", "Median point err"),
    ("frechet_median", "Fréchet (median)"),
    ("path_visibility_acc_mean", "Waypoint vis. acc"),
    ("goal_point_error_median", "Goal err"),
    ("goal_visibility_accuracy", "Goal vis. acc (raw)"),
    ("goal_balanced", "Goal vis. acc (balanced)"),
]


def load(eval_dir, data_dir):
    meta = json.loads((data_dir / "meta.json").read_text())
    runs = []
    for tag, label, size in SIZES:
        mpath = eval_dir / tag / "metrics.json"
        if not mpath.exists():
            print(f"  note: {tag} missing, skipping")
            continue
        metrics = json.loads(mpath.read_text())
        preds = json.loads((eval_dir / tag / "predictions.json").read_text())
        conf = goal_visibility_confusion(preds)
        metrics["goal_balanced"] = conf["balanced_accuracy"]
        runs.append({"tag": tag, "label": label,
                     "size": meta["splits"]["train_full"] if size is None else size,
                     "metrics": metrics, "conf": conf,
                     "preds": {r["id"]: r for r in preds}})
    if not runs:
        raise SystemExit(f"no metrics under {eval_dir}/<tag>/metrics.json — run the evals first")
    return meta, runs


EDGE_ORDER = [("bottom", "Bottom — the rover's own position"),
              ("left", "Left edge — rover off-frame"),
              ("right", "Right edge — rover off-frame")]


def build_baselines(run):
    """What an image-blind predictor scores, and how often the model actually beats it.

    Once the goal is handed over in the prompt, `to_goal` -- the straight line to that
    goal -- is the only honest floor, and the per-sample win rate is the only honest
    summary: a handful of catastrophic misses drag the set mean above a baseline the
    model beats on most frames.
    """
    scored = [r for r in run["preds"].values() if r.get("metrics") and r.get("gt")]
    if not scored:
        return None
    base = trivial_baselines([r["gt"] for r in scored])
    wins = [r["metrics"]["mean_point_error"] < to_goal_error(r["gt"]) for r in scored]
    return {
        "n": len(scored),
        "winRate": sum(wins) / len(wins),
        "modelMean": sum(r["metrics"]["mean_point_error"] for r in scored) / len(scored),
        "modelMedian": sorted(r["metrics"]["mean_point_error"] for r in scored)[len(scored) // 2],
        "floors": [{"key": k, "mean": base[k]["mean_point_error_mean"],
                    "median": base[k]["mean_point_error_median"]}
                   for k in ("to_goal", "straight", "constant") if k in base],
    }


def build_edges(run):
    """Error split by which frame border the route enters from.

    The edge comes from the *ground truth* path (rover_vlm.consistency.start_edge), not
    the prepared sidecar, so this works for every round including the ones prepared
    before entry_edge was recorded.
    """
    groups = {}
    for r in run["preds"].values():
        if not (r.get("metrics") and r.get("gt")):
            continue
        e = start_edge(r["gt"])
        if e not in dict(EDGE_ORDER):
            continue
        g = groups.setdefault(e, {"err": [], "tg": [], "wrong": 0})
        g["err"].append(r["metrics"]["mean_point_error"])
        g["tg"].append(to_goal_error(r["gt"]))
        if r.get("parsed") and start_edge(r["parsed"]) != e:
            g["wrong"] += 1
    if not groups:
        return None
    total = sum(len(g["err"]) for g in groups.values())
    out = []
    for key, title in EDGE_ORDER:
        g = groups.get(key)
        if not g:
            continue
        n = len(g["err"])
        out.append({
            "key": key, "title": title, "n": n, "share": n / total,
            "mean": sum(g["err"]) / n,
            "median": sorted(g["err"])[n // 2],
            "toGoal": sum(g["tg"]) / n,
            "wrongEdge": g["wrong"] / n,
            "winRate": sum(a < b for a, b in zip(g["err"], g["tg"])) / n,
        })
    return out


def build_wrong_edge_gallery(base_run, full_run, eval_records, limit, max_px):
    """Frames where the model started the route from the wrong border.

    The single most expensive failure in round 3 and the reason round 4 hands the entry
    point over: the shape is usually fine, it is simply drawn from the opposite side.
    """
    by_id = {r["id"]: r for r in eval_records}
    cands = []
    for sid, r in full_run["preds"].items():
        if not (r.get("metrics") and r.get("parsed") and sid in by_id):
            continue
        gt_e, pred_e = start_edge(r["gt"]), start_edge(r["parsed"])
        if gt_e in dict(EDGE_ORDER) and pred_e != gt_e:
            cands.append((-r["metrics"]["mean_point_error"], sid, gt_e, pred_e))
    cands.sort()
    out = []
    for negerr, sid, gt_e, pred_e in cands[:limit]:
        path = Path(by_id[sid]["image"][0])
        if not path.exists():
            continue
        fr = full_run["preds"][sid]
        br = base_run["preds"].get(sid) if base_run else None
        img = render_pair(path, fr["gt"], br.get("parsed") if br else None,
                          fr.get("parsed"), COLOR_BASE, COLOR_LORA)
        out.append({"id": sid, "img": embed_jpeg(img, max_px * 2),
                    "err": round(-negerr, 3), "gtEdge": gt_e, "predEdge": pred_e,
                    "toGoal": round(to_goal_error(fr["gt"]), 3)})
    return out


def build_gallery(base_run, full_run, eval_records, per_bucket, max_px):
    """Frames split into terciles by how hard they were for the fine-tuned model.

    Difficulty is ranked by the LoRA model's own waypoint error, not the base model's:
    the base output is mostly unusable here, so it carries no difficulty signal. Frames
    where the base failed to parse are kept — its panel then shows ground truth alone,
    which is itself the result.
    """
    by_id = {r["id"]: r for r in eval_records}
    scored = [(r["metrics"]["mean_point_error"], sid)
              for sid, r in full_run["preds"].items()
              if r.get("metrics") and sid in by_id]
    scored.sort()
    n = len(scored)
    third = n // 3
    buckets = [("easy", "Easy — the fine-tuned model nails it", scored[:third]),
               ("medium", "Medium", scored[third:2 * third]),
               ("hard", "Hard — even the fine-tuned model struggles", scored[2 * third:])]

    out = []
    for key, title, pool in buckets:
        if not pool:
            continue
        step = max(1, len(pool) // per_bucket)
        for err, sid in pool[::step][:per_bucket]:
            path = Path(by_id[sid]["image"][0])
            if not path.exists():
                continue
            fr = full_run["preds"][sid]
            br = base_run["preds"].get(sid) if base_run else None
            img = render_pair(path, fr["gt"], br.get("parsed") if br else None,
                              fr.get("parsed"), COLOR_BASE, COLOR_LORA)
            gt_goal = fr["gt"]["goal"][2]
            out.append({
                "id": sid, "bucket": key, "bucketTitle": title,
                "img": embed_jpeg(img, max_px * 2),
                "err": round(err, 3),
                "frechet": round(fr["metrics"]["frechet"], 3),
                "vis": round(fr["metrics"]["path_visibility_acc"], 2),
                "goalOk": bool(fr["metrics"]["goal_visibility_correct"]),
                "goalVisible": gt_goal == 1,
                "baseParsed": bool(br and br.get("parsed")),
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--label", default="Qwen3.5-2B", help="base model name shown in the page")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--per-bucket", type=int, default=4)
    p.add_argument("--max-image-px", type=int, default=380)
    p.add_argument("--title", default=None,
                   help="page <title>, which is what names the published artifact; "
                        "defaults to model + split sizes so two rounds stay distinguishable")
    p.add_argument("--wrong-edge", type=int, default=4,
                   help="frames to show in the wrong-start-edge gallery (0 to omit)")
    args = p.parse_args()

    meta, runs = load(args.eval_dir, args.data_dir)
    eval_records = json.loads((args.data_dir / "eval.json").read_text())
    base = next((r for r in runs if r["tag"].endswith("base")), None)
    full = next((r for r in runs if r["tag"].endswith("train_full")), None)
    if full is None:
        raise SystemExit("the full-data run is missing — nothing to compare the base against")

    data = {
        "label": args.label,
        # what the prompt handed over; older preps recorded goal_in_prompt or nothing.
        # The floors section reads differently when the model had to infer the goal.
        "framing": meta.get("framing") or ("goal" if meta.get("goal_in_prompt") else "legacy"),
        "nEval": full["metrics"]["num_samples"],
        "trainFull": meta["splits"]["train_full"],
        "metrics": [{"key": k, "name": n, "unit": u, "better": b} for k, n, u, b in METRICS],
        "scaling": [{"label": r["label"], "size": r["size"],
                     **{k: r["metrics"].get(k) for k, _, _, _ in METRICS}} for r in runs],
        "table": [{"label": r["label"], "isBase": r["tag"].endswith("base"),
                   "size": "—" if r["size"] == 0 else f"{r['size']:,}",
                   **{k: r["metrics"].get(k) for k, _ in TABLE_COLS}} for r in runs],
        "tableCols": [{"key": k, "name": n} for k, n in TABLE_COLS],
        "conf": {r["label"]: r["conf"] for r in runs},
        "gallery": build_gallery(base, full, eval_records, args.per_bucket, args.max_image_px),
        "baseMetrics": base["metrics"] if base else None,
        "fullMetrics": full["metrics"],
        "baselines": build_baselines(full),
        "baseBaselines": build_baselines(base) if base else None,
        "edges": build_edges(full),
        "wrongEdge": (build_wrong_edge_gallery(base, full, eval_records,
                                               args.wrong_edge, args.max_image_px)
                      if args.wrong_edge else []),
    }

    out = args.out or (args.eval_dir / "habitat_results.html")
    html = (Path(__file__).parent / "_habitat_report.html").read_text()
    out.parent.mkdir(parents=True, exist_ok=True)
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    # the <title> must name the model in the file itself, not just at runtime: it is what
    # names the published artifact, and two models' reports are otherwise indistinguishable
    title = args.title or (f"Habitat rover path + visibility — {data['label']} LoRA "
                           f"({data['trainFull']:,} train / {data['nEval']:,} eval)")
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, count=1)
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(data['gallery'])} gallery frames)")


if __name__ == "__main__":
    main()
