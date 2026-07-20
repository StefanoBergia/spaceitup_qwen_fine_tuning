"""Compare two base models across both Habitat tasks in one self-contained page.

Login node (CPU-only), after both models' sweeps have run:
    uv run scripts/visualize_model_comparison.py
    uv run scripts/visualize_model_comparison.py --a-label 2B --b-label 0.8B

Reads the four eval trees (classification and path regression, for each base model)
and writes outputs/model_comparison.html.

Both models are scored on identical held-out frames, so every model-vs-model
difference is tested as a *paired* comparison — McNemar's exact test for the binary
classification outcomes, a paired bootstrap for the continuous regression errors (see
src/rover_vlm/compare.py). A 2-point gap on 500 samples is not self-evidently real,
and the page reports whether each gap survives its test rather than leaving the
reader to infer significance from bar heights.
"""

import argparse
import json
from pathlib import Path

from rover_vlm.compare import (
    load_metrics,
    load_predictions,
    mcnemar_exact,
    paired_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "model_comparison.html"

SIZES = [("base", "base", 0), ("train_500", "500", 500), ("train_1000", "1K", 1000),
         ("train_2000", "2K", 2000), ("train_full", "full", None)]

# classification: higher is better. regression: lower is better except the accuracies.
CHOICE_METRICS = [
    ("accepted_accuracy", "accepted_correct", "Accepted accuracy", "up"),
    ("strict_accuracy", "strict_correct", "Strict accuracy", "up"),
]
REG_METRICS = [
    ("mean_point_error_median", "mean_point_error", "Waypoint error (median)", "down"),
    ("frechet_median", "frechet", "Trajectory shape error (Fréchet)", "down"),
    ("path_visibility_acc_mean", "path_visibility_acc", "Waypoint visibility accuracy", "up"),
    ("goal_visibility_accuracy", "goal_visibility_correct", "Goal visibility accuracy", "up"),
]


def collect(task, a_dir, b_dir, full_size):
    """Per-size metrics for both models plus the paired test on the full-data pair."""
    prefix = "habitat_choice_" if task == "choice" else "habitat_"
    rows, tests = [], []
    for key, label, size in SIZES:
        tag = f"{prefix}base" if key == "base" else f"{prefix}{key}"
        ma, mb = load_metrics(a_dir, tag), load_metrics(b_dir, tag)
        if ma is None and mb is None:
            continue
        rows.append({
            "label": label,
            "size": full_size if size is None else size,
            "a": ma,
            "b": mb,
        })
        if ma is None or mb is None or key == "base":
            continue
        pa, pb = load_predictions(a_dir, tag), load_predictions(b_dir, tag)
        if not pa or not pb:
            continue
        entry = {"size": label}
        if task == "choice":
            for mkey, pkey, name, _ in CHOICE_METRICS:
                a_only, b_only, p = mcnemar_exact(pa, pb, pkey)
                entry[mkey] = {"name": name, "a": ma[mkey], "b": mb[mkey],
                               "aOnly": a_only, "bOnly": b_only, "p": p}
        else:
            for mkey, pkey, name, _ in REG_METRICS:
                bs = paired_bootstrap(pa, pb, pkey)
                if bs:
                    entry[mkey] = {"name": name, "a": ma.get(mkey), "b": mb.get(mkey),
                                   "diff": bs["diff"], "lo": bs["lo"], "hi": bs["hi"],
                                   "n": bs["n"]}
        tests.append(entry)
    return {"rows": rows, "tests": tests}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-dir-choice", type=Path, default=REPO_ROOT / "outputs/eval_habitat_choice")
    p.add_argument("--b-dir-choice", type=Path, default=REPO_ROOT / "outputs/eval_habitat_choice_0.8b")
    p.add_argument("--a-dir-reg", type=Path, default=REPO_ROOT / "outputs/eval_habitat")
    p.add_argument("--b-dir-reg", type=Path, default=REPO_ROOT / "outputs/eval_habitat_0.8b")
    p.add_argument("--a-label", default="Qwen3.5-2B")
    p.add_argument("--b-label", default="Qwen3.5-0.8B")
    p.add_argument("--meta", type=Path, default=REPO_ROOT / "data/prepared_habitat/meta.json")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    full_size = json.loads(args.meta.read_text())["splits"]["train_full"]
    choice = collect("choice", args.a_dir_choice, args.b_dir_choice, full_size)
    reg = collect("reg", args.a_dir_reg, args.b_dir_reg, full_size)

    complete = {
        "choice": all(r["a"] and r["b"] for r in choice["rows"]) and len(choice["rows"]) == len(SIZES),
        "reg": all(r["a"] and r["b"] for r in reg["rows"]) and len(reg["rows"]) == len(SIZES),
    }
    for name, ok in complete.items():
        if not ok:
            print(f"  WARNING: {name} results are incomplete — the page will say so")

    data = {
        "aLabel": args.a_label, "bLabel": args.b_label,
        "choice": choice, "reg": reg, "complete": complete,
        "choiceMetrics": [{"key": k, "name": n, "better": d} for k, _, n, d in CHOICE_METRICS],
        "regMetrics": [{"key": k, "name": n, "better": d} for k, _, n, d in REG_METRICS],
        "trainFull": full_size,
    }

    html = (Path(__file__).parent / "_comparison_report.html").read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html.replace("/*__DATA__*/null", json.dumps(data)))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
