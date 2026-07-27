"""What did more training data buy? Round-1 (3,660) vs v2 (8,140) in one page.

Login node (CPU-only), after slurm/eval_crossround.sbatch:
    uv run scripts/visualize_crossround.py
    uv run scripts/visualize_crossround.py --out outputs/crossround_v3.html

Reads three comparisons that have *different biases* and shows them together, because
no single one settles the question:

  paired-553  round-1 vs v2 on frames absent from round 1's pool. Sensitive (same
              frames, paired tests) but the round-1 model has 0% scene exposure to them
              and v2 has ~99% — so a raw gap here overstates the data effect.
  penalty     the round-1 model on its own eval vs on those 553 frames. Same model, so
              this isolates how much it leaned on having trained in the room, which is
              what the paired-553 gap has to be discounted by.
  unpaired    each model on its own round's eval split. No exposure asymmetry, but the
              two eval splits are different frames — licensed only because the untrained
              base model scores the same on both (that check is on the page).

Writes a self-contained outputs/crossround_report.html.
"""

import argparse
import json
from pathlib import Path

from rover_vlm.compare import load_predictions, mcnemar_exact, paired_bootstrap

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "crossround_report.html"

# model -> (round-1-on-subset tag, v2 eval dir, v2 tag, round-1 own-eval dir, own tag)
REG = {
    "2B": ("r1_2b_reg", "outputs/eval_habitat_v2", "habitat_train_full",
           "outputs/eval_habitat", "habitat_train_full"),
    "0.8B": ("r1_0.8b_reg", "outputs/eval_habitat_v2_0.8b", "habitat_train_full",
             "outputs/eval_habitat_0.8b", "habitat_train_full"),
}
CHOICE = {
    "2B": ("r1_2b_choice", "outputs/eval_habitat_choice_v2", "habitat_choice_train_full",
           "outputs/eval_habitat_choice", "habitat_choice_train_full"),
    "0.8B": ("r1_0.8b_choice", "outputs/eval_habitat_choice_v2_0.8b", "habitat_choice_train_full",
             "outputs/eval_habitat_choice_0.8b", "habitat_choice_train_full"),
}
# per-sample key, summary key, label, direction
REG_CONT = [
    ("mean_point_error", "mean_point_error_mean", "Waypoint error", "down"),
    ("frechet", "frechet_mean", "Trajectory shape (Fréchet)", "down"),
    ("goal_point_error", "goal_point_error_mean", "Goal point error", "down"),
    ("path_visibility_acc", "path_visibility_acc_mean", "Waypoint visibility acc", "up"),
]
REG_BIN = [("goal_visibility_correct", "goal_visibility_accuracy", "Goal visibility acc")]
CHOICE_BIN = [("strict_correct", "strict_accuracy", "Strict accuracy"),
              ("accepted_correct", "accepted_accuracy", "Accepted accuracy")]


def _lcg(seed):
    s = (seed * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
    while True:
        s = (s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        yield s >> 33


def boot_unpaired(a, b, iters=4000, seed=7):
    """CI for mean(a) - mean(b) when a and b are *different* frames (no pairing possible)."""
    if not a or not b:
        return None
    rng = _lcg(seed)
    diffs = []
    for _ in range(iters):
        sa = sum(a[next(rng) % len(a)] for _ in range(len(a))) / len(a)
        sb = sum(b[next(rng) % len(b)] for _ in range(len(b))) / len(b)
        diffs.append(sa - sb)
    diffs.sort()
    return {"a": sum(a) / len(a), "b": sum(b) / len(b),
            "diff": sum(a) / len(a) - sum(b) / len(b),
            "lo": diffs[int(0.025 * iters)], "hi": diffs[int(0.975 * iters)]}


def series(preds, key, ids=None):
    return [r["metrics"][key] for i, r in preds.items()
            if (ids is None or i in ids) and r.get("metrics") and r.get("parsed") is not None]


def rate(preds, key, ids):
    sel = [i for i in ids if i in preds and preds[i].get("metrics")]
    return sum(1 for i in sel if preds[i]["metrics"][key]) / len(sel) if sel else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r1-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_crossround")
    p.add_argument("--subset", type=Path,
                   default=REPO_ROOT / "data" / "prepared_habitat_crossround")
    p.add_argument("--old-splits", type=Path, default=REPO_ROOT / "data" / "prepared_habitat")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    meta = json.loads((args.subset / "meta.json").read_text())
    subset_ids = {r["id"] for r in json.loads((args.subset / "eval.json").read_text())}
    r1_eval_ids = {r["id"] for r in json.loads((args.old_splits / "eval.json").read_text())}

    data = {
        "n": meta["n"], "group": meta["group"],
        "exposure": meta["scene_exposure"],
        "counts": meta["counts"], "excluded": meta["excluded_contaminated"],
        "oldTrain": 3660, "newTrain": 8140,
        "reg": [], "choice": [], "penalty": [], "difficulty": [],
    }

    # ---- 1. regression: paired on the 553, plus the unpaired own-eval comparison ----
    for model, (r1_tag, v2_dir, v2_tag, own_dir, own_tag) in REG.items():
        b = load_predictions(args.r1_dir, r1_tag)
        a_full = load_predictions(REPO_ROOT / v2_dir, v2_tag)
        own = load_predictions(REPO_ROOT / own_dir, own_tag)
        if not b or not a_full:
            raise SystemExit(f"missing predictions for {model} regression — run the sbatch")
        shared = set(a_full) & set(b)
        a = {k: v for k, v in a_full.items() if k in shared}
        b = {k: v for k, v in b.items() if k in shared}
        for pkey, skey, name, better in REG_CONT:
            r = paired_bootstrap(a, b, pkey)
            unp = boot_unpaired(series(a_full, pkey), series(own, pkey))
            # the 55 frames both rounds held out: matched scene exposure, small n
            m55 = paired_bootstrap({k: v for k, v in a_full.items() if k in r1_eval_ids},
                                   {k: v for k, v in own.items() if k in r1_eval_ids}, pkey)
            data["reg"].append({
                "model": model, "name": name, "better": better, "kind": "cont",
                "v2": r["a"], "r1": r["b"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"],
                "sig": r["lo"] > 0 or r["hi"] < 0,
                "unpaired": unp and {"diff": unp["diff"], "lo": unp["lo"], "hi": unp["hi"],
                                     "sig": unp["lo"] > 0 or unp["hi"] < 0},
                "m55": m55 and {"diff": m55["diff"], "lo": m55["lo"], "hi": m55["hi"],
                                "n": m55["n"], "sig": m55["lo"] > 0 or m55["hi"] < 0},
            })
        for pkey, skey, name in REG_BIN:
            v2o, r1o, pv = mcnemar_exact(a, b, pkey)
            data["reg"].append({
                "model": model, "name": name, "better": "up", "kind": "bin",
                "v2": rate(a, pkey, shared), "r1": rate(b, pkey, shared),
                "v2Only": v2o, "r1Only": r1o, "p": pv, "sig": pv < 0.05,
            })
        # how much the round-1 model leaned on room familiarity
        own_m = json.loads((REPO_ROOT / own_dir / own_tag / "metrics.json").read_text())
        sub_m = json.loads((args.r1_dir / r1_tag / "metrics.json").read_text())
        for _, skey, name, better in REG_CONT:
            data["penalty"].append({"model": model, "task": "regression", "name": name,
                                    "own": own_m[skey], "unseen": sub_m[skey],
                                    "better": better,
                                    "hurt": (sub_m[skey] > own_m[skey]) if better == "down"
                                            else (sub_m[skey] < own_m[skey])})

    # ---- 2. classification: the raw gap and what survives the exposure discount ----
    for model, (r1_tag, v2_dir, v2_tag, own_dir, own_tag) in CHOICE.items():
        b = load_predictions(args.r1_dir, r1_tag)
        a_full = load_predictions(REPO_ROOT / v2_dir, v2_tag)
        own = load_predictions(REPO_ROOT / own_dir, own_tag)
        if not b or not a_full:
            raise SystemExit(f"missing predictions for {model} classification — run the sbatch")
        shared = set(a_full) & set(b)
        a = {k: v for k, v in a_full.items() if k in shared}
        b = {k: v for k, v in b.items() if k in shared}
        own_m = json.loads((REPO_ROOT / own_dir / own_tag / "metrics.json").read_text())
        sub_m = json.loads((args.r1_dir / r1_tag / "metrics.json").read_text())
        for pkey, skey, name in CHOICE_BIN:
            v2o, r1o, pv = mcnemar_exact(a, b, pkey)
            raw = rate(a, pkey, shared) - rate(b, pkey, shared)
            penalty = own_m[skey] - sub_m[skey]      # >0 means the r1 model was handicapped
            unp = boot_unpaired(series(a_full, pkey), series(own, pkey))
            data["choice"].append({
                "model": model, "name": name,
                "v2": rate(a, pkey, shared), "r1": rate(b, pkey, shared),
                "raw": raw, "penalty": penalty, "adjusted": raw - penalty,
                "v2Only": v2o, "r1Only": r1o, "p": pv, "sig": pv < 0.05,
                "unpaired": unp and {"diff": unp["diff"], "lo": unp["lo"], "hi": unp["hi"],
                                     "sig": unp["lo"] > 0 or unp["hi"] < 0},
            })
            data["penalty"].append({"model": model, "task": "classification", "name": name,
                                    "own": own_m[skey], "unseen": sub_m[skey],
                                    "better": "up", "hurt": sub_m[skey] < own_m[skey]})

    # ---- 3. is the newer eval split just easier? the untrained base model says no ----
    for label, v2_dir, own_dir, tag, keys in [
        ("regression", "outputs/eval_habitat_v2", "outputs/eval_habitat", "habitat_base",
         [(k, n) for k, _, n, _ in REG_CONT]),
        ("classification", "outputs/eval_habitat_choice_v2", "outputs/eval_habitat_choice",
         "habitat_choice_base", [(k, n) for k, _, n in CHOICE_BIN]),
    ]:
        av, bv = load_predictions(REPO_ROOT / v2_dir, tag), load_predictions(REPO_ROOT / own_dir, tag)
        if not av or not bv:
            continue
        for key, name in keys:
            r = boot_unpaired(series(av, key), series(bv, key))
            if r:
                data["difficulty"].append({"task": label, "name": name, "v2Eval": r["a"],
                                           "r1Eval": r["b"], "diff": r["diff"],
                                           "lo": r["lo"], "hi": r["hi"],
                                           "sig": r["lo"] > 0 or r["hi"] < 0})

    html = (Path(__file__).parent / "_crossround_report.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    n_sig_reg = sum(1 for r in data["reg"] if r["sig"])
    print(f"  regression: {n_sig_reg}/{len(data['reg'])} comparisons significant")
    for c in data["choice"]:
        print(f"  classification {c['model']} {c['name']}: raw {c['raw']:+.4f} "
              f"- exposure {c['penalty']:+.4f} = {c['adjusted']:+.4f}")


if __name__ == "__main__":
    main()
