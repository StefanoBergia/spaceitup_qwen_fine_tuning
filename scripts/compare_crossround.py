"""Compare a round-1 model against a v2 model on frames neither could have trained on.

Login node (CPU-only), after slurm/eval_crossround.sbatch has run:
    uv run scripts/compare_crossround.py
    uv run scripts/compare_crossround.py --group shared   # matched-exposure frames only

The round-1 side comes from outputs/eval_crossround/ (produced by the sbatch). The v2
side is *subset* from the existing full-eval predictions rather than re-run: greedy
decoding makes that equivalent in principle, and it avoids the batch-composition drift a
553-frame re-run would introduce relative to the published 1,000-frame numbers.

Differences are paired — same frames, both models — so they get McNemar's exact test
(binary outcomes) or a paired bootstrap (continuous errors), from src/rover_vlm/compare.py.

Interpretation depends entirely on which group is used, and the script prints the scene
exposure so this cannot be read off carelessly:
  new    (553) the round-1 model has never seen these rooms; the v2 model has trained in
         ~99% of them. A v2 win = more data AND the new-scene coverage it brought.
  shared (55)  both models trained in these rooms. Isolates data volume, but n is small:
         only differences of a few points are detectable.
"""

import argparse
import json
from pathlib import Path

from rover_vlm.compare import load_predictions, mcnemar_exact, paired_bootstrap

REPO_ROOT = Path(__file__).resolve().parent.parent

# task -> (round-1 tag, v2 eval dir, v2 tag) per model
LAYOUT = {
    "regression": {
        "2B": ("r1_2b_reg", "outputs/eval_habitat_v2", "habitat_train_full"),
        "0.8B": ("r1_0.8b_reg", "outputs/eval_habitat_v2_0.8b", "habitat_train_full"),
    },
    "classification": {
        "2B": ("r1_2b_choice", "outputs/eval_habitat_choice_v2", "habitat_choice_train_full"),
        "0.8B": ("r1_0.8b_choice", "outputs/eval_habitat_choice_v2_0.8b", "habitat_choice_train_full"),
    },
}
CONTINUOUS = [("mean_point_error", "mean point error", "down"),
              ("frechet", "Fréchet", "down"),
              ("goal_point_error", "goal point error", "down"),
              ("path_visibility_acc", "waypoint visibility acc", "up")]
BINARY_REG = [("goal_visibility_correct", "goal visibility")]
BINARY_CHOICE = [("strict_correct", "strict accuracy"), ("accepted_correct", "accepted accuracy")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r1-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_crossround")
    p.add_argument("--subset-meta", type=Path,
                   default=REPO_ROOT / "data" / "prepared_habitat_crossround" / "meta.json")
    p.add_argument("--group", default=None,
                   help="label only; the frame set comes from whatever the sbatch scored")
    args = p.parse_args()

    if args.subset_meta.exists():
        m = json.loads(args.subset_meta.read_text())
        print(f"subset: group={m['group']}  n={m['n']}  "
              f"scene exposure: round-1 model {m['scene_exposure']['old_model']:.0%}, "
              f"v2 model {m['scene_exposure']['new_model']:.0%}")
        if m["scene_exposure"]["old_model"] < m["scene_exposure"]["new_model"] - 0.1:
            print("  NOTE: exposure is asymmetric — a v2 win here includes the benefit of\n"
                  "        having trained in these rooms, not data volume alone.")
    print()

    any_found = False
    for task, models in LAYOUT.items():
        for model, (r1_tag, v2_dir, v2_tag) in models.items():
            a = load_predictions(REPO_ROOT / v2_dir, v2_tag)     # v2 model, full eval
            b = load_predictions(args.r1_dir, r1_tag)            # round-1 model, subset
            if not b:
                print(f"{task} {model}: {args.r1_dir}/{r1_tag} missing — run "
                      f"slurm/eval_crossround.sbatch")
                continue
            if not a:
                print(f"{task} {model}: {v2_dir}/{v2_tag} missing")
                continue
            any_found = True
            shared = set(a) & set(b)          # restricting v2 to the scored subset
            a = {k: v for k, v in a.items() if k in shared}
            b = {k: v for k, v in b.items() if k in shared}
            print(f"--- {task} · {model}  (n={len(shared)}; v2 = 8,140 train vs round-1 = 3,660)")
            if task == "regression":
                for key, name, better in CONTINUOUS:
                    r = paired_bootstrap(a, b, key)
                    if not r:
                        continue
                    sig = r["lo"] > 0 or r["hi"] < 0
                    winner = "" if not sig else ("  v2 better" if (r["diff"] < 0) == (better == "down")
                                                 else "  round-1 better")
                    print(f"    {name:24s} v2={r['a']:.4f}  r1={r['b']:.4f}  "
                          f"diff={r['diff']:+.4f}  CI[{r['lo']:+.4f},{r['hi']:+.4f}]"
                          f"  {'SIG' if sig else 'ns'}{winner}")
                binaries = BINARY_REG
            else:
                binaries = BINARY_CHOICE
            for key, name in binaries:
                v2_only, r1_only, pval = mcnemar_exact(a, b, key)
                print(f"    {name:24s} v2-only={v2_only:3d}  r1-only={r1_only:3d}  "
                      f"p={pval:.4g}  {'SIG' if pval < 0.05 else 'ns'}")
            print()

    if not any_found:
        raise SystemExit("nothing to compare — run slurm/eval_crossround.sbatch first")


if __name__ == "__main__":
    main()
