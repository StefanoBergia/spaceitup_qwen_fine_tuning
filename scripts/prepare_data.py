"""Prepare train/eval splits in conversation format from the raw ShareRobot trajectory data.

Run on the login node after scripts/download.py:

    uv run scripts/prepare_data.py                       # defaults: eval=500, sizes 500/1k/2k/5k
    uv run scripts/prepare_data.py --eval-size 500 --train-sizes 500 1000 2000 5000 --seed 42

Writes to data/prepared/:
    eval.json, train_500.json, train_1000.json, ..., train_full.json
    meta.json   — seed, sizes, and provenance for reproducibility

Coordinates are normalized to [0,1] with 3 decimals; trajectories longer than 10 points
are uniformly subsampled (first/last kept). The eval split is fixed across all runs and
training subsets are nested (500 ⊂ 1000 ⊂ ... ⊂ full).
"""

import argparse
import json
from pathlib import Path

from rover_vlm.data import load_raw, make_splits, to_conversation

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAJ_JSON = REPO_ROOT / "data" / "sharerobot" / "trajectory" / "trajectory.json"
OUT_DIR = REPO_ROOT / "data" / "prepared"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not TRAJ_JSON.exists():
        raise SystemExit(f"{TRAJ_JSON} not found — run scripts/download.py first")
    samples = load_raw(TRAJ_JSON)
    print(f"Loaded {len(samples)} raw samples")

    splits = make_splits(samples, args.eval_size, args.train_sizes, args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_ids = {s["id"] for s in splits["eval"]}
    for name, split_samples in splits.items():
        if name != "eval":
            overlap = eval_ids & {s["id"] for s in split_samples}
            assert not overlap, f"{name} overlaps eval split: {sorted(overlap)[:5]}"
        records = [to_conversation(s) for s in split_samples]
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(records, indent=1))
        print(f"  {name}: {len(records):5d} samples -> {out_path}")

    meta = {
        "source": str(TRAJ_JSON.relative_to(REPO_ROOT)),
        "total_raw_samples": len(samples),
        "seed": args.seed,
        "eval_size": args.eval_size,
        "train_sizes": args.train_sizes,
        "splits": {name: len(s) for name, s in splits.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Meta -> {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
