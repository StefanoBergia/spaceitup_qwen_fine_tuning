"""Convert the Habitat dataset into conversation-format splits for training/eval.

Login node (CPU-only):
    uv run scripts/prepare_habitat.py
    uv run scripts/prepare_habitat.py --limit 200   # quick subset for testing

Reads every <root>/*/samples/*/ dir, keeps correct-path-in-FOV samples, and writes
fixed-seed nested splits to data/prepared_habitat/. Images are referenced by absolute
NFS path (never copied).
"""

import argparse
import json
from pathlib import Path

from rover_vlm.data import make_splits
from rover_vlm.habitat_data import DATASET_ROOT, build_record

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "prepared_habitat"
EVAL_SIZE = 500
TRAIN_SIZES = [500, 1000, 2000]
SEED = 42


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="cap sample dirs scanned (testing)")
    args = p.parse_args()

    sample_dirs = sorted(args.dataset_root.glob("*/samples/*/"))
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]
    print(f"scanning {len(sample_dirs)} sample dirs under {args.dataset_root}")

    records, skipped = [], 0
    for i, d in enumerate(sample_dirs):
        try:
            rec = build_record(d)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {d.name}: {e}")
            rec = None
        if rec is None:
            skipped += 1
        else:
            records.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(sample_dirs)} ({len(records)} kept, {skipped} skipped)")

    print(f"kept {len(records)} records, skipped {skipped}")
    if len(records) <= EVAL_SIZE:
        raise SystemExit(f"only {len(records)} records — need > {EVAL_SIZE} for an eval split")

    splits = make_splits(records, EVAL_SIZE, TRAIN_SIZES, SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        (args.out_dir / f"{name}.json").write_text(json.dumps(recs, indent=1))
        print(f"  {name}: {len(recs)}")

    # eval disjoint from every train subset
    eval_ids = {r["id"] for r in splits["eval"]}
    for name, recs in splits.items():
        if name.startswith("train_"):
            assert not (eval_ids & {r["id"] for r in recs}), f"{name} overlaps eval!"

    meta = {"splits": {k: len(v) for k, v in splits.items()}, "kept": len(records), "skipped": skipped}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote splits -> {args.out_dir}")


if __name__ == "__main__":
    main()
