"""Convert the Habitat dataset into conversation-format splits for training/eval.

Login node (CPU-only). Round 4 (manifest mode, the current one) — the generator ships
its own scene-disjoint train/val manifests, so no split is invented here, and the prompt
hands over both the route's entry point and its goal:

    uv run scripts/prepare_habitat.py --out-dir data/prepared_habitat_v4 --framing endpoints

Round 3 (goal only in the prompt; --goal-in-prompt is the older spelling) is reproduced by:

    uv run scripts/prepare_habitat.py --out-dir data/prepared_habitat_v3 --framing goal

Rounds 1-2 (glob mode, kept for reproducibility) — scan every sample dir and build a
fixed-seed split locally:

    uv run scripts/prepare_habitat.py
    uv run scripts/prepare_habitat.py --limit 200   # quick subset for testing
    uv run scripts/prepare_habitat.py --out-dir data/prepared_habitat_v2 \
        --eval-size 1000 --train-sizes ""

Manifest mode takes `train_full` from manifest_train.jsonl and `eval` from
manifest_val.jsonl verbatim, then asserts the three properties that make the split
trustworthy: eval and train share no sample id, no *scene*, and no augmented sample is
separated from the parent it was yawed from. The v2 split satisfied only the first
(392 of its 397 eval scenes were also in train), which is why every pre-v3 Habitat
number is scene-leaked and not comparable to these.

Images are referenced by absolute NFS path, never copied.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from rover_vlm.data import make_splits
from rover_vlm.habitat_data import (
    DATASET_ROOTS,
    FRAMINGS,
    MANIFEST_TRAIN,
    MANIFEST_VAL,
    build_record,
    sample_dirs_by_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "prepared_habitat"
EVAL_SIZE = 500
TRAIN_SIZES = [500, 1000, 2000]
SEED = 42


def parse_train_sizes(spec: str) -> list[int]:
    """"500,1000" -> [500, 1000]; "" -> [] (train_full only, always emitted)."""
    return [int(s) for s in spec.replace(",", " ").split()]


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def records_from_manifest(rows, *, framing, reasons):
    """Manifest rows -> conversation records, carrying the manifest's own fields through."""
    records = []
    for row in rows:
        extra = {
            "sample_id": row["sample_id"],
            "source": row.get("source"),
            "label_side": row.get("label_side"),
            "margin": row.get("margin"),
            "decidability": row.get("decidability"),
            "near_symmetric": row.get("near_symmetric"),
            "n_candidates": row.get("n_candidates"),
            # how far the true route bows away from the straight shot; the axis the
            # `to_goal` baseline is weakest on, so the one worth slicing results by
            "detour_ratio": row.get("detour_ratio"),
        }
        rec = build_record(
            row["dir"], framing=framing, meta_extra=extra, reasons=reasons
        )
        if rec is not None:
            records.append(rec)
    return records


def _scenes(records):
    return {r["habitat_meta"]["scene_id"] for r in records}


def check_split_integrity(train, eval_):
    """The three properties that make a train/eval split trustworthy. Fatal if violated."""
    train_ids, eval_ids = {r["id"] for r in train}, {r["id"] for r in eval_}
    shared_ids = train_ids & eval_ids
    assert not shared_ids, f"{len(shared_ids)} sample ids in both splits, e.g. {sorted(shared_ids)[:3]}"

    shared_scenes = _scenes(train) & _scenes(eval_)
    assert not shared_scenes, (
        f"{len(shared_scenes)} scenes in both splits, e.g. {sorted(shared_scenes)[:3]} — "
        "a scene-leaked split inflates every number (this is what v2 got wrong)"
    )

    for name, here, there_ids in (("train", train, eval_ids), ("eval", eval_, train_ids)):
        strays = {r["id"] for r in here
                  if (r["habitat_meta"].get("parent_sample_id") or "") in there_ids}
        assert not strays, (
            f"{len(strays)} augmented {name} samples have their parent in the other split, "
            f"e.g. {sorted(strays)[:3]}"
        )


def split_stats(records):
    """Numbers that make the de-biasing visible in the artifact, not just asserted."""
    ends_x = [json.loads(r["conversations"][1]["value"])["path"][-1][0] for r in records]
    goals_x = [r["habitat_meta"]["goal_uv_norm"][0] for r in records]

    def sd(v):
        return round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0

    by_source = Counter(r["habitat_meta"].get("source") or "unknown" for r in records)
    # the class round 4 targets: routes clipped to enter from a side edge are the ones
    # whose start is undetermined from the image alone
    by_edge = Counter(r["habitat_meta"].get("entry_edge") or "unknown" for r in records)
    stats = {
        "n": len(records),
        "scenes": len(_scenes(records)),
        "by_source": dict(sorted(by_source.items())),
        "by_entry_edge": dict(sorted(by_edge.items())),
        "terminus_x_sd": sd(ends_x),
        "goal_x_sd": sd(goals_x),
        "frac_terminus_at_0.5": round(sum(abs(x - 0.5) < 0.002 for x in ends_x) / len(ends_x), 4),
    }
    for src in sorted(by_source):
        sub = [json.loads(r["conversations"][1]["value"])["path"][-1][0]
               for r in records if (r["habitat_meta"].get("source") or "unknown") == src]
        stats[f"terminus_x_sd_{src}"] = sd(sub)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, nargs="*", default=list(DATASET_ROOTS),
                   help="glob mode: render tree(s) to scan")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="glob mode: cap sample dirs scanned")
    p.add_argument("--eval-size", type=int, default=EVAL_SIZE,
                   help="glob mode: held-out eval split size")
    p.add_argument("--train-sizes", type=parse_train_sizes, default=list(TRAIN_SIZES),
                   help="glob mode: comma/space-separated subset sizes; empty for train_full only")
    p.add_argument("--manifest-train", type=Path, default=None,
                   help=f"manifest mode: train split (use {MANIFEST_TRAIN} for round 3)")
    p.add_argument("--manifest-val", type=Path, default=None,
                   help=f"manifest mode: eval split (use {MANIFEST_VAL} for round 3)")
    p.add_argument("--framing", choices=FRAMINGS, default=None,
                   help="what the prompt hands over: 'legacy' (rounds 1-2), 'goal' "
                        "(round 3), 'endpoints' (round 4, start + goal)")
    p.add_argument("--goal-in-prompt", action="store_true",
                   help="deprecated spelling of --framing goal")
    args = p.parse_args()

    if args.framing is None:
        args.framing = "goal" if args.goal_in_prompt else "legacy"
    elif args.goal_in_prompt and args.framing != "goal":
        raise SystemExit(f"--goal-in-prompt conflicts with --framing {args.framing}")

    # a goal-bearing framing implies the round-3+ manifests unless named explicitly
    if args.framing != "legacy" and args.manifest_train is None and args.manifest_val is None:
        args.manifest_train, args.manifest_val = MANIFEST_TRAIN, MANIFEST_VAL
    manifest_mode = args.manifest_train is not None or args.manifest_val is not None
    if manifest_mode and not (args.manifest_train and args.manifest_val):
        raise SystemExit("--manifest-train and --manifest-val must be given together")

    reasons: dict[str, int] = {}

    if manifest_mode:
        print(f"manifest mode: {args.manifest_train}\n               {args.manifest_val}")
        train_rows, val_rows = read_manifest(args.manifest_train), read_manifest(args.manifest_val)
        print(f"  manifest rows: {len(train_rows)} train, {len(val_rows)} val")
        splits = {
            "train_full": records_from_manifest(
                train_rows, framing=args.framing, reasons=reasons),
            "eval": records_from_manifest(
                val_rows, framing=args.framing, reasons=reasons),
        }
        check_split_integrity(splits["train_full"], splits["eval"])
        meta = {
            "mode": "manifest",
            "framing": args.framing,
            "manifests": {"train": str(args.manifest_train), "val": str(args.manifest_val)},
            "splits": {k: len(v) for k, v in splits.items()},
            "skipped": reasons,
            "stats": {k: split_stats(v) for k, v in splits.items()},
        }
    else:
        by_id = sample_dirs_by_id(args.dataset_root)
        sample_dirs = [by_id[k] for k in sorted(by_id)]
        if args.limit:
            sample_dirs = sample_dirs[: args.limit]
        print(f"glob mode: {len(sample_dirs)} sample dirs under {args.dataset_root}")

        records = []
        for i, d in enumerate(sample_dirs):
            try:
                rec = build_record(d, framing=args.framing, reasons=reasons)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {d.name}: {e}")
                reasons["error"] = reasons.get("error", 0) + 1
                rec = None
            if rec is not None:
                records.append(rec)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(sample_dirs)} ({len(records)} kept)")

        skipped = sum(reasons.values())
        print(f"kept {len(records)} records, skipped {skipped}")
        if len(records) <= args.eval_size:
            raise SystemExit(f"only {len(records)} records — need > {args.eval_size} for an eval split")

        splits = make_splits(records, args.eval_size, args.train_sizes, SEED)
        eval_ids = {r["id"] for r in splits["eval"]}
        for name, recs in splits.items():
            if name.startswith("train_"):
                assert not (eval_ids & {r["id"] for r in recs}), f"{name} overlaps eval!"
        meta = {
            "mode": "glob",
            "framing": args.framing,
            "splits": {k: len(v) for k, v in splits.items()},
            "kept": len(records),
            "skipped": skipped,
            "skip_reasons": reasons,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        (args.out_dir / f"{name}.json").write_text(json.dumps(recs, indent=1))
        print(f"  {name}: {len(recs)}")
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote splits -> {args.out_dir}")
    if manifest_mode:
        print(json.dumps(meta["stats"], indent=2))
        if reasons:
            print(f"skipped: {reasons}")


if __name__ == "__main__":
    main()
