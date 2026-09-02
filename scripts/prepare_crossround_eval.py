"""Build the eval subset on which two dataset rounds' models can be compared.

Login node (CPU-only):
    uv run scripts/prepare_crossround_eval.py
    uv run scripts/prepare_crossround_eval.py --old-splits data/prepared_habitat_v2 \
        --new-splits data/prepared_habitat_v3 --out-suffix _crossround_v3

Rounds re-shuffle a *different* record list, so the newer eval split is partly made of
frames the older round trained on — scoring the old adapter there would report inflated
numbers on seen data. This selects the frames of the new round's eval split that the old
round could not have trained on, and writes them as a normal eval.json for both tasks.

Two disjoint groups qualify, and they answer different questions (`--group`):

  new    frames absent from the old round's pool entirely — they did not exist yet.
         The old model has 0% scene exposure to them while the new model has ~99%, so a
         new-model win here measures data volume *and* the new-room coverage that came
         with it. This is the default: it is the larger group and it is the practical
         "is the newer model better" question.
  shared frames held out by BOTH rounds. Scene exposure is matched (100%/100%), so this
         isolates data volume with room familiarity held constant — unbiased but small.
  both   their union; do not use without reporting the split, since the two halves are
         biased in opposite directions.

Writes <out>/eval.json + meta.json for the regression task and the classification task
(their split membership is mirrored, so one id set serves both).
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW_SPLITS = REPO_ROOT / "data" / "prepared_habitat_v2"
OLD_SPLITS = REPO_ROOT / "data" / "prepared_habitat"


def load_ids(path: Path) -> set[str]:
    return {r["id"] for r in json.loads(path.read_text())}


def scene(sample_id: str) -> str:
    """00579-9hJwm8k7Gka_c000 -> 00579-9hJwm8k7Gka (the Habitat scene it was sampled in)."""
    return sample_id.split("_c")[0]


def scene_exposure(ids: set[str], train_ids: set[str]) -> float:
    """Share of `ids` whose scene appears in `train_ids` — frame-level disjointness does
    not imply the model has never seen the room, and the asymmetry decides what a
    cross-round win actually means."""
    if not ids:
        return 0.0
    seen = {scene(i) for i in train_ids}
    return sum(1 for i in ids if scene(i) in seen) / len(ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new-splits", type=Path, default=NEW_SPLITS)
    p.add_argument("--old-splits", type=Path, default=OLD_SPLITS)
    p.add_argument("--group", choices=["new", "shared", "both"], default="new")
    p.add_argument("--out-suffix", default="_crossround",
                   help="appended to data/prepared_habitat{,_choice} for the output dirs")
    args = p.parse_args()

    new_eval = load_ids(args.new_splits / "eval.json")
    new_train = load_ids(args.new_splits / "train_full.json")
    old_eval = load_ids(args.old_splits / "eval.json")
    old_train = load_ids(args.old_splits / "train_full.json")
    old_pool = old_eval | old_train

    groups = {
        "new": sorted(new_eval - old_pool),
        "shared": sorted(new_eval & old_eval),
    }
    contaminated = new_eval & old_train
    keep = set(groups[args.group]) if args.group != "both" else set(groups["new"]) | set(groups["shared"])

    print(f"new-round eval: {len(new_eval)}")
    print(f"  in old train (excluded, contaminated) : {len(contaminated)}")
    for name, g in groups.items():
        mark = "<-- selected" if args.group in (name, "both") else ""
        print(f"  {name:7s}: {len(g):4d}  old-model scene exposure {scene_exposure(set(g), old_train):.0%}"
              f", new-model {scene_exposure(set(g), new_train):.0%}  {mark}")
    if not keep:
        raise SystemExit("no frames selected — nothing to compare")

    meta = {
        "group": args.group,
        "n": len(keep),
        "new_splits": str(args.new_splits),
        "old_splits": str(args.old_splits),
        "counts": {k: len(v) for k, v in groups.items()},
        "excluded_contaminated": len(contaminated),
        "scene_exposure": {
            "old_model": round(scene_exposure(keep, old_train), 4),
            "new_model": round(scene_exposure(keep, new_train), 4),
        },
    }

    for task_suffix in ("", "_choice"):
        src = (args.new_splits if not task_suffix
               else Path(str(args.new_splits).replace("prepared_habitat", "prepared_habitat_choice")))
        src_eval = src / "eval.json"
        if not src_eval.exists():
            print(f"  note: {src_eval} missing, skipping {task_suffix or 'regression'}")
            continue
        records = [r for r in json.loads(src_eval.read_text()) if r["id"] in keep]
        out_dir = REPO_ROOT / "data" / f"prepared_habitat{task_suffix}{args.out_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval.json").write_text(json.dumps(records, indent=1))
        (out_dir / "meta.json").write_text(json.dumps({**meta, "task": task_suffix or "regression",
                                                      "splits": {"eval": len(records)}}, indent=2))
        print(f"  wrote {out_dir}/eval.json  ({len(records)} records)")


if __name__ == "__main__":
    main()
