"""Build the Habitat path-*classification* splits: render candidate paths onto each
frame and pair them with the correct candidate index.

Login node (CPU-only). Requires prepare_habitat.py to have run first — split
membership is copied from data/prepared_habitat/ so the classification and regression
experiments run on exactly the same frames and can be compared sample by sample.

    uv run scripts/prepare_habitat_choice.py
    uv run scripts/prepare_habitat_choice.py --limit 50    # quick subset for testing

Rendered images go to data/prepared_habitat_choice/images/ (gitignored; the source
dataset is never written to). Re-running skips images that already exist, so an
interrupted run just continues.
"""

import argparse
import collections
import json
from pathlib import Path

from rover_vlm.habitat_choice import (
    DATASET_ROOT,
    build_choice_record,
    drawable_candidates,
    render_choice_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_SPLITS = REPO_ROOT / "data" / "prepared_habitat"
OUT_DIR = REPO_ROOT / "data" / "prepared_habitat_choice"
SPLIT_NAMES = ["eval", "train_500", "train_1000", "train_2000", "train_full"]


def sample_dirs_by_id(root: Path) -> dict[str, Path]:
    return {d.name: d for d in root.glob("*/samples/*/") if d.is_dir()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--src-splits", type=Path, default=SRC_SPLITS,
                   help="regression splits whose membership we mirror")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="cap samples per split (testing)")
    p.add_argument("--render-size", type=int, default=768)
    args = p.parse_args()

    if not (args.src_splits / "eval.json").exists():
        raise SystemExit(
            f"{args.src_splits}/eval.json not found — run scripts/prepare_habitat.py first"
        )

    dirs = sample_dirs_by_id(args.dataset_root)
    print(f"found {len(dirs)} sample dirs under {args.dataset_root}")
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {}
    cache: dict[str, dict | None] = {}   # id -> record, so nested splits render once
    dropped: dict[str, str] = {}

    for name in SPLIT_NAMES:
        src = args.src_splits / f"{name}.json"
        if not src.exists():
            print(f"  {name}: missing in {args.src_splits}, skipping")
            continue
        ids = [r["id"] for r in json.loads(src.read_text())]
        if args.limit:
            ids = ids[: args.limit]

        records = []
        for i, sid in enumerate(ids):
            if sid not in cache:
                cache[sid] = build_one(sid, dirs, img_dir, args.render_size, dropped)
            if cache[sid] is not None:
                records.append(cache[sid])
            if (i + 1) % 500 == 0:
                print(f"  {name}: {i + 1}/{len(ids)} ({len(records)} kept)", flush=True)
        splits[name] = records
        print(f"  {name}: {len(records)}/{len(ids)} kept")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        (args.out_dir / f"{name}.json").write_text(json.dumps(recs, indent=1))

    # eval must stay disjoint from every train split (inherited, but verify anyway)
    eval_ids = {r["id"] for r in splits.get("eval", [])}
    for name, recs in splits.items():
        if name.startswith("train_"):
            assert not (eval_ids & {r["id"] for r in recs}), f"{name} overlaps eval!"

    (args.out_dir / "meta.json").write_text(json.dumps(shortcut_audit(splits, dropped), indent=2))
    if dropped:
        print(f"dropped {len(dropped)} samples; reasons: "
              f"{dict(collections.Counter(dropped.values()))}")
    print(f"wrote splits -> {args.out_dir}")


def build_one(sid, dirs, img_dir, render_size, dropped):
    """Render + build one record, recording why it was dropped if it fails."""
    sample_dir = dirs.get(sid)
    if sample_dir is None:
        dropped[sid] = "sample dir not found"
        return None
    img_path = img_dir / f"{sid}.jpg"
    try:
        fpv = json.loads((sample_dir / "fpv_paths.json").read_text())
        # every candidate must actually appear in the frame, or the choice set the model
        # sees doesn't match the label space it is scored against
        if len(drawable_candidates(fpv)) != len(fpv.get("candidates", [])):
            dropped[sid] = "candidate not visible in frame"
            return None
        if not img_path.exists():
            render_choice_image(sample_dir, img_path, size=render_size)
        rec = build_choice_record(sample_dir, img_path)
    except Exception as e:  # noqa: BLE001
        dropped[sid] = f"{type(e).__name__}: {e}"
        return None
    if rec is None:
        dropped[sid] = "invalid label/accepted/display_index"
    return rec


def shortcut_audit(splits, dropped):
    """Split sizes plus the numbers needed to read any accuracy against chance.

    `direct` is the straight-line candidate present in nearly every sample; it is almost
    never the right answer, so a model can score well by learning only to avoid it. The
    two chance baselines make that explicit.
    """
    recs = splits.get("train_full", []) + splits.get("eval", [])
    n_cand = collections.Counter(r["choice_meta"]["n_candidates"] for r in recs)
    n_acc = collections.Counter(len(r["choice_meta"]["accepted"]) for r in recs)
    label_kind = collections.Counter(
        r["choice_meta"]["kinds"][r["choice_meta"]["label"]] for r in recs
    )
    chance_s, chance_a, chance_s_nd, chance_a_nd = [], [], [], []
    for r in recs:
        cm = r["choice_meta"]
        kinds, acc, n = cm["kinds"], set(cm["accepted"]), cm["n_candidates"]
        for pool, s_out, a_out in (
            (list(range(n)), chance_s, chance_a),
            ([i for i in range(n) if kinds[i] != "direct"], chance_s_nd, chance_a_nd),
        ):
            if pool:
                s_out.append(1.0 / len(pool))
                a_out.append(len([i for i in pool if i in acc]) / len(pool))
    mean = lambda v: float(sum(v) / len(v)) if v else None  # noqa: E731
    return {
        "splits": {k: len(v) for k, v in splits.items()},
        "dropped": len(dropped),
        "n_candidates_dist": dict(sorted(n_cand.items())),
        "n_accepted_dist": dict(sorted(n_acc.items())),
        "correct_label_kind": dict(label_kind),
        "chance_strict": mean(chance_s),
        "chance_accepted": mean(chance_a),
        "chance_strict_excluding_direct": mean(chance_s_nd),
        "chance_accepted_excluding_direct": mean(chance_a_nd),
    }


if __name__ == "__main__":
    main()
