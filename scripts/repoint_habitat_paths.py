"""Repoint prepared-dataset image paths after the Habitat source tree was reorganized.

    uv run scripts/repoint_habitat_paths.py            # dry-run: report only, write nothing
    uv run scripts/repoint_habitat_paths.py --apply    # back up originals, then rewrite in place

Why this exists: on 2026-07-24 the Habitat source images were moved from
    <root>/habitat_generated/dataset/<scene>/samples/<id>/fpv_enhanced.png
into a train/+val/ split
    <root>/habitat_generated/{train,val}/<scene>/samples/<id>/fpv_enhanced.png
and the flat `dataset/` tree was removed. Every prepared_habitat*/*.json stores absolute
image paths under the old `dataset/` location, so training and visualization can no longer
open them. This rewrites those stored paths to the new location.

Non-destructive and verified:
  * DRY-RUN BY DEFAULT — reports the mapping and any unresolved image, writes nothing.
  * --apply first copies every target file to a timestamped backup dir (originals kept),
    then rewrites in place.
  * The new location is resolved per image by checking train/ then val/ ON DISK; a path is
    only rewritten to somewhere that actually exists. If ANY image in a file resolves to
    neither, that file is left untouched and the run aborts — no half-migrated file.
  * After writing, re-reads each file and asserts every image path exists and the entry
    count is unchanged.

This does NOT touch src/rover_vlm/habitat_data.py's DATASET_ROOT (the scan root for FUTURE
prepare runs); with the source now split across train/ and val/, how a fresh prepare should
treat that split is a separate decision, not a mechanical repoint.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_BASE = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated")
OLD_MARK = "/habitat_generated/dataset/"
# Split subdirs to search, in preference order. train/ first: where a tail exists under both
# (byte-identical copies, verified 2026-07-27) either is correct, so prefer a stable choice.
NEW_SUBDIRS = ("train", "val")

TARGETS = [
    "data/prepared_habitat/train_500.json",
    "data/prepared_habitat/train_1000.json",
    "data/prepared_habitat/train_2000.json",
    "data/prepared_habitat/train_full.json",
    "data/prepared_habitat/eval.json",
    "data/prepared_habitat_crossround/eval.json",
    "data/prepared_habitat_v2/train_full.json",
    "data/prepared_habitat_v2/eval.json",
]


def resolve(img):
    """Map one stored image path to its new on-disk location, or None if unresolved.

    Returns (new_path_str, subdir) when the file exists under a NEW_SUBDIRS location,
    ("__ok__", None) if the path is already valid as-is, or None if it cannot be resolved.
    """
    if OLD_MARK not in img:
        # Not an old-layout path. Only fine if it already points at something real.
        return ("__ok__", None) if Path(img).exists() else None
    tail = img.split(OLD_MARK, 1)[1]  # <scene>/samples/<id>/fpv_enhanced.png
    for sub in NEW_SUBDIRS:
        if (SOURCE_BASE / sub / tail).exists():
            return (str(SOURCE_BASE / sub / tail), sub)
    return None


def plan_file(path):
    """Return (entries, rewrites, unresolved, subdir_counts) without writing anything."""
    data = json.loads(path.read_text())
    rewrites = 0
    unresolved = []
    subdir_counts = {s: 0 for s in NEW_SUBDIRS}
    for e in data:
        for img in e["image"]:
            r = resolve(img)
            if r is None:
                unresolved.append(img)
            elif r[0] != "__ok__":
                rewrites += 1
                subdir_counts[r[1]] += 1
    return data, rewrites, unresolved, subdir_counts


def apply_file(path, data):
    """Rewrite image paths in `data` in place on disk. Assumes plan_file found no unresolved."""
    for e in data:
        e["image"] = [
            (resolve(img)[0] if resolve(img)[0] != "__ok__" else img) for img in e["image"]
        ]
    path.write_text(json.dumps(data))


def verify_file(path, expected_entries):
    data = json.loads(path.read_text())
    assert len(data) == expected_entries, f"{path}: entry count changed"
    missing = [img for e in data for img in e["image"] if not Path(img).exists()]
    assert not missing, f"{path}: {len(missing)} image(s) still missing, e.g. {missing[0]}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    p.add_argument("files", nargs="*", help="override the default target list")
    args = p.parse_args()

    targets = [REPO_ROOT / f for f in (args.files or TARGETS)]

    plans = []
    total_rewrites = 0
    blocked = False
    for path in targets:
        if not path.exists():
            print(f"SKIP  {path}  (does not exist)")
            continue
        data, rewrites, unresolved, subdirs = plan_file(path)
        total_rewrites += rewrites
        tag = "" if not unresolved else f"  <-- {len(unresolved)} UNRESOLVED"
        print(f"{path.relative_to(REPO_ROOT)}: {len(data)} entries, "
              f"{rewrites} rewrites {dict(subdirs)}{tag}")
        if unresolved:
            blocked = True
            for u in unresolved[:3]:
                print(f"    unresolved: {u}")
        plans.append((path, data, len(data)))

    if blocked:
        raise SystemExit("\nABORT: some images resolve to neither train/ nor val/ — "
                         "nothing written. Investigate before applying.")

    if not args.apply:
        print(f"\nDRY-RUN: would rewrite {total_rewrites} image path(s) across "
              f"{len(plans)} file(s). Re-run with --apply to write (originals backed up).")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = REPO_ROOT / "data" / f"_prepared_backup_pre_move_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    print(f"\nbacking up originals to {backup_dir.relative_to(REPO_ROOT)}/ ...")
    for path, _data, _n in plans:
        dest = backup_dir / path.relative_to(REPO_ROOT / "data")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    for path, data, n in plans:
        apply_file(path, data)
        verify_file(path, n)
        print(f"rewrote + verified  {path.relative_to(REPO_ROOT)}")

    print(f"\nDONE: {total_rewrites} path(s) rewritten. Originals preserved in "
          f"{backup_dir.relative_to(REPO_ROOT)}/.")


if __name__ == "__main__":
    main()
