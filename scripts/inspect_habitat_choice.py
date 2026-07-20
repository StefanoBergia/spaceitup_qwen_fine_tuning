"""Render Habitat classification samples with the correct candidate highlighted.

Login node (CPU-only):
    uv run scripts/inspect_habitat_choice.py --num 12

Writes outputs/inspection_habitat_choice/*.jpg — the candidate paths exactly as the
model will see them, but with the correct one thickened and the label/accepted set in
the filename. Eyeball these before spending any GPU time: if the overlays are wrong,
every number downstream is wrong.
"""

import argparse
import json
import random
from pathlib import Path

from rover_vlm.habitat_choice import DATASET_ROOT, build_choice_record, render_choice_image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "inspection_habitat_choice"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-size", type=int, default=768)
    p.add_argument("--multi-only", action="store_true",
                   help="only samples with more than one accepted candidate")
    args = p.parse_args()

    dirs = sorted(args.dataset_root.glob("*/samples/*/"))
    random.Random(args.seed).shuffle(dirs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    drawn = 0
    for d in dirs:
        if drawn >= args.num:
            break
        rec = build_choice_record(d, d / "unused.jpg")
        if rec is None:
            continue
        cm = rec["choice_meta"]
        if args.multi_only and len(cm["accepted"]) < 2:
            continue
        acc = "-".join(str(a) for a in cm["accepted"])
        name = (f"{rec['id']}__label{cm['label']}_accepted{acc}"
                f"_n{cm['n_candidates']}_{cm['kinds'][cm['label']]}.jpg")
        render_choice_image(d, OUT_DIR / name, size=args.render_size, highlight=cm["label"])
        print(f"  {name}")
        drawn += 1

    print(f"\nwrote {drawn} overlays -> {OUT_DIR}")
    print("filename encodes: label = correct index (thickened), accepted = all valid indices")


if __name__ == "__main__":
    main()
