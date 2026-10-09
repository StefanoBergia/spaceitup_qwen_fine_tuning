"""Render Habitat classification samples with the correct candidate highlighted.

Login node (CPU-only):
    uv run scripts/inspect_habitat_choice.py --num 12

Writes outputs/inspection_habitat_choice/*.jpg — the candidate paths with the correct one
thickened, occluded stretches dashed, and the label/accepted set in the filename. Eyeball
these before spending any GPU time: if the overlays are wrong, every number downstream is
wrong.

Note the highlight and the dashes are reading aids: training composites draw every path
solid and unhighlighted. Pass --solid to see exactly what the model is given.
"""

import argparse
import json
import random
from pathlib import Path

from rover_vlm.habitat_data import DATASET_ROOTS, sample_dirs_by_id
from rover_vlm.habitat_choice import build_choice_record, render_choice_image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "inspection_habitat_choice"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, nargs="*", default=list(DATASET_ROOTS))
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-size", type=int, default=768)
    p.add_argument("--multi-only", action="store_true",
                   help="only samples with more than one accepted candidate")
    p.add_argument("--solid", action="store_true",
                   help="draw occluded stretches solid, exactly as the training composites are")
    args = p.parse_args()

    dirs = sorted(sample_dirs_by_id(args.dataset_root).values())
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
        render_choice_image(d, OUT_DIR / name, size=args.render_size,
                            highlight=cm["label"], dashed=not args.solid)
        print(f"  {name}")
        drawn += 1

    print(f"\nwrote {drawn} overlays -> {OUT_DIR}")
    print("filename encodes: label = correct index (thickened), accepted = all valid indices")


if __name__ == "__main__":
    main()
