"""Build a self-contained HTML report for the Habitat path-classification results.

Login node (CPU-only), after the eval sweep has run:
    uv run scripts/visualize_choice_results.py
    uv run scripts/visualize_choice_results.py --per-bucket 6

Reads outputs/eval_habitat_choice/<tag>/{metrics,predictions}.json plus the prepared
splits, and writes a single self-contained page (inline CSS/JS, base64 images, no
external requests) to outputs/eval_habitat_choice/choice_results.html — suitable for
opening locally or publishing as an artifact.

Sections: headline tiles, a worked example of the task (the verbatim prompt, a real
frame, the expected answer), accuracy vs training-set size against both chance baselines,
the chosen-index distribution (base vs LoRA vs ground truth), a metrics table, and a
qualitative gallery of real eval frames grouped by how ambiguous the decision was.

The prompt is imported from rover_vlm.habitat_choice rather than copied, so the page
cannot drift out of sync with what the model is actually asked.
"""

import argparse
import base64
import io
import json
import math
import re
import tempfile
from pathlib import Path

from PIL import Image

from rover_vlm.habitat_data import DATASET_ROOTS, sample_dirs_by_id
from rover_vlm.habitat_choice import (
    CANDIDATE_COLORS,
    CHOICE_PROMPT,
    candidate_polylines,
    render_choice_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "outputs" / "eval_habitat_choice"
DATA_DIR = REPO_ROOT / "data" / "prepared_habitat_choice"

ORDER = [
    ("habitat_choice_base", "base", 0),
    ("habitat_choice_train_500", "500", 500),
    ("habitat_choice_train_1000", "1K", 1000),
    ("habitat_choice_train_2000", "2K", 2000),
    ("habitat_choice_train_full", "full", None),  # filled from meta.json
]

# `margin` (metres the runner-up candidate strays from the ideal route) is still used to
# order and bucket the gallery: small margin = candidates nearly tied = hardest frames.


def load(eval_dir: Path, data_dir: Path):
    meta = json.loads((data_dir / "meta.json").read_text())
    full_size = meta["splits"]["train_full"]
    runs = []
    for tag, label, size in ORDER:
        mpath = eval_dir / tag / "metrics.json"
        if not mpath.exists():
            print(f"  note: {tag} missing, skipping")
            continue
        runs.append({
            "tag": tag,
            "label": label,
            "size": full_size if size is None else size,
            "metrics": json.loads(mpath.read_text()),
            "preds": json.loads((eval_dir / tag / "predictions.json").read_text()),
        })
    if not runs:
        raise SystemExit(f"no metrics found under {eval_dir}/<tag>/metrics.json — run the evals first")
    return meta, runs


def pick_distribution(preds, key):
    counts = [0] * 5
    for r in preds:
        v = r[key] if key == "parsed" else r["gt"]["label"]
        if isinstance(v, int) and 0 <= v < 5:
            counts[v] += 1
    return counts


def embed_image(path: Path, max_px: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_gallery(full_run, eval_records, per_bucket, max_px, dataset_root, dashed, tmp_dir):
    """Sample eval frames across margin terciles: ambiguous -> clear.

    Each bucket deliberately includes the model's mistakes when it has any, so the
    gallery shows failure as well as success rather than only flattering cases.

    With `dashed`, frames are re-rendered from the source dataset with occluded stretches
    broken into dashes — a reading aid for the report. The training composites draw those
    stretches solid, so the model itself gets no occlusion cue; `--solid` reproduces them.
    """
    by_id = {r["id"]: r for r in eval_records}
    sample_dirs = sample_dirs_by_id(dataset_root) if dashed else {}
    rows = [r for r in full_run["preds"] if r["gt"].get("margin") is not None and r["id"] in by_id]
    rows.sort(key=lambda r: r["gt"]["margin"])
    n, third = len(rows), len(rows) // 3
    buckets = [
        ("ambiguous", "Ambiguous — candidates nearly tied", rows[:third]),
        ("moderate", "Moderate", rows[third:2 * third]),
        ("clear", "Clear — one obvious winner", rows[2 * third:]),
    ]

    out = []
    for key, title, pool in buckets:
        wrong = [r for r in pool if not r["metrics"]["accepted_correct"]]
        right = [r for r in pool if r["metrics"]["accepted_correct"]]
        # aim for roughly half mistakes where the bucket has enough of them
        want_wrong = min(len(wrong), max(1, per_bucket // 2)) if wrong else 0
        chosen = wrong[:want_wrong]
        step = max(1, len(right) // max(1, per_bucket - want_wrong))
        chosen += right[::step][: per_bucket - want_wrong]
        for r in chosen:
            img_path = Path(by_id[r["id"]]["image"][0])
            if dashed and r["id"] in sample_dirs:
                img_path = tmp_dir / f"{r['id']}.jpg"
                render_choice_image(sample_dirs[r["id"]], img_path, dashed=True)
            if not img_path.exists():
                continue
            out.append({
                "id": r["id"],
                "bucket": key,
                "bucketTitle": title,
                "img": embed_image(img_path, max_px),
                "pick": r["parsed"],
                "label": r["gt"]["label"],
                "accepted": r["gt"]["accepted"],
                "nCand": r["gt"]["n_candidates"],
                "kinds": r["gt"]["kinds"],
                "margin": round(r["gt"]["margin"], 3),
                "correct": bool(r["metrics"]["accepted_correct"]),
            })
    return out


def _legibility(sample_dir):
    """How clearly a frame illustrates the task: the shortest candidate's visible extent.

    Maximising the *minimum* arc length inside the frame favours frames where every
    candidate is well drawn, rather than close-ups where the paths run off-image after a
    few pixels. Returns 0 if anything is unreadable.
    """
    try:
        fpv = json.loads((sample_dir / "fpv_paths.json").read_text())
    except OSError:
        return 0.0
    lengths = []
    for pts in candidate_polylines(fpv):
        if len(pts) < 2:
            return 0.0
        lengths.append(sum(math.dist(pts[i][:2], pts[i + 1][:2]) for i in range(len(pts) - 1)))
    return min(lengths) if lengths else 0.0


def task_example(full_run, eval_records, max_px, sample_dirs):
    """One worked example of the task: the exact image in, the exact answer out.

    Deliberately uses the prepared composite verbatim (solid strokes, no highlight) —
    this panel documents what the model actually receives, so it must not borrow the
    gallery's dashed reading aid. Chooses an unambiguous 3-candidate frame the model got
    right and whose paths are clearly visible: the panel explains the task, so the reader
    should be able to see what is being asked.
    """
    by_id = {r["id"]: r for r in eval_records}
    cands = [r for r in full_run["preds"]
             if r["id"] in by_id and r["gt"]["n_candidates"] == 3
             and len(r["gt"]["accepted"]) == 1 and r["metrics"]["accepted_correct"]
             and (r["gt"].get("margin") or 0) > 0.4]
    cands.sort(key=lambda r: -_legibility(sample_dirs[r["id"]]) if r["id"] in sample_dirs else 0)
    pick = cands[0] if cands else full_run["preds"][0]
    img = Path(by_id[pick["id"]]["image"][0])
    return {
        "prompt": CHOICE_PROMPT,
        "img": embed_image(img, max_px) if img.exists() else None,
        "answer": json.dumps({"choice": pick["gt"]["label"]}, separators=(",", ":")),
        "id": pick["id"],
        "nCand": pick["gt"]["n_candidates"],
    }


def build_data(meta, runs, per_bucket, max_px, data_dir, dataset_root, dashed, tmp_dir):
    full = next(r for r in runs if r["tag"].endswith("train_full"))
    base = next(r for r in runs if r["tag"].endswith("base"))
    m = full["metrics"]
    eval_records = json.loads((data_dir / "eval.json").read_text())

    return {
        "scaling": [
            {"label": r["label"], "size": r["size"],
             "accepted": r["metrics"]["accepted_accuracy"],
             "strict": r["metrics"]["strict_accuracy"],
             "direct": r["metrics"]["picked_direct_rate"]}
            for r in runs
        ],
        "chance": m["chance_accepted"],
        "chanceNoDirect": m["chance_accepted_excluding_direct"],
        "picks": {
            "base": pick_distribution(base["preds"], "parsed"),
            "full": pick_distribution(full["preds"], "parsed"),
            "truth": pick_distribution(full["preds"], "label"),
        },
        "table": [
            {"label": r["label"],
             "size": "—" if r["size"] == 0 else f"{r['size']:,}",
             **{k: r["metrics"][k] for k in
                ("parse_rate", "valid_choice_rate", "strict_accuracy",
                 "accepted_accuracy", "picked_direct_rate")}}
            for r in runs
        ],
        "gallery": build_gallery(full, eval_records, per_bucket, max_px,
                                 dataset_root, dashed, tmp_dir),
        "dashed": dashed,
        "task": task_example(full, eval_records, max_px,
                             sample_dirs_by_id(dataset_root)),
        "colors": ["#%02x%02x%02x" % c for c in CANDIDATE_COLORS],
        "nEval": m["num_samples"],
        "nWrong": sum(1 for r in full["preds"] if not r["metrics"]["accepted_correct"]),
        "multiAcceptedShare": round(
            sum(1 for r in full["preds"] if len(r["gt"]["accepted"]) > 1) / len(full["preds"]), 3),
        "baseAccepted": base["metrics"]["accepted_accuracy"],
        "fullAccepted": m["accepted_accuracy"],
        "fullStrict": m["strict_accuracy"],
        "baseDirect": base["metrics"]["picked_direct_rate"],
        "fullDirect": m["picked_direct_rate"],
        "trainFull": meta["splits"]["train_full"],
        # the base model this tree was actually produced by, recorded by evaluate.py —
        # the page must name it, or a 0.8B report is indistinguishable from a 2B one
        "model": (m.get("model_id") or "Qwen3.5").removeprefix("Qwen/"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--per-bucket", type=int, default=4, help="gallery samples per margin bucket")
    p.add_argument("--max-image-px", type=int, default=420)
    p.add_argument("--dataset-root", type=Path, nargs="*", default=list(DATASET_ROOTS))
    p.add_argument("--solid", action="store_true",
                   help="show the training composites verbatim instead of re-rendering "
                        "the gallery with occluded stretches dashed")
    args = p.parse_args()

    meta, runs = load(args.eval_dir, args.data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        data = build_data(meta, runs, args.per_bucket, args.max_image_px, args.data_dir,
                          args.dataset_root, not args.solid, Path(tmp))

    out = args.out or (args.eval_dir / "choice_results.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (Path(__file__).parent / "_choice_report.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    # the <title> must name the model in the file itself, not just at runtime: it is what
    # names the published artifact, and two models' reports are otherwise indistinguishable
    html = re.sub(r"<title>.*?</title>",
                  f"<title>Habitat path classification — {data['model']} LoRA "
                  f"({data['trainFull']:,} train / {data['nEval']:,} eval)</title>",
                  html, count=1)
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(data['gallery'])} gallery samples)")


if __name__ == "__main__":
    main()
