"""Compare two base models across both Habitat tasks in one self-contained page.

Login node (CPU-only), after both models' sweeps have run:
    uv run scripts/visualize_model_comparison.py
    uv run scripts/visualize_model_comparison.py --a-label 2B --b-label 0.8B

Reads the four eval trees (classification and path regression, for each base model)
and writes outputs/model_comparison.html.

Both models are scored on identical held-out frames, so every model-vs-model
difference is tested as a *paired* comparison — McNemar's exact test for the binary
classification outcomes, a paired bootstrap for the continuous regression errors (see
src/rover_vlm/compare.py). A 2-point gap on 500 samples is not self-evidently real,
and the page reports whether each gap survives its test rather than leaving the
reader to infer significance from bar heights.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

from rover_vlm.compare import (
    load_metrics,
    load_predictions,
    mcnemar_exact,
    paired_bootstrap,
)
from rover_vlm.habitat_choice import CANDIDATE_COLORS, DATASET_ROOT, render_choice_image
from rover_vlm.overlay import draw_path, draw_polyline, embed_jpeg

# must match the --s1 / --s2 series tokens in _comparison_report.html
COLOR_A, COLOR_B, COLOR_GT = (42, 120, 214), (0, 131, 0), (105, 105, 105)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "model_comparison.html"

SIZES = [("base", "base", 0), ("train_500", "500", 500), ("train_1000", "1K", 1000),
         ("train_2000", "2K", 2000), ("train_full", "full", None)]

# classification: higher is better. regression: lower is better except the accuracies.
CHOICE_METRICS = [
    ("accepted_accuracy", "accepted_correct", "Accepted accuracy", "up"),
    ("strict_accuracy", "strict_correct", "Strict accuracy", "up"),
]
REG_METRICS = [
    ("parse_rate", None, "Output format validity", "up"),
    ("mean_point_error_median", "mean_point_error", "Waypoint error (median)", "down"),
    ("frechet_median", "frechet", "Trajectory shape error (Fréchet)", "down"),
    ("path_visibility_acc_mean", "path_visibility_acc", "Waypoint visibility accuracy", "up"),
    ("goal_visibility_accuracy", "goal_visibility_correct", "Goal visibility accuracy", "up"),
]

# Regression errors are averaged over parseable predictions only. Below this parse rate
# the survivors are too few (and too self-selected) for the error to mean anything, so
# the page flags the point instead of drawing it as a peer of a fully-parsing model.
MIN_TRUSTWORTHY_PARSE = 0.5


def collect(task, a_dir, b_dir, full_size):
    """Per-size metrics for both models plus the paired test on the full-data pair."""
    prefix = "habitat_choice_" if task == "choice" else "habitat_"
    rows, tests = [], []
    for key, label, size in SIZES:
        tag = f"{prefix}base" if key == "base" else f"{prefix}{key}"
        ma, mb = load_metrics(a_dir, tag), load_metrics(b_dir, tag)
        if ma is None and mb is None:
            continue
        rows.append({
            "label": label,
            "size": full_size if size is None else size,
            "a": ma,
            "b": mb,
            # error metrics average over parseable outputs only, so carry the parse rate
            # alongside them: a low rate makes the errors a survivorship artefact
            "aLowParse": bool(ma and task != "choice" and ma.get("parse_rate", 1) < MIN_TRUSTWORTHY_PARSE),
            "bLowParse": bool(mb and task != "choice" and mb.get("parse_rate", 1) < MIN_TRUSTWORTHY_PARSE),
            "aParsed": int(round(ma.get("parse_rate", 1) * ma.get("num_samples", 0))) if ma else None,
            "bParsed": int(round(mb.get("parse_rate", 1) * mb.get("num_samples", 0))) if mb else None,
        })
        if ma is None or mb is None or key == "base":
            continue
        pa, pb = load_predictions(a_dir, tag), load_predictions(b_dir, tag)
        if not pa or not pb:
            continue
        entry = {"size": label}
        if task == "choice":
            for mkey, pkey, name, _ in CHOICE_METRICS:
                a_only, b_only, p = mcnemar_exact(pa, pb, pkey)
                entry[mkey] = {"name": name, "a": ma[mkey], "b": mb[mkey],
                               "aOnly": a_only, "bOnly": b_only, "p": p}
        else:
            for mkey, pkey, name, _ in REG_METRICS:
                if pkey is None:          # parse_rate lives only in the summary
                    continue
                bs = paired_bootstrap(pa, pb, pkey)
                if bs:
                    # report the bootstrap's own per-sample means, not the summary's
                    # medians: the CI is a statement about the mean difference, and
                    # pairing a median column with a mean interval would misread
                    entry[mkey] = {"name": name, "a": bs["a"], "b": bs["b"],
                                   "diff": bs["diff"], "lo": bs["lo"], "hi": bs["hi"],
                                   "n": bs["n"]}
        tests.append(entry)
    return {"rows": rows, "tests": tests}


def reg_gallery(pa, pb, eval_records, per_side, max_px):
    """Frames where the two models' visibility judgement differs most, both directions.

    Visibility is where the significant gap lives, so showing the frames that drive it
    is more informative than showing frames picked for looking good. Filled markers mean
    the model called that waypoint visible, hollow means obstructed — so a run of hollow
    dots on open floor is a visible mistake.
    """
    by_id = {r["id"]: r for r in eval_records}
    rows = []
    for sid in set(pa) & set(pb) & set(by_id):
        ma, mb = pa[sid].get("metrics"), pb[sid].get("metrics")
        if not ma or not mb or not pa[sid].get("parsed") or not pb[sid].get("parsed"):
            continue
        rows.append((ma["path_visibility_acc"] - mb["path_visibility_acc"], sid))
    rows.sort()
    picks = rows[:per_side] + rows[-per_side:][::-1]     # 0.8B better first, then 2B better

    out = []
    for delta, sid in picks:
        rec = by_id[sid]
        path = Path(rec["image"][0])
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        gt = pa[sid]["gt"]
        # ground truth as a wide pale corridor underneath, so the two predictions read
        # as deviations from it rather than as a third competing line
        draw_polyline(d, [(p[0], p[1]) for p in gt["path"]], COLOR_GT, W, H, width=9)
        draw_path(d, pa[sid]["parsed"]["path"], COLOR_A, W, H)
        draw_path(d, pb[sid]["parsed"]["path"], COLOR_B, W, H)
        out.append({
            "id": sid,
            "img": embed_jpeg(img, max_px),
            "aVis": pa[sid]["metrics"]["path_visibility_acc"],
            "bVis": pb[sid]["metrics"]["path_visibility_acc"],
            "aErr": pa[sid]["metrics"]["mean_point_error"],
            "bErr": pb[sid]["metrics"]["mean_point_error"],
            "winner": "a" if delta > 0 else "b",
        })
    return out


def choice_gallery(pa, pb, eval_records, sample_dirs, per_side, max_px, tmp_dir):
    """Frames where the two models picked differently — the discordant pairs McNemar counts.

    Balanced across both directions so the gallery shows where the smaller model wins as
    well as where it loses.
    """
    by_id = {r["id"]: r for r in eval_records}
    a_right, b_right = [], []
    for sid in sorted(set(pa) & set(pb) & set(by_id)):
        ma, mb = pa[sid].get("metrics"), pb[sid].get("metrics")
        if not ma or not mb or pa[sid].get("parsed") == pb[sid].get("parsed"):
            continue
        if ma["accepted_correct"] and not mb["accepted_correct"]:
            a_right.append(sid)
        elif mb["accepted_correct"] and not ma["accepted_correct"]:
            b_right.append(sid)

    out = []
    for side, ids in (("a", a_right[:per_side]), ("b", b_right[:per_side])):
        for sid in ids:
            if sid not in sample_dirs:
                continue
            img_path = tmp_dir / f"{sid}.jpg"
            render_choice_image(sample_dirs[sid], img_path, dashed=True)
            gt = pa[sid]["gt"]
            out.append({
                "id": sid,
                "img": embed_jpeg(Image.open(img_path), max_px),
                "aPick": pa[sid]["parsed"],
                "bPick": pb[sid]["parsed"],
                "accepted": gt["accepted"],
                "nCand": gt["n_candidates"],
                "margin": gt.get("margin"),
                "winner": side,
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-dir-choice", type=Path, default=REPO_ROOT / "outputs/eval_habitat_choice")
    p.add_argument("--b-dir-choice", type=Path, default=REPO_ROOT / "outputs/eval_habitat_choice_0.8b")
    p.add_argument("--a-dir-reg", type=Path, default=REPO_ROOT / "outputs/eval_habitat")
    p.add_argument("--b-dir-reg", type=Path, default=REPO_ROOT / "outputs/eval_habitat_0.8b")
    p.add_argument("--a-label", default="Qwen3.5-2B")
    p.add_argument("--b-label", default="Qwen3.5-0.8B")
    p.add_argument("--meta", type=Path, default=REPO_ROOT / "data/prepared_habitat/meta.json")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--per-side", type=int, default=3,
                   help="gallery samples per direction (each model winning)")
    p.add_argument("--max-image-px", type=int, default=480)
    args = p.parse_args()

    full_size = json.loads(args.meta.read_text())["splits"]["train_full"]
    choice = collect("choice", args.a_dir_choice, args.b_dir_choice, full_size)
    reg = collect("reg", args.a_dir_reg, args.b_dir_reg, full_size)

    # galleries compare the full-data models, the pair the verdict is about
    reg_eval = json.loads((REPO_ROOT / "data/prepared_habitat/eval.json").read_text())
    choice_eval = json.loads((REPO_ROOT / "data/prepared_habitat_choice/eval.json").read_text())
    reg_a = load_predictions(args.a_dir_reg, "habitat_train_full")
    reg_b = load_predictions(args.b_dir_reg, "habitat_train_full")
    ch_a = load_predictions(args.a_dir_choice, "habitat_choice_train_full")
    ch_b = load_predictions(args.b_dir_choice, "habitat_choice_train_full")

    import tempfile
    reg_samples, choice_samples = [], []
    if reg_a and reg_b:
        reg_samples = reg_gallery(reg_a, reg_b, reg_eval, args.per_side, args.max_image_px)
    if ch_a and ch_b:
        dirs = {d.name: d for d in args.dataset_root.glob("*/samples/*/")}
        with tempfile.TemporaryDirectory() as tmp:
            choice_samples = choice_gallery(ch_a, ch_b, choice_eval, dirs,
                                            args.per_side, args.max_image_px, Path(tmp))

    # "Complete" means *paired*: every size that ran has both models, so nothing below
    # compares one model against a gap. It deliberately does NOT require all of SIZES —
    # a sweep configured with a single training size (VERSION=_v2 SIZES=train_full) is a
    # finished experiment, not a half-finished one, and flagging it would both cry wolf
    # and suppress the verdict section that is the point of the page.
    def paired(rows):
        return bool(rows) and all(r["a"] and r["b"] for r in rows)

    complete = {"choice": paired(choice["rows"]), "reg": paired(reg["rows"])}
    for name, ok in complete.items():
        if not ok:
            print(f"  WARNING: {name} runs are unpaired (a size is missing for one model)"
                  f" — the page will say so")
        else:
            sizes = ", ".join(r["label"] for r in (choice if name == "choice" else reg)["rows"])
            print(f"  {name}: paired across {sizes}")

    data = {
        "aLabel": args.a_label, "bLabel": args.b_label,
        "choice": choice, "reg": reg, "complete": complete,
        # the held-out split actually scored, so the verdict can state its own n instead
        # of hardcoding the size the first round happened to use
        "evalN": next((r["a"]["num_samples"] for r in reg["rows"] + choice["rows"]
                       if r["a"] and r["a"].get("num_samples")), None),
        "choiceMetrics": [{"key": k, "name": n, "better": d} for k, _, n, d in CHOICE_METRICS],
        "regMetrics": [{"key": k, "name": n, "better": d} for k, _, n, d in REG_METRICS],
        "trainFull": full_size,
        "regSamples": reg_samples,
        "choiceSamples": choice_samples,
        "candidateColors": ["#%02x%02x%02x" % c for c in CANDIDATE_COLORS],
    }

    html = (Path(__file__).parent / "_comparison_report.html").read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    # the <title> must state which run this is in the file itself — it names the published
    # artifact, and successive dataset rounds are otherwise indistinguishable in a gallery
    html = re.sub(r"<title>.*?</title>",
                  f"<title>{args.a_label} vs {args.b_label} on Habitat rover tasks "
                  f"({full_size:,} train / {data['evalN']:,} eval)</title>",
                  html, count=1)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
