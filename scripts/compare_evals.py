"""Build the scaling comparison (table + plot) from completed eval runs.

Run on the login node after evals (CPU-only):

    uv run scripts/compare_evals.py

Reads every outputs/eval/<tag>/metrics.json. Tags named lora_train_<size> (and
lora_train_full) become points on the scaling curve; the 'base' tag becomes the
zero-shot reference line. Writes:
    outputs/eval/comparison.md    — full metrics table (markdown)
    outputs/eval/scaling_curve.png

Two base models can be overlaid by pointing --compare-dir at a second eval tree
(whose tags must match, which is why per-model results live in separate trees
rather than being distinguished by tag):

    uv run scripts/compare_evals.py \\
        --eval-dir outputs/eval_habitat_0.8b --label 0.8B \\
        --compare-dir outputs/eval_habitat --compare-label 2B \\
        --meta data/prepared_habitat/meta.json

Output always lands in --eval-dir; --compare-dir is read-only.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = REPO_ROOT / "outputs" / "eval"

# palette / chrome (light mode)
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE_AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#008300", "#b8860b"]  # categorical slots 1-3

TRAJECTORY_COLUMNS = [
    ("parse_rate", "Parse rate"),
    ("in_range_rate", "In-range rate"),
    ("rescaled_rate", "Rescaled (0-1000)"),
    ("mean_point_error_mean", "Mean point err"),
    ("mean_point_error_median", "Median point err"),
    ("endpoint_error_mean", "Endpoint err"),
    ("frechet_mean", "Fréchet"),
    ("path_visibility_acc_mean", "Vis. acc"),
    ("goal_point_error_mean", "Goal err"),
    ("goal_visibility_accuracy", "Goal vis. acc"),
    ("goal_copy_rate", "Goal copied"),
]

CHOICE_COLUMNS = [
    ("parse_rate", "Parse rate"),
    ("valid_choice_rate", "Valid index"),
    ("strict_accuracy", "Strict acc"),
    ("accepted_accuracy", "Accepted acc"),
    ("chance_strict", "Chance (strict)"),
    ("chance_accepted", "Chance (accepted)"),
    ("chance_accepted_excluding_direct", "Chance (no direct)"),
    ("picked_direct_rate", "Picked direct"),
]

TASK_COLUMNS = {"trajectory": TRAJECTORY_COLUMNS, "choice": CHOICE_COLUMNS}

TASK_PANELS = {
    "trajectory": [
        ("Format validity", [("parse_rate", "parse rate"), ("in_range_rate", "in-range rate")], "rate"),
        ("Waypoint error", [("mean_point_error_mean", "mean point error")], "normalized dist."),
        ("Trajectory shape error", [("frechet_mean", "Fréchet distance")], "normalized dist."),
    ],
    "choice": [
        ("Format validity", [("parse_rate", "parse rate"), ("valid_choice_rate", "valid index")], "rate"),
        # the "no direct" line is the bar that matters: a model that learns only to
        # avoid the straight-line candidate already scores that well
        ("Accuracy vs. chance",
         [("accepted_accuracy", "accepted acc"), ("chance_accepted", "chance"),
          ("chance_accepted_excluding_direct", "chance, no direct")], "accuracy"),
        ("Strict accuracy (canonical label)", [("strict_accuracy", "strict acc")], "accuracy"),
    ],
}

# {models} is filled with the label(s) of the tree(s) being plotted. The subject is
# deliberately generic for "trajectory": that metric set serves both the ShareRobot
# runs and the Habitat path+visibility runs, so callers pass --title to say which.
TASK_TITLES = {
    "trajectory": "{models} LoRA — waypoint performance vs. training set size",
    "choice": "{models} LoRA on Habitat path choice — accuracy vs. training set size",
}

# line style per model when overlaying: primary solid, comparison dashed
MODEL_STYLES = ["-", "--"]


def train_size(tag: str, meta_path: Path) -> int | None:
    """lora_train_500 / habitat_train_500 / habitat_choice_train_500 -> 500;
    *_train_full -> actual full-split size; else None."""
    for prefix in ("lora_train_", "habitat_choice_train_", "habitat_train_"):
        if tag.startswith(prefix):
            suffix = tag.removeprefix(prefix)
            if suffix == "full":
                return json.loads(meta_path.read_text())["splits"]["train_full"]
            return int(suffix) if suffix.isdigit() else None
    return None


def model_label(base: dict | None, scaling: list[tuple[int, dict]], eval_dir: Path) -> str:
    """Best available name for a tree's base model, for table rows and plot legends.

    Prefers the model_id recorded by evaluate.py; eval trees produced before that
    was recorded fall back to the directory name rather than guessing a model.
    """
    for m in ([base] if base else []) + [m for _, m in scaling]:
        if m.get("model_id"):
            return m["model_id"].removeprefix("Qwen/Qwen3.5-")
    return eval_dir.name


def load_runs(eval_dir: Path, meta_path: Path) -> tuple[dict | None, list[tuple[int, dict]]]:
    base, scaling = None, []
    for metrics_file in sorted(eval_dir.glob("*/metrics.json")):
        m = json.loads(metrics_file.read_text())
        if m["tag"].endswith("base"):
            base = m
        else:
            size = train_size(m["tag"], meta_path)
            if size is not None:
                scaling.append((size, m))
            else:
                print(f"note: skipping unrecognized tag {m['tag']!r}")
    scaling.sort()
    return base, scaling


def write_table(groups: list[tuple[str, dict | None, list[tuple[int, dict]]]],
                task: str = "trajectory") -> str:
    """Markdown table for one or more eval trees.

    A single group renders exactly as it always has; a base-model column is added
    only when overlaying, so existing single-model comparison.md files keep their
    established shape.
    """
    columns = TASK_COLUMNS[task]
    overlay = len(groups) > 1
    lead = ["Base model"] if overlay else []
    header = "| " + " | ".join([*lead, "Model", "Train size", *(label for _, label in columns)]) + " |"
    sep = "|" + "---|" * (len(columns) + 2 + len(lead))

    rows = []
    for model_label, base, scaling in groups:
        entries = ([("base (zero-shot)", "—", base)] if base else []) + [
            (m["tag"], f"{size:,}", m) for size, m in scaling
        ]
        for name, size, m in entries:
            cells = [f"{m[k]:.3f}" if k in m else "—" for k, _ in columns]
            prefix = [model_label] if overlay else []
            rows.append("| " + " | ".join([*prefix, name, size, *cells]) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def _load_predictions(eval_dir: Path, tag: str) -> list[dict] | None:
    f = eval_dir / tag / "predictions.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def baselines_section(eval_dir: Path, tags: list[str]) -> str:
    """What an image-blind predictor scores on this eval set.

    `to_goal` is the one that matters from round 3 on: once the goal is handed to the
    model in the prompt, a blind straight line to that goal costs nothing, so a model
    that does not beat it has not used the image at all. `straight` and `constant` are
    kept for continuity with the earlier rounds' tables.
    """
    from rover_vlm.eval import trivial_baselines

    for tag in tags:
        preds = _load_predictions(eval_dir, tag)
        if not preds:
            continue
        gts = [r["gt"] for r in preds if r.get("gt")]
        if not gts:
            continue
        b = trivial_baselines(gts)
        rows = [
            "| Image-blind baseline | Mean point err | Median point err |",
            "|---|---|---|",
        ]
        for name, blurb in (("to_goal", "straight line to the given goal"),
                            ("straight", "centre line, set median goal height"),
                            ("constant", "the set's own mean path")):
            if name in b:
                rows.append(f"| `{name}` — {blurb} | {b[name]['mean_point_error_mean']:.3f} "
                            f"| {b[name]['mean_point_error_median']:.3f} |")
        section = ("\n### Image-blind floors (computed from ground truth, model-independent)\n\n"
                   + "\n".join(rows) + "\n")

        # the same floor per detour bucket, so a model row can be read against the bar
        # for its own bucket rather than against the set average
        buckets: dict[str, list[dict]] = {}
        for r in preds:
            b = _detour_bucket((r.get("habitat_meta") or {}).get("detour_ratio"))
            if b and r.get("gt"):
                buckets.setdefault(b, []).append(r["gt"])
        names = [n for _, _, n in DETOUR_BUCKETS if n in buckets]
        if names:
            brows = ["| Detour | n | `to_goal` mean | `to_goal` median |", "|---|---|---|---|"]
            for name in names:
                tb = trivial_baselines(buckets[name])["to_goal"]
                brows.append(f"| {name} | {len(buckets[name]):,} "
                             f"| {tb['mean_point_error_mean']:.3f} "
                             f"| {tb['mean_point_error_median']:.3f} |")
            section += "\n" + "\n".join(brows) + "\n"
        return section
    return ""


DETOUR_BUCKETS = [(0.0, 1.05, "near-straight"), (1.05, 1.15, "slight"),
                  (1.15, 1.35, "moderate"), (1.35, 1e9, "strong detour")]


def _detour_bucket(v):
    if v is None:
        return None
    for lo, hi, name in DETOUR_BUCKETS:
        if lo <= v < hi:
            return name
    return None


def _slice_table(eval_dir: Path, tags: list[str], key: str, order=None) -> list[str]:
    """Rows of `| tag | bucket | n | mean | median |` for one habitat_meta key."""
    import numpy as np

    rows = []
    for tag in tags:
        preds = _load_predictions(eval_dir, tag)
        if not preds:
            continue
        groups: dict[str, list[float]] = {}
        for r in preds:
            hm = r.get("habitat_meta") or {}
            bucket = _detour_bucket(hm.get("detour_ratio")) if key == "detour" else hm.get(key)
            if bucket and r.get("metrics"):
                groups.setdefault(bucket, []).append(r["metrics"]["mean_point_error"])
        names = [n for n in order if n in groups] if order else sorted(groups)
        for name in names:
            v = groups[name]
            rows.append(f"| {tag} | {name} | {len(v):,} | {float(np.mean(v)):.3f} "
                        f"| {float(np.median(v)):.3f} |")
    return rows


def source_slice_section(eval_dir: Path, tags: list[str]) -> str:
    """Split each run's error by sample source and by how much the route bends.

    Source: the round-3 training mix is 38% yaw-0 originals, whose goal sits at x=0.5
    exactly. If that pinned mode still costs the model anything, the augmented slice
    scores worse than the original one.

    Detour: with the goal given, a straight line to it is already a strong answer on a
    near-straight route, so the bending routes are the only place a model can show it
    read the image. Read the model's number here against `to_goal` in the same bucket.
    """
    out = []
    src_rows = _slice_table(eval_dir, tags, "source")
    if src_rows:
        out.append("\n### Error by sample source (original = yaw 0, goal pinned at x=0.5)\n\n"
                   + "\n".join(["| Run | Source | n | Mean point err | Median point err |",
                                "|---|---|---|---|---|", *src_rows]) + "\n")
    det_rows = _slice_table(eval_dir, tags, "detour", order=[n for _, _, n in DETOUR_BUCKETS])
    if det_rows:
        out.append("\n### Error by route detour (how far the true route bows off the straight shot)\n\n"
                   + "\n".join(["| Run | Detour | n | Mean point err | Median point err |",
                                "|---|---|---|---|---|", *det_rows]) + "\n")
    return "".join(out)


def plot(groups: list[tuple[str, dict | None, list[tuple[int, dict]]]], out_path: Path,
         task: str = "trajectory", title: str | None = None) -> None:
    """Scaling curves for one or more eval trees.

    When overlaying, colour encodes the metric and line style encodes the base
    model (solid = primary, dashed = comparison), so a panel reads as "same colour,
    different weight" rather than doubling the palette.
    """
    overlay = len(groups) > 1
    sizes = sorted({s for _, _, scaling in groups for s, _ in scaling})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)

    for ax, (panel_title, metrics, ylabel) in zip(axes, TASK_PANELS[task]):
        ax.set_facecolor(SURFACE)
        for g, (model_label, base, scaling) in enumerate(groups):
            group_sizes = [s for s, _ in scaling]
            for i, (key, label) in enumerate(metrics):
                values = [m.get(key) for _, m in scaling]
                # a metric the task doesn't compute (e.g. in_range_rate on habitat runs)
                # would otherwise draw nothing but still claim a legend entry
                if all(v is None for v in values):
                    continue
                series_label = f"{model_label} · {label}" if overlay else label
                ax.plot(
                    group_sizes, values, MODEL_STYLES[g % len(MODEL_STYLES)], marker="o",
                    color=SERIES[i], linewidth=2, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=series_label,
                )
                if base and key in base:
                    name = f"base {label}" if len(metrics) > 1 else "base"
                    if overlay:
                        name = f"{model_label} {name}"
                    ax.axhline(base[key], color=MUTED, linewidth=1.5,
                               linestyle=(0, (4, 3)) if g == 0 else (0, (1, 2)))
                    ax.annotate(
                        f"{name}: {base[key]:.3f}", xy=(1.0, base[key]),
                        xycoords=("axes fraction", "data"),
                        # stagger by model so two close baselines don't overprint
                        xytext=(-4, 4 + 10 * g), textcoords="offset points", ha="right",
                        fontsize=8, color=INK2,
                    )
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        # compact tick labels (500 / 1K / 6.4K) so the tight 5000-vs-full pair doesn't collide
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(
                lambda v, _: f"{v / 1000:.1f}K".replace(".0K", "K") if v >= 1000 else f"{int(v)}"
            )
        )
        ax.minorticks_off()
        ax.set_title(panel_title, fontsize=11, color=INK, pad=10)
        ax.set_xlabel("training samples (log)", fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE_AXIS)
        if (len(metrics) > 1 or overlay) and ax.get_legend_handles_labels()[0]:
            # handlelength: the swatch must be long enough that solid-vs-dashed
            # (which is what encodes the model) is actually distinguishable
            ax.legend(fontsize=8, frameon=False, labelcolor=INK2,
                      handlelength=3.2 if overlay else 2.0)

    fig.suptitle(title or TASK_TITLES[task], fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--meta", type=Path, default=REPO_ROOT / "data/prepared/meta.json")
    parser.add_argument("--task", choices=sorted(TASK_COLUMNS), default="trajectory",
                        help="which metric set to report ('choice' for the classification runs)")
    parser.add_argument("--label", default=None, help="name for --eval-dir's base model in the table/plot")
    parser.add_argument("--compare-dir", type=Path, default=None,
                        help="a second eval tree to overlay (read-only; output goes to --eval-dir)")
    parser.add_argument("--compare-label", default=None, help="name for --compare-dir's base model")
    parser.add_argument("--title", default=None, help="override the plot title")
    args = parser.parse_args()

    base, scaling = load_runs(args.eval_dir, args.meta)
    if not scaling and not base:
        raise SystemExit(f"no metrics found under {args.eval_dir}/<tag>/metrics.json — run evals first")

    groups = [(args.label or model_label(base, scaling, args.eval_dir), base, scaling)]
    if args.compare_dir is not None:
        c_base, c_scaling = load_runs(args.compare_dir, args.meta)
        if not c_scaling and not c_base:
            raise SystemExit(f"no metrics found under {args.compare_dir}/<tag>/metrics.json")
        groups.append(
            (args.compare_label or model_label(c_base, c_scaling, args.compare_dir), c_base, c_scaling)
        )

    title = args.title
    if title is None:
        title = TASK_TITLES[args.task].format(models=" vs. ".join(label for label, _, _ in groups))

    table = write_table(groups, args.task)
    if args.task == "trajectory":
        tags = [m["tag"] for _, b, sc in groups for m in ([b] if b else []) + [x for _, x in sc]]
        table += baselines_section(args.eval_dir, tags)
        table += source_slice_section(args.eval_dir, tags)
    (args.eval_dir / "comparison.md").write_text(table)
    print(table)

    if any(scaling for _, _, scaling in groups):
        plot(groups, args.eval_dir / "scaling_curve.png", args.task, title)
        print(f"plot  -> {args.eval_dir / 'scaling_curve.png'}")
    else:
        print("no scaling runs yet — table only")
    print(f"table -> {args.eval_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
