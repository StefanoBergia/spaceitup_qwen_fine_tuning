"""Build the scaling comparison (table + plot) from completed eval runs.

Run on the login node after evals (CPU-only):

    uv run scripts/compare_evals.py

Reads every outputs/eval/<tag>/metrics.json. Tags named lora_train_<size> (and
lora_train_full) become points on the scaling curve; the 'base' tag becomes the
zero-shot reference line. Writes:
    outputs/eval/comparison.md    — full metrics table (markdown)
    outputs/eval/scaling_curve.png
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
SERIES = ["#2a78d6", "#008300"]  # categorical slots 1-2

TABLE_COLUMNS = [
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
]


def train_size(tag: str, meta_path: Path) -> int | None:
    """lora_train_500 / habitat_train_500 -> 500; *_train_full -> actual full-split size; else None."""
    for prefix in ("lora_train_", "habitat_train_"):
        if tag.startswith(prefix):
            suffix = tag.removeprefix(prefix)
            if suffix == "full":
                return json.loads(meta_path.read_text())["splits"]["train_full"]
            return int(suffix) if suffix.isdigit() else None
    return None


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


def write_table(base: dict | None, scaling: list[tuple[int, dict]]) -> str:
    header = "| Model | Train size | " + " | ".join(label for _, label in TABLE_COLUMNS) + " |"
    sep = "|" + "---|" * (len(TABLE_COLUMNS) + 2)
    rows = []
    entries = ([("base (zero-shot)", "—", base)] if base else []) + [
        (m["tag"], f"{size:,}", m) for size, m in scaling
    ]
    for name, size, m in entries:
        cells = [f"{m[k]:.3f}" if k in m else "—" for k, _ in TABLE_COLUMNS]
        rows.append(f"| {name} | {size} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def plot(base: dict | None, scaling: list[tuple[int, dict]], out_path: Path) -> None:
    sizes = [s for s, _ in scaling]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)

    panels = [
        ("Format validity", [("parse_rate", "parse rate"), ("in_range_rate", "in-range rate")], "rate"),
        ("Waypoint error", [("mean_point_error_mean", "mean point error")], "normalized dist."),
        ("Trajectory shape error", [("frechet_mean", "Fréchet distance")], "normalized dist."),
    ]
    for ax, (title, metrics, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, (key, label) in enumerate(metrics):
            values = [m.get(key) for _, m in scaling]
            ax.plot(
                sizes, values, "-o", color=SERIES[i], linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label,
            )
            if base and key in base:
                name = f"base {label}" if len(metrics) > 1 else "base"
                ax.axhline(base[key], color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
                ax.annotate(
                    f"{name}: {base[key]:.3f}", xy=(1.0, base[key]), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
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
        ax.set_title(title, fontsize=11, color=INK, pad=10)
        ax.set_xlabel("training samples (log)", fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE_AXIS)
        if len(metrics) > 1:
            ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    fig.suptitle(
        "Qwen3.5-2B LoRA on ShareRobot trajectory — performance vs. training set size",
        fontsize=12, color=INK, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--meta", type=Path, default=REPO_ROOT / "data/prepared/meta.json")
    args = parser.parse_args()

    base, scaling = load_runs(args.eval_dir, args.meta)
    if not scaling and not base:
        raise SystemExit(f"no metrics found under {args.eval_dir}/<tag>/metrics.json — run evals first")

    table = write_table(base, scaling)
    (args.eval_dir / "comparison.md").write_text(table)
    print(table)

    if scaling:
        plot(base, scaling, args.eval_dir / "scaling_curve.png")
        print(f"plot  -> {args.eval_dir / 'scaling_curve.png'}")
    else:
        print("no scaling runs yet — table only")
    print(f"table -> {args.eval_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
