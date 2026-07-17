"""ShareRobot trajectory data: loading, normalization, prompt templating, splits.

Raw schema (data/sharerobot/trajectory/trajectory.json): each entry has
    id, image_path, meta_data.{original_dataset, original_width, original_height},
    instruction, trajectory = [[x_px, y_px], ...]   (pixel coordinates, >= 3 points)

We convert to the RoboBrain conversation format: normalized [0,1] coordinates with
3 decimals (Qwen grounding convention), at most MAX_WAYPOINTS points.
"""

import json
import random
from pathlib import Path

MAX_WAYPOINTS = 10

# Prompt template used by RoboBrain-LoRA-Trajectory (see CLAUDE.md)
PROMPT_TEMPLATE = (
    '<image>\nYou are a robot using the joint control. The task is "{instruction}". '
    f"Please predict up to {MAX_WAYPOINTS} key trajectory points to complete the task. "
    "Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], "
    "where each tuple contains the x and y coordinates of a point."
)


def load_raw(trajectory_json: Path) -> list[dict]:
    """Load the raw trajectory.json annotation list."""
    return json.loads(Path(trajectory_json).read_text())


def subsample_waypoints(points: list[list[float]], max_points: int = MAX_WAYPOINTS) -> list[list[float]]:
    """Uniformly subsample a trajectory to at most max_points, always keeping first and last."""
    if len(points) <= max_points:
        return points
    idx = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
    return [points[i] for i in idx]


def normalize_trajectory(sample: dict) -> list[list[float]]:
    """Pixel coords -> [0,1] normalized, 3 decimals, clamped, capped at MAX_WAYPOINTS."""
    w = sample["meta_data"]["original_width"]
    h = sample["meta_data"]["original_height"]
    points = subsample_waypoints(sample["trajectory"])
    return [
        [round(min(max(x / w, 0.0), 1.0), 3), round(min(max(y / h, 0.0), 1.0), 3)]
        for x, y in points
    ]


def format_answer(norm_points: list[list[float]]) -> str:
    """Render normalized waypoints as the target string, e.g. [[0.123, 0.456], ...]."""
    return "[" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in norm_points) + "]"


def to_conversation(sample: dict) -> dict:
    """Convert one raw sample to the conversation-format record used for training/eval."""
    return {
        "id": sample["id"],
        "image": [f"images/{sample['image_path']}"],  # relative to the trajectory/ dir
        "conversations": [
            {"from": "human", "value": PROMPT_TEMPLATE.format(instruction=sample["instruction"])},
            {"from": "gpt", "value": format_answer(normalize_trajectory(sample))},
        ],
    }


def make_splits(
    samples: list[dict],
    eval_size: int,
    train_sizes: list[int],
    seed: int,
) -> dict[str, list[dict]]:
    """Fixed-seed eval split + nested training subsets.

    The eval split is disjoint from all training subsets and identical across runs.
    Training subsets are nested (each smaller subset is a prefix of the larger ones)
    so the scaling curve isn't confounded by subset composition.
    Returns {"eval": [...], "train_500": [...], ..., "train_full": [...]}.
    """
    order = list(samples)
    random.Random(seed).shuffle(order)

    splits = {"eval": order[:eval_size]}
    pool = order[eval_size:]
    for size in train_sizes:
        if size < len(pool):
            splits[f"train_{size}"] = pool[:size]
    splits["train_full"] = pool
    return splits
