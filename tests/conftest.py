"""Shared pytest fixtures: a synthetic Habitat sample with a feasible left and right route."""
import json

import pytest
from PIL import Image


def write_sample(d, *, left_uv, right_uv, label=0, goal_uv=(256, 300), accepted=(0, 1),
                 labels=("left", "right"), extra=None):
    """extra: optional list of (uv_points, side_label, feasible) appended as candidates."""
    d.mkdir(parents=True, exist_ok=True)
    cands_uv = [left_uv, right_uv] + [e[0] for e in (extra or [])]
    fpv = {
        "image_size": [512, 512],
        "goal": {"uv": list(goal_uv), "hidden": False},
        "candidates": [{"runs": [{"hidden": False, "uv": [list(p) for p in uv]}]} for uv in cands_uv],
    }
    cand_meta = [{"label": labels[0], "feasible": 0 in accepted},
                 {"label": labels[1], "feasible": 1 in accepted}]
    cand_meta += [{"label": e[1], "feasible": e[2]} for e in (extra or [])]
    meta = {
        "sample_id": d.name, "scene_id": "scene_a", "flavor": "doorway",
        "label": label, "accepted": list(accepted), "correct_path_in_fov": True,
        "candidates": cand_meta,
    }
    (d / "fpv_paths.json").write_text(json.dumps(fpv))
    (d / "meta.json").write_text(json.dumps(meta))
    Image.new("RGB", (64, 64), (120, 120, 120)).save(d / "fpv_enhanced.png")
    return d


@pytest.fixture
def two_sided_sample(tmp_path):
    return write_sample(
        tmp_path / "scene_a_c000",
        left_uv=[(256, 512), (140, 400), (256, 300)],
        right_uv=[(256, 512), (370, 400), (256, 300)],
    )
