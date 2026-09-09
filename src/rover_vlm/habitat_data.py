"""Habitat rover-navigation data: select correct path, normalize, clip, resample,
format the path+visibility conversation record.

Source layout: <root>/<split>/<scene>/samples/<scene>_cNNN/ with fpv_enhanced.png,
fpv_paths.json, meta.json, where <root> is one of DATASET_ROOTS and <split> is
train/val. See docs/superpowers/specs/2026-07-17-habitat-path-visibility-design.md.

Coordinates normalize to [0,1] by fpv_paths["image_size"]; visibility v is
1 = visible, 0 = obstructed (v = 0 if the run/goal is "hidden").

Two prompt variants live side by side rather than one being edited in place:

* HABITAT_PROMPT (rounds 1-2) asserts "the goal is located straight ahead", true of
  the un-augmented renders and false everywhere else. Frozen, because
  data/prepared_real/* and every prediction under outputs/eval_real* was built with
  it -- editing it would silently reinterpret those artifacts.
* HABITAT_PROMPT_GOAL (round 3) hands the model the goal as [x, y, v] and asks only
  for the route to it; the answer's final waypoint is that goal. See
  docs/endpoint_collapse.md for why the first framing had to go.
"""

import json
from pathlib import Path

import numpy as np

_DATA_BASE = Path("/nfs/projects/spaceitup/rover_navigation/data")
#: Both render trees: the original yaw-0 samples and the yaw-jittered re-render.
DATASET_ROOTS = (_DATA_BASE / "habitat_generated", _DATA_BASE / "habitat_generated_aug")
DATASET_ROOT = DATASET_ROOTS[0]  # back-compat alias for single-root callers
MANIFEST_TRAIN = _DATA_BASE / "manifest_train.jsonl"
MANIFEST_VAL = _DATA_BASE / "manifest_val.jsonl"
IMAGE_NAME = "fpv_enhanced.png"
MAX_WAYPOINTS = 10
HARD_CAP = 12

HABITAT_PROMPT = (
    "<image>\n"
    "You are a rover navigating an indoor environment. The goal is located straight "
    "ahead. Predict the traversable path to the goal as a list of waypoints. Each "
    "waypoint is [x, y, v] where x and y are normalized image coordinates in [0,1] and "
    "v is 1 if the point is on visible, unobstructed ground or 0 if it is obstructed "
    "(hidden behind an obstacle). Then give the goal as [x, y, v]. Answer as JSON: "
    '{"path": [[x, y, v], ...], "goal": [x, y, v]}.'
)

HABITAT_PROMPT_GOAL = (
    "<image>\n"
    "You are a rover navigating an indoor environment. The goal is at [{gx}, {gy}] in "
    "normalized image coordinates, where x runs right and y runs down, and it is "
    "{vis}. Predict the traversable path from the rover to that goal as a list of "
    "waypoints. Each waypoint is [x, y, v] where x and y are normalized image "
    "coordinates in [0,1] and v is 1 if the point is on visible, unobstructed ground "
    "or 0 if it is obstructed (hidden behind an obstacle). The final waypoint must be "
    'the goal itself. Answer as JSON: {{"path": [[x, y, v], ...]}}.'
)


def goal_prompt(goal_wp):
    """Render HABITAT_PROMPT_GOAL for one (x, y, v) goal waypoint."""
    return HABITAT_PROMPT_GOAL.format(
        gx=goal_wp[0],
        gy=goal_wp[1],
        vis="visible" if goal_wp[2] else "hidden behind an obstacle",
    )


def sample_dirs_by_id(roots=DATASET_ROOTS):
    """{sample_id: sample_dir} across every render tree.

    Accepts one root or several. Globs the current
    <root>/<split>/<scene>/samples/<id>/ depth and the older
    <root>/<scene>/samples/<id>/, so a caller pointed at a single scene tree still
    works. Sample ids are unique across the trees (augmented ones carry a _yN
    suffix), so one flat dict is safe.
    """
    if isinstance(roots, (str, Path)):
        roots = (roots,)
    out = {}
    for root in roots:
        for pattern in ("*/samples/*/", "*/*/samples/*/"):
            for d in Path(root).glob(pattern):
                if d.is_dir():
                    out[d.name] = d
    return out


def normalize_points(raw, w, h):
    """(u, v, hidden) pixel points -> (x, y, hidden) normalized by image size."""
    return [(u / w, v / h, hidden) for u, v, hidden in raw]


def _liang_barsky(x0, y0, x1, y1):
    """Clip segment (x0,y0)->(x1,y1) to the unit square. Returns (t0, t1) or None."""
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - 0.0, 1.0 - x0, y0 - 0.0, 1.0 - y0]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None  # parallel and outside
        else:
            t = qi / pi
            if pi < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return t0, t1


def clip_polyline_unit(points):
    """Clip an ordered (x,y,hidden) polyline to [0,1]^2.

    Keeps in-frame vertices, inserts boundary-crossing points (inheriting the
    segment's hidden flag), preserves order and visibility transitions.
    """
    out = []

    def push(pt):
        if not out or abs(out[-1][0] - pt[0]) > 1e-9 or abs(out[-1][1] - pt[1]) > 1e-9 or out[-1][2] != pt[2]:
            out.append(pt)

    for i in range(len(points) - 1):
        x0, y0, h = points[i]
        x1, y1, _ = points[i + 1]
        seg = _liang_barsky(x0, y0, x1, y1)
        if seg is None:
            continue
        t0, t1 = seg
        # exit point keeps points[i]'s flag only if it's a boundary crossing (t1<1);
        # if the segment reaches points[i+1], b is that vertex and takes its own flag
        h_b = points[i + 1][2] if t1 >= 1.0 - 1e-9 else h
        a = (round(x0 + t0 * (x1 - x0), 6), round(y0 + t0 * (y1 - y0), 6), h)
        b = (round(x0 + t1 * (x1 - x0), 6), round(y0 + t1 * (y1 - y0), 6), h_b)
        push(a)
        push(b)
    return out


def resample_with_transitions(clipped, target=MAX_WAYPOINTS, cap=HARD_CAP):
    """Reduce a clipped (x,y,hidden) polyline to [x,y,v] waypoints, arc-length
    uniform-sampled toward `target`.

    `cap` bounds the uniform-fill budget. The first and last point and both sides
    of every visible<->obstructed transition are always kept, so a path with more
    transition points than `cap` keeps all of them -- `cap` is a soft upper bound
    that protects visibility structure and never drops a transition label.
    v = 1 visible, 0 obstructed.
    """
    n = len(clipped)
    if n == 0:
        return []
    xy = np.array([(x, y) for x, y, _ in clipped], dtype=float)
    hid = [h for _, _, h in clipped]
    if n == 1:
        return [(round(xy[0, 0], 3), round(xy[0, 1], 3), 0 if hid[0] else 1)]

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    forced = {0, n - 1}
    for i in range(1, n):
        if hid[i] != hid[i - 1]:
            forced.add(i - 1)
            forced.add(i)

    if total > 0:
        targets = np.linspace(0.0, total, target)
        uni = {int(np.argmin(np.abs(cum - t))) for t in targets}
    else:
        uni = set(forced)

    keep = set(forced) | uni
    if len(keep) > cap:
        extra = sorted(keep - forced)
        while len(forced) + len(extra) > cap and extra:
            extra.pop(len(extra) // 2)  # thin the uniformly-spaced extras from the middle
        keep = forced | set(extra)

    return [
        (round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3), 0 if hid[i] else 1)
        for i in sorted(keep)
    ]


def select_correct_path(fpv_paths, meta):
    """Concatenate the runs of the correct candidate (meta['label']) into
    (u, v, hidden) points. None if the label is missing/misaligned."""
    label = meta.get("label")
    cands = fpv_paths.get("candidates", [])
    if label is None or not (0 <= label < len(cands)) or len(cands) != len(meta.get("candidates", [])):
        return None
    runs = cands[label].get("runs", [])
    return [(float(u), float(v), bool(run["hidden"])) for run in runs for (u, v) in run["uv"]]


def _clamp01(v):
    return round(min(max(v, 0.0), 1.0), 3)


def format_answer(path_wps, goal_wp):
    obj = {
        "path": [[x, y, v] for x, y, v in path_wps],
        "goal": [goal_wp[0], goal_wp[1], goal_wp[2]],
    }
    return json.dumps(obj, separators=(",", ":"))


def format_answer_goal(path_wps):
    """Round-3 answer: path only, its final waypoint being the goal handed in the prompt."""
    obj = {"path": [[x, y, v] for x, y, v in path_wps]}
    return json.dumps(obj, separators=(",", ":"))


#: build_record rejection reasons, counted by the prep script so a drop is never silent.
SKIP_REASONS = (
    "not_in_fov",
    "no_path",
    "path_clipped_away",
    "goal_offscreen",
)


def build_record(sample_dir, *, goal_in_prompt=False, meta_extra=None, reasons=None):
    """Habitat sample dir -> conversation record, or None if filtered/invalid.

    `goal_in_prompt` selects the round-3 framing: the goal goes into the prompt as
    [x, y, v] and the answer carries only the path, whose final waypoint is forced to
    that goal. The candidate polyline already terminates exactly at goal.uv (verified
    across the whole dataset), so the assignment is a guard, not a correction.

    `meta_extra` is merged into a `habitat_meta` sidecar on the record -- invisible to
    TrajectoryDataset, carried through eval so results can be sliced by source/scene.
    `reasons` is an optional dict-like counter that records why a sample was dropped.
    """
    def _skip(reason):
        if reasons is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
        return None

    sample_dir = Path(sample_dir)
    fpv = json.loads((sample_dir / "fpv_paths.json").read_text())
    meta = json.loads((sample_dir / "meta.json").read_text())
    if not meta.get("correct_path_in_fov"):
        return _skip("not_in_fov")
    raw = select_correct_path(fpv, meta)
    if not raw or len(raw) < 2:
        return _skip("no_path")
    w, h = fpv["image_size"]
    clipped = clip_polyline_unit(normalize_points(raw, w, h))
    if len(clipped) < 2:
        return _skip("path_clipped_away")
    path_wps = resample_with_transitions(clipped)
    g = fpv["goal"]
    gx_raw, gy_raw = g["uv"][0] / w, g["uv"][1] / h
    goal_wp = (_clamp01(gx_raw), _clamp01(gy_raw), 0 if g["hidden"] else 1)

    if goal_in_prompt:
        # A goal outside the frame cannot be named in the prompt as an image coordinate,
        # and the clipped path would not reach it. None exist today; count if that changes.
        if not (0.0 <= gx_raw <= 1.0 and 0.0 <= gy_raw <= 1.0):
            return _skip("goal_offscreen")
        path_wps = [*path_wps[:-1], goal_wp]
        prompt, answer = goal_prompt(goal_wp), format_answer_goal(path_wps)
    else:
        prompt, answer = HABITAT_PROMPT, format_answer(path_wps, goal_wp)

    record = {
        "id": sample_dir.name,
        "image": [str(sample_dir / IMAGE_NAME)],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }
    sidecar = {
        "scene_id": meta.get("scene_id"),
        "flavor": meta.get("flavor"),
        "goal_uv_norm": [goal_wp[0], goal_wp[1]],
        "goal_visible": goal_wp[2],
        "goal_in_prompt": bool(goal_in_prompt),
    }
    aug = meta.get("aug") or {}
    sidecar["source"] = "augmented" if aug else "original"
    sidecar["yaw_deg"] = aug.get("yaw_deg")
    sidecar["parent_sample_id"] = aug.get("parent_sample_id")
    if meta_extra:
        sidecar.update(meta_extra)
    record["habitat_meta"] = sidecar
    return record
