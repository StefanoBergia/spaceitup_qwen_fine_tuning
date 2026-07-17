"""Habitat rover-navigation data: select correct path, normalize, clip, resample,
format the path+visibility conversation record.

Source layout: <dataset_root>/<scene>/samples/<scene>_cNNN/ with fpv_enhanced.png,
fpv_paths.json, meta.json. See docs/superpowers/specs/2026-07-17-habitat-path-visibility-design.md.

Coordinates normalize to [0,1] by fpv_paths["image_size"]; visibility v is
1 = visible, 0 = obstructed (v = 0 if the run/goal is "hidden").
"""

import json
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/dataset")
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
        a = (round(x0 + t0 * (x1 - x0), 6), round(y0 + t0 * (y1 - y0), 6), h)
        b = (round(x0 + t1 * (x1 - x0), 6), round(y0 + t1 * (y1 - y0), 6), h)
        push(a)
        push(b)
    return out
