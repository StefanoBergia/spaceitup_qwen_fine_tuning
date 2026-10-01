"""Route descriptions (prompt d1): what a route passes, in order, and where it stops.

The i4 spot check (2026-09-30) showed one geometry-chosen landmark per route is often the
wrong one — the straight entry->goal line grazes a wall while the route bends round a pillar —
and one landmark under-specifies a path once the goal point leaves the prompt. Here every
salient obstacle the route passes is a candidate; the best three become events (pass /
between / through), a stop phrase says where the route ends, and each object's outline is
trimmed with the frame's depth so it does not swallow what stands behind. Cosmos later only
names the objects and phrases the sentence (rover_vlm.descriptions).

Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-09-30-route-descriptions-design.md
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .landmarks import (
    OBJECT_TOP_M, OCC, Camera, Grid, _dilate, _poly_area, column_in_frame, components,
    convex_hull, densify,
)

PASS_M = 0.7              # an obstacle this close to the route is something it passes
START_SKIP_M = 0.3        # ignore what the route passes in its first metres (the rover's own spot)
# The rover's start pose is collision-free: occupied cells this close to the camera are map
# artefacts or things below the lens, never occluders or landmarks (00372, 2026-09-30).
NEAR_CAMERA_M = 0.35
MIN_VISIBLE_CELLS = 4
MIN_AREA = 0.01           # outline area as a fraction of the image
KEEP_NEAR_ROUTE_M = 1.5   # outline only the part of a long blob (a wall) near the route
MAX_VIS_SAMPLES = 200     # visibility rays per blob (each ray is a python loop)
OFFSET_FLOOR_M = 0.1      # score floor so obstacles near the start still count
ROUTE_STEP_M = 0.05
CROP_MARGIN = 0.15        # naming crop = outline box + this fraction of its size
PAIR_ARC_M = 0.5          # two obstacles on opposite sides this close along the route = "between"


@dataclass
class Frame:
    name: str
    grid: Grid
    cam: Camera
    lab: np.ndarray                  # gap-merged blob label per occupied cell, 0 elsewhere
    depth: np.ndarray | None         # depth.png as float (0-255, per-frame min-max), or None
    depth_fit: tuple | None = None   # (scale, offset, resid_m): metres = scale * pixel + offset


def load_frame(sample_dir):
    """(meta, Frame) for one sample dir. Blobs one empty cell apart are one obstacle."""
    sd = Path(sample_dir)
    meta = json.loads((sd / "meta.json").read_text())
    grid = Grid.load(sd / "state.npz")
    occ = grid.s == OCC
    merged, _ = components(_dilate(occ))
    dp = sd / "depth.png"
    depth = np.asarray(Image.open(dp), dtype=float) if dp.exists() else None
    frame = Frame(sd.name, grid, Camera(meta["pose_xz"], meta["heading_xz"]),
                  np.where(occ, merged, 0), depth)
    return meta, frame


def first_blocker(frame, xz):
    """Blob label of the first occupied cell between the camera and xz, or None.

    Cells within NEAR_CAMERA_M of the camera are skipped (see NEAR_CAMERA_M)."""
    for p in densify(np.array([frame.cam.pos, xz], float), frame.grid.mpp / 2)[1:]:
        if np.linalg.norm(p - frame.cam.pos) <= NEAR_CAMERA_M:
            continue
        r, c = (int(round(v)) for v in frame.grid.rc(p)[0])
        if frame.grid.cell((r, c)) == OCC:
            return int(frame.lab[r, c])
    return None


def visible_cells(frame, cells_rc, label):
    """World xz of the cells of blob `label` the camera sees (column in frame, not hidden)."""
    cells_rc = np.asarray(cells_rc)
    if len(cells_rc) > MAX_VIS_SAMPLES:
        cells_rc = cells_rc[np.linspace(0, len(cells_rc) - 1, MAX_VIS_SAMPLES).astype(int)]
    xz = frame.grid.xz(cells_rc)
    ok = column_in_frame(frame.cam, xz) & (np.linalg.norm(xz - frame.cam.pos, axis=1) > NEAR_CAMERA_M)
    vis = [q for q, k in zip(xz, ok) if k and first_blocker(frame, q) in (None, label)]
    return np.array(vis, float).reshape(-1, 2)


def route_geometry(route_xz):
    """Resampled route: points, arclength, right-of-travel normals, offset from entry->goal line."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    right = np.stack([-d[:, 1], d[:, 0]], 1)        # Camera.right's handedness
    a, g = pts[0], pts[-1]
    ch = (g - a) / max(np.linalg.norm(g - a), 1e-9)
    offset = np.abs((pts - a) @ np.array([-ch[1], ch[0]]))
    return {"pts": pts, "arc": arc, "right": right, "offset": offset}


def crop_box(bbox, margin=CROP_MARGIN):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    m = margin * max(w, h, 0.05)
    return [max(0.0, bbox[0] - m), max(0.0, bbox[1] - m), min(1.0, bbox[2] + m), min(1.0, bbox[3] + m)]


def outline_dict(poly, depth_fallback):
    poly = [[float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))] for u, v in poly]
    if len(poly) < 3:
        return {"polygon": poly, "bbox": None, "crop_box": None, "area": 0.0, "depth_fallback": depth_fallback}
    us, vs = zip(*poly)
    bbox = [min(us), min(vs), max(us), max(vs)]
    return {"polygon": poly, "bbox": bbox, "crop_box": crop_box(bbox),
            "area": round(_poly_area(poly), 5), "depth_fallback": depth_fallback}


def column_outline(frame, vis_xz, top=OBJECT_TOP_M):
    """i4's outline: hull of the visible cells' floor-to-`top` columns (no depth)."""
    pts = np.vstack([frame.cam.uv(vis_xz)[0], frame.cam.uv(vis_xz, top)[0]])
    return outline_dict(convex_hull(pts), depth_fallback=True)


def candidates(frame, route_xz, outline=None):
    """Every salient obstacle the route passes, in route order (see module docstring)."""
    outline = outline or column_outline
    rg = route_geometry(route_xz)
    usable = np.nonzero(rg["arc"] > START_SKIP_M)[0]
    if not len(usable):
        return []
    pts = rg["pts"][usable]
    out = []
    for b in np.unique(frame.lab[frame.lab > 0]):
        cells = np.argwhere(frame.lab == b)
        xz = frame.grid.xz(cells)
        dist = np.linalg.norm(xz[:, None] - pts[None], axis=2)
        j, k = np.unravel_index(dist.argmin(), dist.shape)
        if dist[j, k] > PASS_M:
            continue
        near = dist.min(1) <= KEEP_NEAR_ROUTE_M
        vis = visible_cells(frame, cells[near], int(b))
        if len(vis) < MIN_VISIBLE_CELLS:
            continue
        o = outline(frame, vis)
        if o["area"] < MIN_AREA:
            continue
        i = usable[k]
        on_right = (xz[j] - rg["pts"][i]) @ rg["right"][i] > 0
        ext = xz.max(0) - xz.min(0)
        out.append({
            "blob": int(b), "pass_m": float(dist[j, k]), "arc_m": float(rg["arc"][i]),
            "side": "left" if on_right else "right",
            "offset_m": float(rg["offset"][i]), "extent_m": float(np.hypot(*ext)),
            "gap": _threaded_gap(xz, dist, usable, rg), "vis_xz": vis, "outline": o,
            "score": float(np.sqrt(o["area"]) * max(rg["offset"][i], OFFSET_FLOOR_M)),
        })
    return sorted(out, key=lambda c: c["arc_m"])


def _threaded_gap(xz, dist, usable, rg):
    """(p, q) when the route threads through this blob (cells within PASS_M on both sides at
    about the same arclength — a doorway in one connected wall), else None."""
    close = np.nonzero(dist.min(1) <= PASS_M)[0]
    if not len(close):
        return None
    k = dist[close].argmin(1)
    i = usable[k]
    s = np.einsum("ij,ij->i", xz[close] - rg["pts"][i], rg["right"][i])
    left, right = close[s < 0], close[s > 0]
    if not (len(left) and len(right)):
        return None
    arc = dict(zip(close, rg["arc"][i]))
    d = np.linalg.norm(xz[left][:, None] - xz[right][None], axis=2)
    a, b = np.unravel_index(d.argmin(), d.shape)
    if abs(arc[left[a]] - arc[right[b]]) > PAIR_ARC_M:
        return None
    return (xz[left[a]], xz[right[b]])
