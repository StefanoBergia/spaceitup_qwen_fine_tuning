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
from PIL import Image, ImageDraw

from .landmarks import (
    CAM_HEIGHT_M, FREE, OBJECT_TOP_M, OCC, Camera, Grid, _dilate, _poly_area, column_in_frame, components,
    convex_hull, densify,
)

PASS_M = 0.7              # an obstacle this close to the route is something it passes
START_SKIP_M = 0.3        # ignore what the route passes in its first metres (the rover's own spot)
# The rover's start pose is collision-free for its footprint (objects.json rover.radius_m = 0.5), so
# occupied cells this close to the camera are map artefacts or things below the lens, never occluders
# or landmarks. 00372 (2026-09-30): a 2 m wide strip 0.22-0.45 m ahead blocked every ray at 0.35.
NEAR_CAMERA_M = 0.5
MIN_VISIBLE_CELLS = 4
MIN_AREA = 0.01           # outline area as a fraction of the image
KEEP_NEAR_ROUTE_M = 1.5   # outline only the part of a long blob (a wall) near the route
MAX_VIS_SAMPLES = 200     # visibility rays per blob (each ray is a python loop)
OFFSET_FLOOR_M = 0.1      # score floor so obstacles near the start still count
ROUTE_STEP_M = 0.05
CROP_MARGIN = 0.15        # naming crop = outline box + this fraction of its size
MIN_GAP_M = 0.5           # narrowest opening the route can thread (the rover's radius)
MAX_GAP_PAIRS = 2000      # closest wall-cell pairs tried per blob when looking for a crossing gap
PAIR_ARC_M = 0.5          # two obstacles on opposite sides this close along the route = "between"


@dataclass
class Frame:
    name: str
    grid: Grid
    cam: Camera
    lab: np.ndarray                  # gap-merged blob label per occupied cell, 0 elsewhere
    depth: np.ndarray | None         # depth.png as float (0-255, per-frame min-max), or None
    depth_fit: tuple | None = None   # (scale, offset, resid_m): metres = scale * pixel + offset
    blob_cells: dict | None = None   # label -> (cells_rc, world xz); built on first use


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
    if depth is not None:
        frame.depth_fit = calibrate_depth(depth, frame.cam, grid)
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


DEPTH_BAND_M = 0.3            # object pixels lie within its own depth range +- this
DEPTH_FIT_MAX_RESID_M = 0.3   # worse fit -> column outline, flagged depth_fallback
FIT_MAX_M = 8.0               # floor pixels farther than this are too coarse to fit on
MIN_MASK_PX = 20


def calibrate_depth(depth, cam, grid, step=8):
    """(scale, offset, resid_m) with metres = scale * pixel + offset, or None.

    depth.png is min-max normalised per frame and the range is not stored
    (spaceitup-habitat/src/roverbench/dataset/writer.py:_depth_png). A floor pixel's planar
    depth is known from the camera height alone, so the range is re-fitted on pixels whose
    floor point is a FREE map cell. Pixels showing an object in front of free floor are
    outliers; two trimmed refits drop them. resid_m is measured over ALL candidate pixels, not
    the kept inliers: a residual on the fit's own inliers cannot detect a bad fit (noise passed).
    None when the fitted scale is not positive."""
    H, W = depth.shape
    vals, metres = [], []
    for r in range(int(H * 0.55), H, max(1, H // 64)):
        v = (r + 0.5) / H
        fwd = 0.5 * CAM_HEIGHT_M / (v - 0.5)
        if fwd > FIT_MAX_M:
            continue
        for c in range(0, W, max(1, W // 64)):
            u = (c + 0.5) / W
            xz = cam.pos + fwd * cam.h + (u - 0.5) * 2 * fwd * cam.right
            if depth[r, c] > 0 and grid.cell(grid.rc(xz)[0]) == FREE:
                vals.append(depth[r, c])
                metres.append(fwd)
    if len(vals) < 50:
        return None
    x0, y0 = np.array(vals), np.array(metres)
    x, y = x0, y0
    for _ in range(2):
        a, b = np.polyfit(x, y, 1)
        res = np.abs(a * x + b - y)
        keep = res <= max(DEPTH_BAND_M, np.percentile(res, 80))
        if keep.sum() < 20:
            break
        x, y = x[keep], y[keep]
    a, b = np.polyfit(x, y, 1)
    if a <= 0:
        return None
    return float(a), float(b), float(np.median(np.abs(a * x0 + b - y0)))


def object_outline(frame, vis_xz, top=OBJECT_TOP_M):
    """The column outline trimmed to pixels at the object's own depth (group B: the column
    of a coffee table also covered the sofa behind it, and Cosmos named the sofa)."""
    col = column_outline(frame, vis_xz, top)
    fit = frame.depth_fit
    if frame.depth is None or fit is None or fit[2] > DEPTH_FIT_MAX_RESID_M or len(col["polygon"]) < 3:
        return col
    H, W = frame.depth.shape
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(u * W, v * H) for u, v in col["polygon"]], fill=1)
    metres = fit[0] * frame.depth + fit[1]
    fwd = (np.asarray(vis_xz) - frame.cam.pos) @ frame.cam.h
    keep = (np.asarray(mask, bool) & (frame.depth > 0)
            & (metres >= fwd.min() - DEPTH_BAND_M) & (metres <= fwd.max() + DEPTH_BAND_M))
    if keep.sum() < MIN_MASK_PX:
        return col
    keep, seeded = _seeded_pieces(frame, keep, vis_xz, top)
    if not seeded:
        return col
    ys, xs = np.nonzero(keep)
    pts = np.stack([(xs + 0.5) / W, (ys + 0.5) / H], 1)[:: max(1, len(xs) // 4000)]
    return outline_dict(convex_hull(pts), depth_fallback=False)


PIECE_MAX_PX = 128   # the connected-pieces labelling runs on a mask at most this many px on its long side


def _seeded_pieces(frame, keep, vis_xz, top):
    """(keep', True) with `keep` cut to its 8-connected pieces that hold a seed pixel (the cells'
    projections at heights 0, top/2, top); (keep, False) when no piece holds one. Gate 1: a hull over
    several things at one depth (two chairs and a doorway) made Cosmos name the wrong object."""
    H, W = keep.shape
    st = max(1, -(-max(H, W) // PIECE_MAX_PX))
    small = keep[::st, ::st]
    lab, _ = components(small)
    ids = set()
    for h in (0.0, top / 2, top):
        uv, fwd = frame.cam.uv(np.asarray(vis_xz), h)
        uv = np.nan_to_num(uv[fwd > 1e-6], nan=0.0, posinf=1.0, neginf=0.0)
        c = np.clip((uv[:, 0] * W).astype(int), 0, W - 1) // st
        r = np.clip((uv[:, 1] * H).astype(int), 0, H - 1) // st
        ids.update(int(i) for i in lab[np.minimum(r, lab.shape[0] - 1), np.minimum(c, lab.shape[1] - 1)] if i)
    if not ids:
        return keep, False
    full = np.isin(np.kron(lab, np.ones((st, st), int))[:H, :W], list(ids)) & keep
    return full, bool(full.any())


def candidates(frame, route_xz, outline=None):
    """Every salient obstacle the route passes, in route order (see module docstring)."""
    outline = outline or object_outline
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
            "gap": _threaded_gap(xz, dist, usable, rg), "vis_xz": vis, "outline": o, "route_pts": rg["pts"],
            "score": float(np.sqrt(o["area"]) * max(rg["offset"][i], OFFSET_FLOOR_M)),
        })
    return sorted(out, key=lambda c: c["arc_m"])


def _crosses(p, q, pts):
    """True when segment p-q intersects the polyline `pts` (2-D)."""
    a, b = pts[:-1], pts[1:]
    r = q - p

    def cr(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    d1, d2 = cr(r, a - p), cr(r, b - p)
    d3, d4 = cr(b - a, p - a), cr(b - a, q - a)
    return bool(np.any((d1 * d2 <= 0) & (d3 * d4 <= 0)))


def _crossing_pair(P, Q, pts, ok=None):
    """(p, q): the closest cell pair, one from each of P and Q, whose segment crosses the route
    polyline `pts` (and satisfies the optional mask `ok`) -- the opening the route threads --
    or None when there is none or that opening is narrower than MIN_GAP_M."""
    d = np.linalg.norm(P[:, None] - Q[None], axis=2)
    idx = np.argwhere(d >= 0 if ok is None else ok)
    if not len(idx):
        return None
    order = np.argsort(d[tuple(idx.T)])[:MAX_GAP_PAIRS]
    for i, j in idx[order]:
        if _crosses(P[i], Q[j], pts):
            return (P[i], Q[j]) if d[i, j] >= MIN_GAP_M else None
    return None


def _threaded_gap(xz, dist, usable, rg):
    """(p, q) when the route threads through this blob: cells within PASS_M on both sides at
    about the same arclength, at least MIN_GAP_M apart, with the segment between them crossing
    the route (a doorway in one connected wall), else None."""
    close = np.nonzero(dist.min(1) <= PASS_M)[0]
    if not len(close):
        return None
    k = dist[close].argmin(1)
    i = usable[k]
    s = np.einsum("ij,ij->i", xz[close] - rg["pts"][i], rg["right"][i])
    left, right = close[s < 0], close[s > 0]
    if not (len(left) and len(right)):
        return None
    arc = rg["arc"][i]
    al, ar = arc[s < 0], arc[s > 0]
    return _crossing_pair(xz[left], xz[right], rg["pts"], np.abs(al[:, None] - ar[None]) <= PAIR_ARC_M)


MAX_EVENTS = 3
WALL_EXTENT_M = 1.5        # two blobs this long on both sides of the route = an opening
OPENING_TOP_M = 2.0        # an opening's outline reaches this high (a door)
STOP_NEAR_M = 1.0
AFTER_LAST_M = 1.5
STOP_FAR_M = 4.0
APPROACH_COS = 0.866       # within 30 degrees of the final heading, ahead or behind
DISTANCE_STEPS = (1.0, 1.5, 2.0, 3.0, 4.0)
REASONS = ("no_state", "bad_candidate", "no_stop_anchor", "error")
CHECK_CELLS = 40           # cells per object kept for the eval's follow check


def opening_outline(frame, p, q):
    """Floor-to-OPENING_TOP_M quad between the two cells flanking a gap."""
    ends = np.array([p, q], float)
    lo, _ = frame.cam.uv(ends)
    hi, _ = frame.cam.uv(ends, OPENING_TOP_M)
    o = outline_dict(convex_hull(np.vstack([lo, hi])), depth_fallback=True)
    o["gap_xz"] = [[round(float(v), 3) for v in p], [round(float(v), 3) for v in q]]
    return o


def build_events(frame, cands):
    """Top-MAX_EVENTS events by score, returned in route order."""
    events, used = [], set()
    for k, a in enumerate(cands):
        if k in used:
            continue
        used.add(k)
        if a["gap"] is not None and a["extent_m"] >= WALL_EXTENT_M:
            events.append({"type": "through", "side": None, "members": [k], "arc_m": a["arc_m"],
                           "score": a["score"], "opening": opening_outline(frame, *a["gap"])})
            continue
        m = next((m for m, b in enumerate(cands) if m not in used and b["side"] != a["side"]
                  and abs(b["arc_m"] - a["arc_m"]) <= PAIR_ARC_M), None)
        if m is None:
            events.append({"type": "pass", "side": a["side"], "members": [k], "arc_m": a["arc_m"],
                           "score": a["score"]})
            continue
        b = cands[m]
        used.add(m)
        ev = {"type": "between", "side": None, "members": [k, m],
              "arc_m": min(a["arc_m"], b["arc_m"]), "score": a["score"] + b["score"]}
        if a["extent_m"] >= WALL_EXTENT_M and b["extent_m"] >= WALL_EXTENT_M:
            pq = _crossing_pair(a["vis_xz"], b["vis_xz"], a["route_pts"])
            if pq is not None:
                ev.update(type="through", opening=opening_outline(frame, *pq))
        events.append(ev)
    top = sorted(events, key=lambda e: -e["score"])[:MAX_EVENTS]
    return sorted(top, key=lambda e: e["arc_m"])


def _blob_cells(frame):
    """label -> (cells_rc, world xz) for every blob; one pass over the grid, cached on the frame."""
    if frame.blob_cells is None and not (frame.lab > 0).any():
        frame.blob_cells = {}
    if frame.blob_cells is None:
        rc = np.argwhere(frame.lab > 0)
        lab = frame.lab[rc[:, 0], rc[:, 1]]
        order = np.argsort(lab, kind="stable")
        rc, lab = rc[order], lab[order]
        cuts = np.nonzero(np.diff(lab))[0] + 1
        frame.blob_cells = {int(g[0]): (c, frame.grid.xz(c))
                            for g, c in zip(np.split(lab, cuts), np.split(rc, cuts))}
    return frame.blob_cells


def _anchor(frame, blob, goal, cells, xz, d):
    """The stop anchor for one blob (visible part within STOP_FAR_M of the goal), or None."""
    vis = visible_cells(frame, cells[d <= STOP_FAR_M + KEEP_NEAR_ROUTE_M], blob)
    if len(vis) < MIN_VISIBLE_CELLS:
        return None
    o = object_outline(frame, vis)
    if o["area"] < MIN_AREA:
        return None
    return {"blob": blob, "near_xz": xz[d.argmin()], "dist": float(d.min()), "vis_xz": vis, "outline": o}


def _blobs_near(frame, goal, max_dist):
    """(blob, cells, xz, d-to-goal) for the blobs with a cell within max_dist of the goal."""
    out = []
    for b, (cells, xz) in _blob_cells(frame).items():
        d = np.linalg.norm(xz - goal, axis=1)
        if d.min() <= max_dist:
            out.append((b, cells, xz, d))
    return out


def _event_anchor(event, cands):
    """The stop anchor standing for an event: its first candidate, or the opening of a through event."""
    if event["type"] == "through":
        blobs = [cands[m]["blob"] for m in event["members"]]
        gap = event["opening"]["gap_xz"]
        return {"blob": blobs[0], "vis_xz": gap, "outline": event["opening"],
                "key": _gap_key(blobs, gap), "kind": "opening"}
    c = cands[event["members"][0]]
    return {"blob": c["blob"], "vis_xz": c["vis_xz"], "outline": c["outline"]}


def stop_phrase(frame, route_xz, events, cands=None):
    """Where the route ends: near an object, just after the last event, or a rough distance
    from an object on the line of approach; None when nothing fits (the route is dropped).
    Visibility and outlines (the expensive part) are only computed for the blobs each rule needs."""
    rg = route_geometry(route_xz)
    goal = rg["pts"][-1]
    back = rg["pts"][max(0, len(rg["pts"]) - 11)]          # ~0.5 m before the goal
    head = goal - back
    head /= max(np.linalg.norm(head), 1e-9)
    right = np.array([-head[1], head[0]])
    blobs = _blobs_near(frame, goal, STOP_FAR_M)
    near = [x for x in (_anchor(frame, b, goal, c, xz, d) for b, c, xz, d in blobs if d.min() <= STOP_NEAR_M) if x]
    if near:
        x = min(near, key=lambda x: x["dist"])
        v = x["near_xz"] - goal
        along, lat = float(v @ head), float(v @ right)
        rel = "in front of" if along > abs(lat) else "just past" if along < -abs(lat) else "next to"
        return {"type": "near", "relation": rel, "anchor": x}
    if events and rg["arc"][-1] - events[-1]["arc_m"] <= AFTER_LAST_M:
        return {"type": "after_last", "relation": "just after", "anchor": None}
    # rough distance past the last thing passed, when nothing visible is near the goal
    if events and cands is not None:
        remaining = rg["arc"][-1] - events[-1]["arc_m"]
        if remaining <= STOP_FAR_M:
            step = min(DISTANCE_STEPS, key=lambda s: abs(s - remaining))
            return {"type": "distance", "relation": "past", "distance_m": step,
                    "anchor": _event_anchor(events[-1], cands)}
    far = []
    for b, cells, xz, d in blobs:
        v = xz[d.argmin()] - goal
        if abs(v @ head) / max(np.linalg.norm(v), 1e-9) < APPROACH_COS:
            continue
        x = _anchor(frame, b, goal, cells, xz, d)
        if x is not None:
            far.append((x["dist"], x, bool(v @ head > 0)))
    if far:
        dist, x, ahead = min(far, key=lambda t: t[0])
        step = min(DISTANCE_STEPS, key=lambda s: abs(s - dist))
        return {"type": "distance", "relation": "before" if ahead else "past", "distance_m": step, "anchor": x}
    return None


def _none(reason, **diag):
    return {"reason": reason, "events": [], "stop": None, "objects": {}, "diag": diag}


def _cells(xz, n=CHECK_CELLS):
    xz = np.asarray(xz, float)
    if len(xz) > n:
        xz = xz[np.linspace(0, len(xz) - 1, n).astype(int)]
    return [[round(float(a), 3), round(float(b), 3)] for a, b in xz]


def _gap_key(blobs, gap_xz):
    """Object-id suffix of an opening. One blob can hold several doorways and gap_xz depends on
    the route, so a single-blob gap is named by its midpoint; a pair by its sorted blob ids."""
    if len(blobs) == 1:
        mx, mz = np.mean(gap_xz, axis=0)
        return f"gap{blobs[0]}@{mx:.1f},{mz:.1f}"
    return "gap" + "-".join(str(b) for b in sorted(blobs))


def describe_route(sample_dir, candidate_index):
    """The events + stop phrase for one candidate route of one frame (see module docstring)."""
    sd = Path(sample_dir)
    if not (sd / "state.npz").exists():
        return _none("no_state")
    meta, frame = load_frame(sd)
    cm = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cm)) or len(cm[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    route = np.asarray(cm[candidate_index]["polyline_xz"], float)
    cands = candidates(frame, route)
    events = build_events(frame, cands)
    stop = stop_phrase(frame, route, events, cands)
    if stop is None:
        return _none("no_stop_anchor", n_events=len(events))
    objects = {}

    def add(key, kind, outline, cells):
        oid = f"{frame.name}:{key}"
        objects.setdefault(oid, {"object_id": oid, "kind_hint": kind,
                                 **{k: outline[k] for k in ("polygon", "bbox", "crop_box", "area", "depth_fallback")},
                                 "cells_xz": _cells(cells)})
        return oid

    rows = []
    for e in events:
        blobs = [cands[m]["blob"] for m in e["members"]]
        if e["type"] == "through":
            ids = [add(_gap_key(blobs, e["opening"]["gap_xz"]), "opening", e["opening"], e["opening"]["gap_xz"])]
        else:
            ids = [add(f"b{cands[m]['blob']}", "object", cands[m]["outline"], cands[m]["vis_xz"]) for m in e["members"]]
        row = {"type": e["type"], "side": e["side"], "objects": ids, "arc_m": round(e["arc_m"], 2)}
        if e["type"] == "through":
            row["gap_xz"] = e["opening"]["gap_xz"]
        rows.append(row)
    st = {"type": stop["type"], "relation": stop["relation"], "distance_m": stop.get("distance_m"), "object": None}
    if stop["anchor"] is not None:
        a = stop["anchor"]
        st["object"] = add(a.get("key", f"b{a['blob']}"), a.get("kind", "object"), a["outline"], a["vis_xz"])
    return {"reason": None, "events": rows, "stop": st, "objects": objects,
            "goal_xz": [round(float(v), 3) for v in route[-1]],
            "pose_xz": [float(v) for v in meta["pose_xz"]], "heading_xz": [float(v) for v in meta["heading_xz"]],
            "route_len_m": round(float(route_geometry(route)["arc"][-1]), 2)}


def collect_objects(rows):
    """(rows without their "objects", one object row per id). Both routes of a pair share
    their frame's objects; the first route's outline wins (its name serves both)."""
    objects, kept = {}, []
    for r in rows:
        r = dict(r)
        for oid, o in (r.pop("objects", None) or {}).items():
            objects.setdefault(oid, {**o, "image": r["image"]})
        kept.append(r)
    return kept, list(objects.values())
