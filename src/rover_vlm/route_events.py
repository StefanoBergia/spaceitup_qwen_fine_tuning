"""Route descriptions (prompt d1): what a route passes, in order, and where it stops.

The i4 spot check (2026-09-30) showed one geometry-chosen landmark per route is often the
wrong one — the straight entry->goal line grazes a wall while the route bends round a pillar —
and one landmark under-specifies a path once the goal point leaves the prompt. Here every
salient obstacle the route passes is a candidate; the best three become events (pass of one or
two objects / wall the route runs alongside / between / through -- Addendum A merges adjacent
opposite passes into between), a stop phrase says where the route ends (coarse amounts: just /
a little / well), and each outline is trimmed with the frame's depth so it does not swallow what stands behind. Cosmos later only
names the objects and phrases the sentence (rover_vlm.descriptions).

A blob the route runs through (closest distance < TOUCH_M) has no defined side: candidates() marks it "touch" and build_events lets it join only a through event (a doorway jamb), never a pass / wall / between (Addendum A2).

d2 (2026-10-13): a route that ends >= 45 deg off its start->goal line gets a "turn" event (behind / round an object, or bare); routes that double back on themselves are dropped as "spur".

d2 T2 (2026-10-14): a route whose end is out of sight is described up to where it disappears and ends "behind" the object hiding it (or with its hidden turn); events after the disappearance are dropped; a visible "well past" stop re-anchors within 1.5 m.

d3 (2026-11-10, describe_xz(stops="d3")): an end in view no longer gets a distance amount; it stops "behind / in front of / next to" a non-wall object within CLOSE_M (object_relation), else at a place read from the occupancy grid (stop_places.place_at: doorway, corner, end of the hallway, in the hallway, by the wall, middle of the room), else (Addendum A1 backup) "in front of" a visible wall within CLOSE_M ahead (wall_ahead); no fit -> "no_stop_place". Spec: docs/superpowers/specs/2026-11-10-stop-phrases-d3-design.md.

d3 Addendum A2 (2026-11-16): a wall-sized blob on the 2D map can be low furniture (a sofa set, a dining set); each map cell gets a height class from the depth image (cell_heights) and only a "by the wall" stop is looked at again with low furniture as furniture; when that finds no stop, "by the wall" stays.

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
from .stop_places import place_at

PASS_M = 0.7              # an obstacle this close to the route is something it passes
# Gate 1 r2 (2026-10-02): a blob the route runs *through* (closest distance ~0.01 m, vs the planner's
# ~0.45-0.5 m clearance everywhere else) has no defined side at the closest point -- left/right is a
# coin flip (8% of routes, 64/67 of them walls; bimodal: <0.05 m or >=0.2 m). Not a candidate.
TOUCH_M = 0.2
START_SKIP_M = 0.3        # ignore what the route passes in its first metres (the rover's own spot)
# The rover's start pose is collision-free for its footprint (objects.json rover.radius_m = 0.5), so
# occupied cells this close to the camera are map artefacts or things below the lens, never occluders
# or landmarks. 00372 (2026-09-30): a 2 m wide strip 0.22-0.45 m ahead blocked every ray at 0.35.
NEAR_CAMERA_M = 0.5
MIN_VISIBLE_CELLS = 4
MIN_AREA = 0.005          # outline area as a fraction of the image (tight floor-free outlines of thin posts are ~0.01)
KEEP_NEAR_ROUTE_M = 1.5   # outline only the part of a long blob (a wall) near the route
MAX_VIS_SAMPLES = 200     # visibility rays per blob (each ray is a python loop)
OFFSET_FLOOR_M = 0.1      # score floor so obstacles near the start still count
ROUTE_STEP_M = 0.05
CROP_MARGIN = 0.15        # naming crop = outline box + this fraction of its size
MIN_GAP_M = 0.5           # narrowest opening the route can thread (the rover's radius)
MAX_GAP_PAIRS = 2000      # closest wall-cell pairs tried per blob when looking for a crossing gap
PAIR_ARC_M = 0.5          # two wall blobs on opposite sides this close along the route may flank an opening
ALONG_MIN_M = 1.0         # route length within PASS_M of a long blob for it to be a wall the route runs alongside
WALL_KEEP_M = 0.8         # a wall's cells / outline cover only the stretch this close to the route
MERGE_ARC_M = 3.0         # adjacent events at most this far apart along the route may merge (between / two-object pass)
AMOUNT_STEPS = ((1.5, "just"), (3.0, "a little"), (float("inf"), "well"))   # coarse stop amounts (Gate 1)


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


def _first_hit(frame, xz):
    """(blob label, hit xz) of the first occupied cell between the camera and xz, or (None, None).

    Cells within NEAR_CAMERA_M of the camera are skipped (see NEAR_CAMERA_M)."""
    for p in densify(np.array([frame.cam.pos, xz], float), frame.grid.mpp / 2)[1:]:
        if np.linalg.norm(p - frame.cam.pos) <= NEAR_CAMERA_M:
            continue
        r, c = (int(round(v)) for v in frame.grid.rc(p)[0])
        if frame.grid.cell((r, c)) == OCC:
            return int(frame.lab[r, c]), p
    return None, None


def first_blocker(frame, xz):
    """Blob label of the first occupied cell between the camera and xz, or None (see _first_hit)."""
    return _first_hit(frame, xz)[0]


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
    keep, seeded = _seeded_pieces(frame, keep, metres, vis_xz, top)
    # no seeded piece: hull the floor-free band itself; the column outline (the group-B shape) only
    # when nothing but floor was at the object's depth (final review I2: low objects, no visible top)
    if not seeded and keep.sum() < MIN_MASK_PX:
        return col
    ys, xs = np.nonzero(keep)
    pts = np.stack([(xs + 0.5) / W, (ys + 0.5) / H], 1)[:: max(1, len(xs) // 4000)]
    return outline_dict(convex_hull(pts), depth_fallback=False)


FLOOR_TOL_M = 0.2    # floor at the object's depth would join neighbouring objects into one piece
PIECE_MAX_PX = 128   # the connected-pieces labelling runs on a mask at most this many px on its long side
LOW_SEED_M = 0.25    # extra seed height: an object lower than top/2 still holds a seed (final review I2)


def _seeded_pieces(frame, keep, metres, vis_xz, top):
    """(keep', True) with `keep` cut to its 8-connected pieces that hold a seed pixel (the cells'
    projections at heights min(top, 0.25), top/2, top -- the low seed catches objects shorter than
    top/2; height 0 lands on floor); floor pixels are removed first;
    (floor-free keep, False) when no piece holds one. Gate 1: a hull over
    several things at one depth (two chairs and a doorway) made Cosmos name the wrong object."""
    H, W = keep.shape
    v = (np.arange(H)[:, None] + 0.5) / H
    floor = 0.5 * CAM_HEIGHT_M / np.maximum(v - 0.5, 1e-6)
    keep = keep & ~((v > 0.5) & (np.abs(metres - floor) < FLOOR_TOL_M))
    st = max(1, -(-max(H, W) // PIECE_MAX_PX))
    small = keep[::st, ::st]
    lab, _ = components(small)
    ids = set()
    for h in (min(top, LOW_SEED_M), top / 2, top):
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
    step = np.gradient(rg["arc"][usable]) if len(usable) > 1 else np.zeros(1)
    out = []
    for b in np.unique(frame.lab[frame.lab > 0]):
        cells = np.argwhere(frame.lab == b)
        xz = frame.grid.xz(cells)
        dist = np.linalg.norm(xz[:, None] - pts[None], axis=2)
        j, k = np.unravel_index(dist.argmin(), dist.shape)
        if dist[j, k] > PASS_M:
            continue
        gap = _threaded_gap(xz, dist, usable, rg)
        ext = xz.max(0) - xz.min(0)
        is_doorway = gap is not None and float(np.hypot(*ext)) >= WALL_EXTENT_M
        touch = bool(dist[j, k] < TOUCH_M and not is_doorway)   # the route runs through it: no side (Addendum A2)
        alongside = dist.min(0) <= PASS_M
        along = float(step[alongside].sum())
        along_end = float(rg["arc"][usable][alongside].max())   # where the route leaves the blob
        wall = bool(np.hypot(*ext) >= WALL_EXTENT_M and along >= ALONG_MIN_M)
        near = dist.min(1) <= (WALL_KEEP_M if wall else KEEP_NEAR_ROUTE_M)
        vis = visible_cells(frame, cells[near], int(b))
        if len(vis) < MIN_VISIBLE_CELLS:
            continue
        o = outline(frame, vis)
        if o["area"] < MIN_AREA:
            continue
        i = usable[k]
        on_right = (xz[j] - rg["pts"][i]) @ rg["right"][i] > 0
        out.append({
            "blob": int(b), "pass_m": float(dist[j, k]), "arc_m": float(rg["arc"][i]),
            "side": "left" if on_right else "right",
            "offset_m": float(rg["offset"][i]), "extent_m": float(np.hypot(*ext)),
            "wall": wall, "touch": touch, "along_m": along, "along_end_arc_m": along_end,
            "gap": gap, "vis_xz": vis, "outline": o, "route_pts": rg["pts"],
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
REANCHOR_M = 1.5          # a visible "well past" stop first tries the near rule this far out (d2 T2)
AFTER_LAST_M = 1.5
STOP_FAR_M = 4.0           # the line-of-approach rule looks this far from the goal (no limit past the last event)
APPROACH_COS = 0.866       # within 30 degrees of the final heading, ahead or behind
STRAIGHT_MAX_M = 0.3          # a zero-event route is described "go straight": its max offset from entry->goal must be within this
REASONS = ("no_state", "bad_candidate", "spur", "no_stop_anchor", "no_behind_anchor", "no_events_bent", "no_stop_place",
           "error")
CLOSE_M = 1.5              # d3: an object this close to a visible end is the stop's anchor (~1 m of gap)
BEHIND_GAP_M = 0.3         # d3 "behind X": X spans the goal's image column and is this much nearer the camera
CHECK_CELLS = 40           # cells per object kept for the eval's follow check


def opening_outline(frame, p, q):
    """Floor-to-OPENING_TOP_M quad between the two cells flanking a gap."""
    ends = np.array([p, q], float)
    lo, _ = frame.cam.uv(ends)
    hi, _ = frame.cam.uv(ends, OPENING_TOP_M)
    o = outline_dict(convex_hull(np.vstack([lo, hi])), depth_fallback=True)
    o["gap_xz"] = [[round(float(v), 3) for v in p], [round(float(v), 3) for v in q]]
    return o


OTHER_SIDE = {"left": "right", "right": "left"}


def _end_arc(c):
    """Where the route is done with a candidate: a wall where the route leaves it, else its closest approach."""
    return c["along_end_arc_m"] if c["wall"] else c["arc_m"]


def _event(kind, side, members, cands, **extra):
    """An event row; arc_m = where it starts, end_arc_m = where the route is done with its last member.

    A through event is done at its crossing (arc_m): its members are usually wall-flagged, and with
    gap-merged blobs the walls beyond the door are the same blob, so along_end_arc_m would put the
    end far past the door (final review I1: "stop just after the doorway" 3.5 m past it)."""
    arc = min(cands[m]["arc_m"] for m in members)
    end = arc if kind == "through" else max(_end_arc(cands[m]) for m in members)
    return {"type": kind, "side": side, "members": list(members), "arc_m": arc, "end_arc_m": end,
            "score": sum(cands[m]["score"] for m in members), **extra}


def _obstacle_side(e):
    """The route side the obstacle of a pass/wall event is on: a wall's side IS that side (the
    rover's own side), a pass's side is the side the route passes on, so the obstacle is opposite."""
    return e["side"] if e["type"] == "wall" else OTHER_SIDE[e["side"]]


def _merge(frame, a, b, cands):
    """One event for two adjacent ones (Addendum A 3), or None when they stay apart: obstacles on
    opposite sides -> between (two wall-long blobs flanking a crossing opening -> through, as before);
    two objects on the same side -> one pass of both. Never with a through event, never one blob twice,
    never farther apart along the route than MERGE_ARC_M (from a's end -- a wall's end is where the route
    leaves it -- to b's start). build_events pairs greedily left to right; each event merges at most once."""
    if "through" in (a["type"], b["type"]) or b["arc_m"] - a["end_arc_m"] > MERGE_ARC_M:
        return None
    members = a["members"] + b["members"]
    if len({cands[m]["blob"] for m in members}) < len(members):
        return None
    touching = any(cands[m]["touch"] for m in members)   # a touching blob's side is undefined: only a through can use it
    if touching or _obstacle_side(a) != _obstacle_side(b):
        ca, cb = cands[a["members"][0]], cands[b["members"][0]]
        if (ca["extent_m"] >= WALL_EXTENT_M and cb["extent_m"] >= WALL_EXTENT_M
                and abs(cb["arc_m"] - ca["arc_m"]) <= PAIR_ARC_M):
            pq = _crossing_pair(ca["vis_xz"], cb["vis_xz"], ca["route_pts"])
            if pq is not None:
                return _event("through", None, members, cands, opening=opening_outline(frame, *pq))
        return None if touching else _event("between", None, members, cands)
    if a["type"] == b["type"] == "pass":
        return _event("pass", a["side"], members, cands)
    return None


def build_events(frame, cands):
    """Top-MAX_EVENTS events by score, returned in route order.

    Each candidate is classified first: a crossing gap in a wall-long blob -> through (wins over
    wall, Review Focus 1); a wall the route runs alongside -> wall (side = the rover's side of it);
    anything else -> pass. Then adjacent single events merge left to right (see _merge)."""
    single = []
    for k, a in enumerate(cands):
        if a["gap"] is not None and a["extent_m"] >= WALL_EXTENT_M:
            single.append(_event("through", None, [k], cands, opening=opening_outline(frame, *a["gap"])))
        elif a["wall"]:
            single.append(_event("wall", OTHER_SIDE[a["side"]], [k], cands))
        else:
            single.append(_event("pass", a["side"], [k], cands))
    touch = [cands[e["members"][0]]["touch"] for e in single]
    # A touching blob (the route runs through it, Addendum A2) has no side: it may only be one half of a
    # through (a doorway jamb). Pair those first; every other touching single is dropped from the sequence,
    # so its clear neighbours still merge exactly as they did before the rule.
    seq, i = [], 0
    while i < len(single):
        m = None
        if i + 1 < len(single) and (touch[i] or touch[i + 1]):
            m = _merge(frame, single[i], single[i + 1], cands)
            m = m if m and m["type"] == "through" else None
        if m:
            seq.append(m)
            i += 2
        else:
            if not touch[i]:
                seq.append(single[i])
            i += 1
    events, i = [], 0
    while i < len(seq):
        m = _merge(frame, seq[i], seq[i + 1], cands) if i + 1 < len(seq) else None
        events.append(m or seq[i])
        i += 2 if m else 1
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


def amount(metres):
    """The coarse stop amount for a distance: "just" / "a little" / "well" (AMOUNT_STEPS)."""
    return next(word for limit, word in AMOUNT_STEPS if metres < limit)


def _wall_like(xz):
    return float(np.hypot(*np.ptp(np.asarray(xz), axis=0))) >= WALL_EXTENT_M


def _event_anchor(event, cands, route_len):
    """(anchor, arc_m) standing for an event in a "past" stop, or None.

    A through event -> its opening. Otherwise the member passed last that can be stopped past:
    an object from its closest approach (arc_m); a wall-like member (extent >= WALL_EXTENT_M, Review
    Focus 3) only once the route has left it -- from the last route point within PASS_M of it
    (along_end_arc_m), and only when more than AFTER_LAST_M of route follows (controller ruling)."""
    if event["type"] == "through":
        blobs = [cands[m]["blob"] for m in event["members"]]
        gap = event["opening"]["gap_xz"]
        return {"blob": blobs[0], "vis_xz": gap, "outline": event["opening"],
                "key": _gap_key(blobs, gap), "kind": "opening"}, event["arc_m"]
    ok = []
    for m in event["members"]:
        c = cands[m]
        if c["extent_m"] < WALL_EXTENT_M:
            ok.append((c["arc_m"], c))
        elif route_len - c["along_end_arc_m"] > AFTER_LAST_M:
            ok.append((c["along_end_arc_m"], c))
    if not ok:
        return None
    arc, c = max(ok, key=lambda t: t[0])
    return {"blob": c["blob"], "vis_xz": c["vis_xz"], "outline": c["outline"],
            "kind": "wall" if c["wall"] else "object"}, arc


def _just_after(event, cands, route_len):
    """At most AFTER_LAST_M of route after the event; when its last member is a wall, the goal must
    also be past the wall's end (a goal still beside the wall is not "just after" it). Never for a
    through event: it ends at its crossing, whatever its members' wall flags (final review I1)."""
    remaining = route_len - event["end_arc_m"]
    last = None if cands is None or event["type"] == "through" else \
        max((cands[m] for m in event["members"]), key=_end_arc)
    return remaining <= AFTER_LAST_M and (remaining > 1e-6 or last is None or not last["wall"])


def _distance(relation, metres, anchor):
    return {"type": "distance", "relation": relation, "amount": amount(metres),
            "distance_m": round(float(metres), 1), "anchor": anchor}


def _near_stop(frame, goal, head, right, blobs, radius):
    """stop_phrase rule 1 within `radius`: the nearest blob the goal is clearly in front of / just past (a blob
    mostly beside the goal is skipped; a wall-like blob only "in front of"), or None."""
    for b, cells, xz, d in blobs:
        if d.min() > radius:
            break
        v = xz[d.argmin()] - goal
        along, lat = float(v @ head), float(v @ right)
        rel = "in front of" if along > abs(lat) else "just past" if along < -abs(lat) else None
        if rel is None or (rel != "in front of" and _wall_like(xz)):
            continue
        x = _anchor(frame, b, goal, cells, xz, d)
        if x is not None:
            if _wall_like(xz):
                x["kind"] = "wall"
            return {"type": "near", "relation": rel, "anchor": x}
    return None


def _reanchored(stop, find_near):
    """A "well past" distance stop becomes the near stop `find_near()` finds within REANCHOR_M (d2 T2), marked
    reanchored; the near rule runs only for a "well past" stop (it computes visibility and outlines)."""
    if stop["relation"] == "past" and stop["amount"] == "well":
        near = find_near()
        if near is not None:
            return {**near, "reanchored": True}
    return stop


def stop_phrase(frame, route_xz, events, cands=None):
    """Where the route ends, first rule that fits (Addendum A 1):
    1. near: the nearest blob within STOP_NEAR_M the goal is clearly in front of / just past
       (a blob mostly beside the goal is skipped -- "next to" is gone -- and a wall-like blob,
       extent >= WALL_EXTENT_M, only takes "in front of"); the next-nearest is tried in turn;
    2. after_last: at most AFTER_LAST_M of route after the last event;
    3. distance past the last event's anchor, any remaining length, as a coarse amount; a wall-like
       member only counts once the route has left it (see _event_anchor); when the last event has
       no such anchor, the latest earlier event that has one;
    4. distance before / past the nearest blob on the line of approach (within STOP_FAR_M);
       a wall-like blob only "before" (it is ahead: in front of, at a distance).
    A "well past" result of rule 3 or 4 first tries rule 1 within REANCHOR_M (d2 T2: "in front of / just past Y"
    beats "well past X"), marked "reanchored".
    None when nothing fits (the route is dropped). Visibility and outlines (the expensive part)
    are only computed for the blobs a rule actually tries, nearest first."""
    rg = route_geometry(route_xz)
    goal = rg["pts"][-1]
    back = rg["pts"][max(0, len(rg["pts"]) - 11)]          # ~0.5 m before the goal
    head = goal - back
    head /= max(np.linalg.norm(head), 1e-9)
    right = np.array([-head[1], head[0]])
    blobs = sorted(_blobs_near(frame, goal, STOP_FAR_M), key=lambda t: t[3].min())
    near = _near_stop(frame, goal, head, right, blobs, STOP_NEAR_M)
    if near is not None:
        return near
    if events and _just_after(events[-1], cands, rg["arc"][-1]):
        return {"type": "after_last", "relation": "just after", "anchor": None}
    # how far past the last thing passed, when nothing visible is near the goal
    last = next((a for a in (_event_anchor(e, cands, rg["arc"][-1]) for e in reversed(events)) if a), None) \
        if cands is not None else None
    if last is not None:
        return _reanchored(_distance("past", rg["arc"][-1] - last[1], last[0]),
                           lambda: _near_stop(frame, goal, head, right, blobs, REANCHOR_M))
    for b, cells, xz, d in blobs:
        v = xz[d.argmin()] - goal
        if abs(v @ head) / max(np.linalg.norm(v), 1e-9) < APPROACH_COS:
            continue
        ahead = bool(v @ head > 0)
        if not ahead and _wall_like(xz):
            continue
        x = _anchor(frame, b, goal, cells, xz, d)
        if x is not None:
            return _reanchored(_distance("before" if ahead else "past", x["dist"], x),
                               lambda: _near_stop(frame, goal, head, right, blobs, REANCHOR_M))
    return None


# --- d3 visible stops (docs/superpowers/specs/2026-11-10-stop-phrases-d3-design.md) ----------------------------
def object_relation(frame, goal, head, right, blobs, low=frozenset()):
    """The d3 object stop: the nearest visible, non-wall blob within CLOSE_M of the goal and the first relation that
    fits, in the order behind -> in front of -> next to; a blob that fits none (diagonal to the goal) is skipped.
    A wall-sized blob in `low` (low furniture, Addendum A2) is an object here, not a wall.
    behind = the blob's image outline spans the goal's image column and a visible cell is >= BEHIND_GAP_M nearer the
    camera than the goal along the camera->goal line (final review I1: a metric distance to that line also caught
    objects beside the route's last metre); in front of = ahead of the goal along the final heading (as _near_stop);
    next to = beside it, the goal within the blob's visible extent along the heading. None when no blob qualifies."""
    cam = frame.cam.pos
    ray = goal - cam
    length = float(np.linalg.norm(ray))
    ray = ray / max(length, 1e-9)
    u_goal = float(frame.cam.uv(goal)[0][0, 0])
    for b, cells, xz, d in sorted(blobs, key=lambda t: t[3].min()):
        if d.min() > CLOSE_M:
            break
        if _wall_like(xz) and b not in low:
            continue
        x = _anchor(frame, b, goal, cells, xz, d)
        if x is None:
            continue
        vis = x["vis_xz"]
        dv = np.linalg.norm(vis - goal, axis=1)
        if dv.min() > CLOSE_M:
            continue
        t = (vis - cam) @ ray
        bbox = x["outline"].get("bbox")
        v = vis[dv.argmin()] - goal
        along, lat = float(v @ head), float(v @ right)
        ext = (vis - goal) @ head
        if bbox is not None and bbox[0] <= u_goal <= bbox[2] and np.any((t > 0) & (t <= length - BEHIND_GAP_M)):
            rel = "behind"
        elif along > abs(lat):
            rel = "in front of"
        elif ext.min() <= 0.0 <= ext.max():                 # |lateral| >= along here: beside the goal
            rel = "next to"
        else:
            continue
        return {"type": "object", "relation": rel, "anchor": {**x, "kind": "object"}}
    return None


def _place_grid(frame, low=frozenset()):
    """The frame's grid with every non-wall blob (and every blob in `low`, Addendum A2) cleared to FREE: furniture is
    not the wall of a d3 place."""
    s = frame.grid.s.copy()
    for b, (cells, xz) in _blob_cells(frame).items():
        if not _wall_like(xz) or b in low:
            s[cells[:, 0], cells[:, 1]] = FREE
    return Grid(s, frame.grid.mpp, frame.grid.x0, frame.grid.z0)


def wall_ahead(frame, goal, head, right, blobs, low=frozenset()):
    """The d3 backup (Addendum A1, option 2): the nearest visible wall-like blob within CLOSE_M ahead of the goal
    (along > |lateral|, as _near_stop) -> {"type": "wall", "relation": "in front of", "anchor"}, else None.
    A blob in `low` (low furniture, Addendum A2) is not a wall."""
    for b, cells, xz, d in sorted(blobs, key=lambda t: t[3].min()):
        if d.min() > CLOSE_M:
            break
        if not _wall_like(xz) or b in low:
            continue
        x = _anchor(frame, b, goal, cells, xz, d)
        if x is None:
            continue
        dv = np.linalg.norm(x["vis_xz"] - goal, axis=1)
        v = x["vis_xz"][dv.argmin()] - goal
        if dv.min() <= CLOSE_M and float(v @ head) > abs(float(v @ right)):
            return {"type": "wall", "relation": "in front of", "anchor": {**x, "kind": "wall"}}
    return None


def _stop_d3_on(frame, route_xz, goal, head, right, blobs, low=frozenset(), clear_low_cells=False):
    """stop_d3's order (object relation -> place -> wall ahead) with the blobs in `low` as furniture and, with
    clear_low_cells, the cells classed low within LOW_CLEAR_M of the goal cleared from the place grid (Addendum A2)."""
    rel = object_relation(frame, goal, head, right, blobs, low)
    if rel is not None:
        return rel
    grid = _place_grid(frame, low)
    if clear_low_cells:
        grid = _clear_low_cells(frame, grid, goal)
    p = place_at(grid, route_xz)
    if p is None:
        return wall_ahead(frame, goal, head, right, blobs, low)
    return {"type": "place", "relation": None, "place": p["place"], "measures": p["measures"], "anchor": None}


def stop_d3(frame, route_xz):
    """Where a visible route ends (d3): object_relation, else the place (stop_places.place_at on _place_grid), else a
    wall ahead (wall_ahead, Addendum A1 backup), else None (the route is dropped as "no_stop_place").
    Addendum A2: a "by the wall" stop is looked at again with low furniture as furniture (low_furniture, and the low
    cells near the goal cleared); that stop replaces it, and when there is none "by the wall" stays."""
    rg = route_geometry(route_xz)
    goal = rg["pts"][-1]
    back = rg["pts"][max(0, len(rg["pts"]) - 11)]          # ~0.5 m before the goal, as stop_phrase
    head = goal - back
    head /= max(np.linalg.norm(head), 1e-9)
    right = np.array([-head[1], head[0]])
    blobs = _blobs_near(frame, goal, CLOSE_M)
    stop = _stop_d3_on(frame, route_xz, goal, head, right, blobs)
    if stop is None or stop.get("place") != "wall":
        return stop
    again = _stop_d3_on(frame, route_xz, goal, head, right, blobs, low_furniture(frame, goal), clear_low_cells=True)
    return again if again is not None else stop


# --- d3 Addendum A2: furniture is not "the wall" (a height class per map cell from the depth image) ----------------
HEIGHT_LOW_M = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)   # a cell's own depth is read at these heights above the floor
HEIGHT_HIGH_M = (1.5, 1.7, 1.9)                    # ... and compared with what the image shows at these
OWN_NEARER_M, OWN_FARTHER_M = 1.0, 0.5    # a low reading is the cell's own within [fwd - 1.0, fwd + 0.5] m
SEEN_OVER_M = 0.5                         # a high reading this much farther than the cell's own depth: seen over it
SAME_SURFACE_M = 0.3                      # a high reading this close to it: the same upright surface (a wall)
HEIGHT_UNKNOWN, HEIGHT_LOW, HEIGHT_TALL = 0, 1, 2
LOW_NEAR_M = 2.0          # a wall-sized blob's visible cells this close to the goal decide whether it is low furniture
LOW_MIN_CELLS = 8         # ... at least this many of them with a height class
LOW_SHARE = 0.6           # ... and at least this share of those low
LOW_CLEAR_M = 3.5         # cells classed low this close to the goal are cleared from the place grid


def _image_depth(frame, xz, height):
    """(metres, fwd): planar metres the depth image shows at the pixel of each xz at `height` (inf where the image has
    no surface; NaN outside the image or within NEAR_CAMERA_M) and each point's forward distance."""
    uv, fwd = frame.cam.uv(xz, height)
    ok = (fwd > NEAR_CAMERA_M) & (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
    uv = np.where(ok[:, None], uv, 0.0)
    H, W = frame.depth.shape
    px = frame.depth[np.clip((uv[:, 1] * H).astype(int), 0, H - 1), np.clip((uv[:, 0] * W).astype(int), 0, W - 1)]
    metres = np.where(px > 0, frame.depth_fit[0] * px + frame.depth_fit[1], np.inf)
    return np.where(ok, metres, np.nan), fwd


def cell_heights(frame, xz):
    """HEIGHT_LOW / HEIGHT_TALL / HEIGHT_UNKNOWN for each map cell (world xz), from the depth profile in the cell's own
    image column (relative, so the depth fit's bias at range does not matter). The cell's own depth is the median of
    the readings at HEIGHT_LOW_M that fall within [fwd - OWN_NEARER_M, fwd + OWN_FARTHER_M] (legs seen through and
    nearer things are left out); at HEIGHT_HIGH_M a wall shows that depth again (within SAME_SURFACE_M) and low
    furniture shows something farther (by more than SEEN_OVER_M). With >= 2 usable high readings: low or tall when
    more than half say so, else unknown. All unknown without a depth image, a fit, or with a fit residual above
    DEPTH_FIT_MAX_RESID_M."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    out = np.full(len(xz), HEIGHT_UNKNOWN)
    fit = frame.depth_fit
    if frame.depth is None or fit is None or fit[2] > DEPTH_FIT_MAX_RESID_M or not len(xz):
        return out
    own = []
    for h in HEIGHT_LOW_M:
        d, fwd = _image_depth(frame, xz, h)
        own.append(np.where((d >= fwd - OWN_NEARER_M) & (d <= fwd + OWN_FARTHER_M), d, np.nan))
    own = np.array(own)
    has = np.isfinite(own).any(0)
    d_own = np.full(len(xz), np.nan)
    d_own[has] = np.nanmedian(own[:, has], axis=0)
    n, over, same = (np.zeros(len(xz), int) for _ in range(3))
    for h in HEIGHT_HIGH_M:
        d, _ = _image_depth(frame, xz, h)
        k = has & ~np.isnan(d)
        n += k
        over += k & (d > d_own + SEEN_OVER_M)
        same += k & (np.abs(d - d_own) <= SAME_SURFACE_M)
    out[(n >= 2) & (over * 2 > n)] = HEIGHT_LOW
    out[(n >= 2) & (same * 2 > n)] = HEIGHT_TALL
    return out


def low_furniture(frame, goal):
    """The wall-sized blobs that are low furniture: >= LOW_MIN_CELLS of their visible cells within LOW_NEAR_M of the
    goal have a height class (cell_heights) and >= LOW_SHARE of those are low."""
    out = set()
    for b, cells, xz, d in _blobs_near(frame, goal, LOW_NEAR_M):
        if not _wall_like(xz):
            continue
        c = cell_heights(frame, visible_cells(frame, cells[d <= LOW_NEAR_M], b))
        known = int((c != HEIGHT_UNKNOWN).sum())
        if known >= LOW_MIN_CELLS and (c == HEIGHT_LOW).sum() >= LOW_SHARE * known:
            out.add(b)
    return frozenset(out)


def _clear_low_cells(frame, grid, goal):
    """`grid` with its occupied cells within LOW_CLEAR_M of the goal that are classed low set to FREE."""
    rc = np.argwhere(grid.s == OCC)
    xz = frame.grid.xz(rc)
    rc = rc[np.linalg.norm(xz - goal, axis=1) <= LOW_CLEAR_M]
    low = rc[cell_heights(frame, frame.grid.xz(rc)) == HEIGHT_LOW]
    s = grid.s.copy()
    s[low[:, 0], low[:, 1]] = FREE
    return Grid(s, grid.mpp, grid.x0, grid.z0)


def stop_kind(stop):
    """"object:<relation>" / "wall:in front of" / "place:<place>" for a d3 visible stop, else None (d2 stops, behind
    and turn endings)."""
    if not stop:
        return None
    if stop.get("type") in ("object", "wall"):
        return f"{stop['type']}:{stop['relation']}"
    if stop.get("type") == "place":
        return f"place:{stop['place']}"
    return None


# --- turns and spur routes (d2, docs/superpowers/specs/2026-10-13-turn-events-d2-design.md) -----------------
TURN_MIN_DEG = 45.0          # a route that ENDS this far off its start->goal line has turned
HEADING_CHORD_M = 0.5        # heading at s = direction of the chord from s - 0.25 m to s + 0.25 m
TURN_END_WINDOW_M = 0.5      # the final heading = mean heading over the last this-many metres
TURN_PRE_M = 1.0             # the pre-turn heading is read this far before the turn point
TURN_HOLD_DEG = 15.0         # turn start / end: heading back within this of the pre-turn / final heading
TURN_ANCHOR_NEAR_M = 1.0     # a visible turn is anchored to a blob this close on its inner side
SPUR_REVERSAL_DEG = 150.0    # the chord heading turning this much within SPUR_CHECK_M = an out-and-back spur
SPUR_CHECK_M = 0.6           # (normal GT routes reach ~100 deg within 0.6 m; spurs 180)


def heading_profile(route_xz, step=ROUTE_STEP_M):
    """(pts, arc, theta_deg, line) of a route: its points densified at `step` (zero-length steps dropped),
    their arclength, the heading at every point in degrees relative to the start->goal line (+ = right,
    Camera.right handedness; unwrapped) taken from a HEADING_CHORD_M chord, and that line's unit vector.
    Relative to the line, not the camera: a yaw-augmented frame sees the same route turn the same way."""
    pts = densify(np.asarray(route_xz, float), step)
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    g = pts[-1] - pts[0]
    n = float(np.linalg.norm(g))
    g = g / n if n > 1e-9 else np.array([0.0, 1.0])
    right = np.array([-g[1], g[0]])

    def at(s):
        s = np.clip(s, 0.0, arc[-1])
        return np.stack([np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], -1)

    d = at(arc + HEADING_CHORD_M / 2) - at(arc - HEADING_CHORD_M / 2)
    return pts, arc, np.degrees(np.unwrap(np.arctan2(d @ right, d @ g))), g


def detect_turn(route_xz):
    """The route's final turn (spec detector A), or None. The route has turned when its mean heading over the
    last TURN_END_WINDOW_M is at least TURN_MIN_DEG off its start->goal line; a swerve round an obstacle ends
    back on that line. The turn point is the first bend that takes the route clearly onto the turn side: the first
    point, after the route last headed along or away from that side of the line, at least TURN_MIN_DEG / 2 onto it
    (G1 r1, 00641: a turn made in two bends belongs at the first one). A route that never heads along or away from
    the turn side falls back to where the heading last crosses half the final angle. The turn starts at the last
    earlier point within TURN_HOLD_DEG of the heading TURN_PRE_M before it, and ends at the first later point within
    TURN_HOLD_DEG of the final heading. Run it on a FULL route: a cut that stops mid-swerve would look like a turn
    (steering passes the full route's turn to its cuts)."""
    pts, arc, th, g = heading_profile(route_xz)
    if arc[-1] < HEADING_CHORD_M or np.linalg.norm(pts[-1] - pts[0]) < 1e-6:
        return None
    e = float(np.mean(th[arc >= arc[-1] - TURN_END_WINDOW_M]))
    if abs(e) < TURN_MIN_DEG:
        return None
    onto = np.sign(e) * th                                   # heading onto the turn side of the line, degrees
    away = np.nonzero(onto <= 0)[0]
    clear = np.nonzero((np.arange(len(onto)) > away[-1]) & (onto >= TURN_MIN_DEG / 2))[0] if len(away) else []
    if len(clear):
        k = int(clear[0])
    else:
        not_over = np.nonzero(np.sign(e) * (th - e / 2) <= 0)[0]
        k = int(min(not_over[-1] + 1, len(pts) - 1)) if len(not_over) else 0
    j = int(np.searchsorted(arc, arc[k] - TURN_PRE_M))
    pre = float(th[j])
    i0 = k
    while i0 > j and abs(th[i0] - pre) > TURN_HOLD_DEG:
        i0 -= 1
    i1 = k
    while i1 < len(th) - 1 and abs(th[i1] - e) > TURN_HOLD_DEG:
        i1 += 1
    r = np.radians(pre)
    h0 = np.cos(r) * g + np.sin(r) * np.array([-g[1], g[0]])
    return {"side": "right" if e > 0 else "left", "angle_deg": round(e, 1), "arc_m": float(arc[k]),
            "start_arc_m": float(arc[i0]), "end_arc_m": float(arc[i1]),
            "xz": [float(v) for v in pts[k]], "pre_heading_xz": [float(v) for v in h0]}


def has_spur(route_xz):
    """True when the route doubles back on itself: its chord heading turns by >= SPUR_REVERSAL_DEG within
    SPUR_CHECK_M of arclength. These are the generator's via-point routes that drive out and back along one
    line (87% of d1's "opposite" routes); d2 drops them (spec "Spur routes")."""
    _, arc, th, _ = heading_profile(route_xz)
    j = np.minimum(np.searchsorted(arc, arc + SPUR_CHECK_M), len(arc) - 1)
    return bool(np.any(np.abs(th[j] - th) >= SPUR_REVERSAL_DEG))


# --- endings (d2 Addendum T2, 2026-10-14): a route that ends out of sight "proceeds behind" what hides it -------
HIDDEN_BLIP_M = 0.3      # a visible run this short between two hidden runs is flicker along an occluder's edge
FPV_STEP_M = 0.05        # fpv_paths.json points are the route resampled this far apart, counted back from its end


def hidden_tail_m(flags, step=ROUTE_STEP_M):
    """Length of a route's final hidden stretch, counted back from its last point, or None when that point is in view.
    `flags` holds one value per point, `step` metres apart (True = hidden). A visible run shorter than HIDDEN_BLIP_M
    between two hidden runs counts as hidden; a visible run at the very start never does."""
    f = [bool(x) for x in flags]
    if not f or not f[-1]:
        return None
    k = len(f) - 1
    while True:
        while k > 0 and f[k - 1]:
            k -= 1                       # k: first point of the current hidden run
        if k == 0:
            break
        i = k - 1
        while i > 0 and not f[i - 1]:
            i -= 1                       # i: first point of the visible run before it
        if i == 0 or (k - i) * step >= HIDDEN_BLIP_M:
            break                        # a visible start, or a real visible stretch
        k = i                            # a blip: absorb it and keep walking back
    return (len(f) - 1 - k) * step


def fpv_hidden_flags(sample_dir, candidate_index):
    """The dataset's own hidden flag per fpv point of one candidate (fpv_paths.json runs), or None without the file."""
    p = Path(sample_dir) / "fpv_paths.json"
    if not p.exists():
        return None
    runs = json.loads(p.read_text())["candidates"][candidate_index]["runs"]
    return [bool(r["hidden"]) for r in runs for _ in r["uv"]]


BEHIND_READ_M = (0.15, 0.3, 0.5, 0.8)   # where past the disappearance the blob hiding the route is read


def behind_anchor(frame, route_xz, disappear_arc_m):
    """The nameable blob a route goes out of sight behind, or None. The first blob a camera ray hits on the way to the
    route BEHIND_READ_M past `disappear_arc_m` decides; if it is not nameable, or no read point is hidden on the map,
    the blob hiding the route's end is tried. Nameable as a turn anchor is (_anchor_outline near the ray's hit)."""
    rg = route_geometry(route_xz)
    pts, arc = rg["pts"], rg["arc"]
    for ds in BEHIND_READ_M:
        s = disappear_arc_m + ds
        if s > arc[-1] + 1e-9:
            break
        occ, hit = _first_hit(frame, pts[min(int(np.searchsorted(arc, s)), len(pts) - 1)])
        if occ is not None:
            got = _anchor_outline(frame, occ, hit)
            if got:
                return {"blob": occ, "vis_xz": got[0], "outline": got[1]}
            break
    occ, hit = _first_hit(frame, pts[-1])
    if occ is not None:
        got = _anchor_outline(frame, occ, hit)
        if got:
            return {"blob": occ, "vis_xz": got[0], "outline": got[1]}
    return None


def _anchor_outline(frame, blob, near_xz):
    """(visible cells, outline) of the part of a blob within KEEP_NEAR_ROUTE_M of near_xz, or None when it is
    not nameable (fewer than MIN_VISIBLE_CELLS visible cells, or an outline under MIN_AREA)."""
    cells, xz = _blob_cells(frame)[blob]
    keep = np.linalg.norm(xz - np.asarray(near_xz, float), axis=1) <= KEEP_NEAR_ROUTE_M
    vis = visible_cells(frame, cells[keep], blob)
    if len(vis) < MIN_VISIBLE_CELLS:
        return None
    o = object_outline(frame, vis)
    return (vis, o) if o["area"] >= MIN_AREA else None


def turn_anchor(frame, turn):
    """The object a turn is said relative to, or None (a bare "turn left"). A hidden turn point -> the
    nameable blob hiding it, relation "behind" (its outline: the part near where the camera ray hits it).
    Else the nearest nameable blob within TURN_ANCHOR_NEAR_M of the turn point on the turn's inner side
    (the rover's left for a left turn, in the pre-turn heading's frame) -> "round"."""
    p = np.asarray(turn["xz"], float)
    occ, hit = _first_hit(frame, p)
    if occ is not None:
        got = _anchor_outline(frame, occ, hit)
        if got:
            return {"blob": occ, "relation": "behind", "vis_xz": got[0], "outline": got[1]}
    h0 = np.asarray(turn["pre_heading_xz"], float)
    inner = np.array([-h0[1], h0[0]]) * (1.0 if turn["side"] == "right" else -1.0)
    near = []
    for b, (_, xz) in _blob_cells(frame).items():
        d = np.linalg.norm(xz - p, axis=1)
        k = int(d.argmin())
        if d[k] <= TURN_ANCHOR_NEAR_M and float((xz[k] - p) @ inner) > 0:
            near.append((float(d[k]), b))
    for _, b in sorted(near):
        got = _anchor_outline(frame, b, p)
        if got:
            return {"blob": b, "relation": "round", "vis_xz": got[0], "outline": got[1]}
    return None


def _straight_before(route, arc_m):
    """True when the route up to `arc_m` stays within STRAIGHT_MAX_M of its own chord ("go ahead, then turn")."""
    rg = route_geometry(route)
    pts = rg["pts"][rg["arc"] <= max(arc_m, ROUTE_STEP_M)]
    return len(pts) < 2 or route_geometry(pts)["offset"].max() <= STRAIGHT_MAX_M


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


def describe_route(sample_dir, candidate_index, stops="d2"):
    """The events + stop phrase for one candidate route of one frame (see module docstring); `stops` as describe_xz."""
    sd = Path(sample_dir)
    if not (sd / "state.npz").exists():
        return _none("no_state")
    meta, frame = load_frame(sd)
    cm = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cm)) or len(cm[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    route = cm[candidate_index]["polyline_xz"]
    flags = fpv_hidden_flags(sd, candidate_index)
    tail = hidden_tail_m(flags, FPV_STEP_M) if flags else None
    return describe_xz(meta, frame, route, turn=detect_turn(route), hidden_tail=tail, stops=stops)


def describe_xz(meta, frame, route, turn=None, hidden_tail=None, stops="d2"):
    """The events + stop phrase for any route polyline (world xz) on a loaded frame (steering pairs describe
    routes that are not candidates of the frame). `turn` is the FULL route's turn (detect_turn): a route
    shorter than its end_arc_m (a steering cut that stops before or inside the turn) gets no turn event, and
    without `turn` none is emitted. A route that doubles back on itself is dropped as "spur".
    `hidden_tail` (d2 T2) is the length of the route's final hidden stretch (hidden_tail_m), None when its end is in view: a hidden end ends with its turn (when the turn comes after the disappearance) or "behind" the blob it disappears behind (behind_anchor; none -> "no_behind_anchor"), and obstacle events after the disappearance are dropped.
    `stops` picks the rule for an end in view: "d2" stop_phrase (distance amounts), "d3" stop_d3 (an object relation
    or a place; none -> "no_stop_place"). Hidden ends are the same for both."""
    if stops not in ("d2", "d3"):
        raise ValueError(f"stops must be 'd2' or 'd3', got {stops!r}")
    route = np.asarray(route, float)
    if has_spur(route):
        return _none("spur")
    cands = candidates(frame, route)
    events = build_events(frame, cands)
    route_len = float(route_geometry(route)["arc"][-1])
    if turn is not None and route_len + 1e-6 < turn["end_arc_m"]:
        turn = None
    anchor = turn_anchor(frame, turn) if turn is not None else None
    if anchor is not None:          # a pass of the anchor alone, after the turn point, is what the turn says
        events = [e for e in events if not (e["type"] == "pass" and len(e["members"]) == 1
                                            and cands[e["members"][0]]["blob"] == anchor["blob"]
                                            and e["arc_m"] >= turn["arc_m"])]
    dis = None if hidden_tail is None else max(0.0, route_len - hidden_tail)
    if dis is not None:             # the text describes what is in view
        events = [e for e in events if e["arc_m"] <= dis + 1e-6]
    if not events:
        bent = (not _straight_before(route, dis)) if dis is not None else route_geometry(route)["offset"].max() > STRAIGHT_MAX_M
        if turn is None and bent:
            return _none("no_events_bent")                   # "go straight ahead" would be false
        if turn is not None and not _straight_before(route, turn["start_arc_m"]):
            return _none("no_events_bent")                   # an unexplained swing before the turn
    if dis is not None and turn is not None and turn["arc_m"] >= dis - 1e-6:
        stop = {"type": "turn", "relation": None, "anchor": None}          # the hidden turn is the ending
    elif dis is not None:
        x = behind_anchor(frame, route, dis)
        if x is None:
            return _none("no_behind_anchor", n_events=len(events) + (turn is not None))
        x["kind"] = "wall" if _wall_like(_blob_cells(frame)[x["blob"]][1]) else "object"
        stop = {"type": "behind", "relation": "behind", "anchor": x}
    elif stops == "d3":
        stop = stop_d3(frame, route)
        if stop is None:
            return _none("no_stop_place", n_events=len(events) + (turn is not None))
    else:
        stop = stop_phrase(frame, route, events, cands)      # obstacle events only: "just after" / "past" name objects
        if stop is None:
            return _none("no_stop_anchor", n_events=len(events) + (turn is not None))
    # Object rows in explicit passes. Events first: an event member's kind_hint is its event role
    # (pass member -> "object", wall member -> "wall", opening -> "opening"). Then the turn anchor, then the stop
    # anchor: each reuses an existing object row unchanged and sets kind_hint / outline only for a new object.
    objects = {}

    def add(key, kind, outline, cells):
        oid = f"{frame.name}:{key}"
        if oid not in objects:
            objects[oid] = {"object_id": oid, "kind_hint": kind,
                            **{k: outline[k] for k in ("polygon", "bbox", "crop_box", "area", "depth_fallback")},
                            "cells_xz": _cells(cells)}
        return oid

    rows = []
    for e in events:
        blobs = [cands[m]["blob"] for m in e["members"]]
        if e["type"] == "through":
            ids = [add(_gap_key(blobs, e["opening"]["gap_xz"]), "opening", e["opening"], e["opening"]["gap_xz"])]
        else:
            ids = [add(f"b{cands[m]['blob']}", "wall" if cands[m]["wall"] else "object",
                       cands[m]["outline"], cands[m]["vis_xz"]) for m in e["members"]]
        row = {"type": e["type"], "side": e["side"], "objects": ids, "arc_m": round(e["arc_m"], 2)}
        if e["type"] == "through":
            row["gap_xz"] = e["opening"]["gap_xz"]
        rows.append(row)
    if turn is not None:
        ids = []
        if anchor is not None:
            kind = "wall" if _wall_like(_blob_cells(frame)[anchor["blob"]][1]) else "object"
            ids = [add(f"b{anchor['blob']}", kind, anchor["outline"], anchor["vis_xz"])]
        p, h0 = np.asarray(turn["xz"], float), np.asarray(turn["pre_heading_xz"], float)
        lateral = float((route[-1] - p) @ np.array([-h0[1], h0[0]]))     # how far the end is off the pre-turn line
        rows.append({"type": "turn", "side": turn["side"], "relation": anchor["relation"] if anchor else None,
                     "objects": ids, "arc_m": round(turn["arc_m"], 2),
                     "turn_xz": [round(float(v), 3) for v in p], "pre_heading_xz": [round(float(v), 4) for v in h0],
                     "gt_lateral_m": round(abs(lateral), 2)})
        rows.sort(key=lambda r: r["arc_m"])                  # stable: an obstacle event at the same arc stays first
    st = {"type": stop["type"], "relation": stop["relation"], "amount": stop.get("amount"),
          "distance_m": stop.get("distance_m"), "object": None}
    if dis is not None:
        st["disappear_arc_m"] = round(dis, 2)
    if stop.get("reanchored"):
        st["reanchored"] = True
    if stop["type"] in ("object", "place", "wall"):      # d3: the place (None for an object stop) and its measures
        st["place"] = stop.get("place")
        if stop.get("measures"):
            st["measures"] = stop["measures"]
    if stop["anchor"] is not None:                       # after every event row (see above)
        a = stop["anchor"]
        st["object"] = add(a.get("key", f"b{a['blob']}"), a.get("kind", "object"), a["outline"], a["vis_xz"])
    return {"reason": None, "events": rows, "stop": st, "objects": objects,
            "goal_xz": [round(float(v), 3) for v in route[-1]],
            "pose_xz": [float(v) for v in meta["pose_xz"]], "heading_xz": [float(v) for v in meta["heading_xz"]],
            "route_len_m": round(route_len, 2)}


EVENTS_SCHEMA = "d2-endings"  # stamped into events_meta.json by prepare_habitat_instr.py events (d2 T2)
EVENTS_SCHEMA_D3 = "d3-stops"  # events --stops d3 (visible stops are object relations or places)
EVENTS_SCHEMAS = (EVENTS_SCHEMA, EVENTS_SCHEMA_D3)


def check_events_schema(events_path):
    """SystemExit unless the events_meta.json next to `events_path` carries one of EVENTS_SCHEMAS: events
    rows from before Addendum A have distance stops without amounts and no wall / between events."""
    mp = Path(events_path).parent / "events_meta.json"
    try:
        schema = json.loads(mp.read_text()).get("schema")
    except (OSError, ValueError):
        schema = None
    if schema not in EVENTS_SCHEMAS:
        raise SystemExit(f"events files are not one of {EVENTS_SCHEMAS!r} — rerun prepare_habitat_instr.py events "
                         f"({mp}: schema {schema!r})")


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
