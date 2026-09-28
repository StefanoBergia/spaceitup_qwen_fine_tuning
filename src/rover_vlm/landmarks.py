"""Grounded landmarks: which thing an instruction should name, found by geometry.

A v2 human audit (2026-09-28) found Cosmos names real objects well but picks the wrong
one ~20% of the time (75% on the second route of a pair): choosing which object a drawn
route goes around is a geometric judgment VLMs fail. Here geometry chooses the target on
the rover's own occupancy map (`state.npz`, built from this very frame) and projects it
into the image; Cosmos only names it (prompt i4, instructions.py).

Pure numpy + PIL, CPU only. Spec: docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

UNKNOWN, FREE, OCC = 0, 1, 2
CAM_HEIGHT_M = 0.8        # Habitat camera height above the floor
MIN_FWD_M = 0.3           # closer than this along the view axis = not usable in the image


class Camera:
    """Level pinhole, hfov 90°, at pose_xz facing heading_xz; normalised image coords.

    Verified 2026-09-28: projecting chord.goal reproduces fpv_paths' goal pixel
    (256.2, 282.0) vs (256.0, 282.0) on 00366-fxbzYAGkrtm_c012.
    """

    def __init__(self, pose_xz, heading_xz):
        self.pos = np.asarray(pose_xz, float)
        h = np.asarray(heading_xz, float)
        self.h = h / np.linalg.norm(h)
        self.right = np.array([-self.h[1], self.h[0]])

    def uv(self, xz, height=0.0):
        d = np.atleast_2d(np.asarray(xz, float)) - self.pos
        fwd, lat = d @ self.h, d @ self.right
        with np.errstate(divide="ignore", invalid="ignore"):
            u = 0.5 + 0.5 * lat / fwd
            v = 0.5 + 0.5 * (CAM_HEIGHT_M - height) / fwd
        return np.stack([u, v], 1), fwd

    @staticmethod
    def in_frame(uv, fwd):
        return (fwd > MIN_FWD_M) & (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)


class Grid:
    """The rover's occupancy map: row <- z, col <- x (same frame as roverbench's OccupancyMap)."""

    def __init__(self, state, mpp, x_min, z_min):
        self.s = np.asarray(state)
        self.mpp, self.x0, self.z0 = float(mpp), float(x_min), float(z_min)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["state"], z["mpp"], z["x_min"], z["z_min"])

    def rc(self, xz):
        xz = np.atleast_2d(np.asarray(xz, float))
        return np.stack([(xz[:, 1] - self.z0) / self.mpp - 0.5, (xz[:, 0] - self.x0) / self.mpp - 0.5], 1)

    def xz(self, rc):
        rc = np.atleast_2d(np.asarray(rc, float))
        return np.stack([self.x0 + (rc[:, 1] + 0.5) * self.mpp, self.z0 + (rc[:, 0] + 0.5) * self.mpp], 1)

    def cell(self, rc):
        r, c = int(round(rc[0])), int(round(rc[1]))
        if 0 <= r < self.s.shape[0] and 0 <= c < self.s.shape[1]:
            return int(self.s[r, c])
        return UNKNOWN

    def first_occupied(self, a_xz, b_xz):
        """First occupied cell on the segment a -> b (a's own cell excluded), or None."""
        for p in densify(np.array([a_xz, b_xz], float), self.mpp / 2)[1:]:
            rc = self.rc(p)[0]
            if self.cell(rc) == OCC:
                return int(round(rc[0])), int(round(rc[1]))
        return None


def densify(points, step):
    pts = np.asarray(points, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t / n for t in range(1, n + 1)]
    return np.array(out)


def components(mask):
    """8-connected labelling (no scipy in this env). Returns (labels, n)."""
    lab = np.zeros(mask.shape, int)
    n = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                            and mask[rr, cc] and not lab[rr, cc]):
                        lab[rr, cc] = n
                        q.append((rr, cc))
    return lab, n


def convex_hull(points):
    """Monotone-chain hull of 2D points, counter-clockwise, no repeated first point."""
    p = sorted({(round(float(x), 4), round(float(y), 4)) for x, y in points})
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], q) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, up = half(p), half(p[::-1])
    return lo[:-1] + up[:-1]


MIN_OPENING_M, MAX_OPENING_M = 0.6, 2.5   # a rover-passable gap, not a whole room
DOOR_HEIGHT_M = 2.0                       # top of an opening's outline
OBJECT_TOP_M = 1.0                        # top of an object's outline (the 2D map has no heights)
MIN_VISIBLE_CELLS = 4                     # visible occupied cells for an object to count
MAX_CROSSING_M = 0.3                      # route may stray this far to the other side of the chord
MIN_POLY_AREA = 0.002                     # outline area as a fraction of the image
REASONS = ("bad_candidate", "no_state", "route_crosses_chord", "no_visible_target",
           "both_sides", "target_too_small")


def _none(reason, **diag):
    return {"target": "none", "reason": reason, "polygon": [], "bbox": None,
            "world_xz": None, "diag": diag}


def _chord_frame(route):
    a, g = route[0], route[-1]
    ch = (g - a) / max(np.linalg.norm(g - a), 1e-9)
    return a, np.array([-ch[1], ch[0]])


def crossing_excursion(route):
    """How far (m) the route strays to the opposite side of the straight start->goal line
    from the side it mostly takes. Large = the route bends both ways (no single side word)."""
    pts = densify(route, 0.05)
    a, n = _chord_frame(route)
    off = (pts - a) @ n
    main = np.sign(off.mean()) or 1.0
    return float(max(0.0, (-main * off).max()))


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _finish(kind, poly, world_xz, **diag):
    poly = [[float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))] for u, v in poly]
    area = _poly_area(poly)
    if area < MIN_POLY_AREA:
        return _none("target_too_small", area=round(area, 5), **diag)
    us, vs = zip(*poly)
    return {"target": kind, "reason": None, "polygon": poly,
            "bbox": [min(us), min(vs), max(us), max(vs)],
            "world_xz": [float(world_xz[0]), float(world_xz[1])],
            "diag": {"area": round(area, 5), **diag}}


def crossed_blob(grid, crossing_xz, radius_m=0.5):
    """Mask of the occupied blob the straight chord runs into (its occupied cell nearest
    `crossing_xz`, within radius_m), or None when the map has no such cell."""
    if crossing_xz is None:
        return None
    occ = np.argwhere(grid.s == OCC)
    if not len(occ):
        return None
    d = np.linalg.norm(grid.xz(occ) - np.asarray(crossing_xz, float), axis=1)
    if d.min() > radius_m:
        return None
    lab, _ = components(grid.s == OCC)
    r, c = occ[int(d.argmin())]
    return lab == lab[r, c]


def find_opening(grid, route, wall=None):
    """Narrowest rover-passable gap between occupied cells that the route passes through.

    At each route point, scan perpendicular both ways for the first occupied cell; a point
    with a hit on both sides within MAX_OPENING_M is a gap. With `wall` (a cell mask), a gap
    counts only if one of its flanks is in it — the doorway is in the wall the chord crosses,
    not a squeeze between furniture elsewhere on the route. Returns the narrowest one with
    width in [MIN_OPENING_M, MAX_OPENING_M] as {"width_m", "centre", "flanks"} or None.
    """
    pts = densify(route, 0.05)
    best = None
    for i in range(1, len(pts) - 1):
        d = pts[i + 1] - pts[i - 1]
        d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([-d[1], d[0]])
        hits = []
        for sgn in (1.0, -1.0):
            hit = None
            for t in np.arange(grid.mpp, MAX_OPENING_M, grid.mpp / 2):
                q = pts[i] + sgn * n * t
                if grid.cell(grid.rc(q)[0]) == OCC:
                    hit = q
                    break
            hits.append(hit)
        if hits[0] is None or hits[1] is None:
            continue
        if wall is not None and not any(wall[tuple(np.round(grid.rc(h)[0]).astype(int))] for h in hits):
            continue
        w = float(np.linalg.norm(hits[0] - hits[1]))
        if MIN_OPENING_M <= w <= MAX_OPENING_M and (best is None or w < best["width_m"]):
            best = {"width_m": w, "centre": pts[i], "flanks": (hits[0], hits[1])}
    return best


def enclosed_blobs(grid, cam, route):
    """Occupied blobs inside the region between the route and its straight chord.

    Each: {"label", "cells", "visible_xz" (cells the camera sees first, in frame),
    "side" (mean signed offset from the chord; its sign is the side the blob is on)}.
    """
    H, W = grid.s.shape
    ring = grid.rc(np.vstack([route, route[:1]]))
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(float(c) + 0.5, float(r) + 0.5) for r, c in ring], fill=1)
    inside = (np.asarray(im) > 0) & (grid.s == OCC)
    lab, n = components(inside)
    a, nrm = _chord_frame(route)
    out = []
    for k in range(1, n + 1):
        rcs = np.argwhere(lab == k)
        xz = grid.xz(rcs)
        uv, fwd = cam.uv(xz)
        ok = cam.in_frame(uv, fwd)
        vis = np.zeros(len(xz), bool)
        for i in np.nonzero(ok)[0]:
            first = grid.first_occupied(cam.pos, xz[i])
            vis[i] = first is None or lab[first] == k
        out.append({"label": k, "cells": len(rcs), "visible_xz": xz[vis],
                    "side": float(((xz - a) @ nrm).mean())})
    return out


def select_target(sample_dir, candidate_index):
    """The thing an instruction for this route should name, or a reason there is none.

    Order: the opening the route passes through in the wall the chord crosses (doorway
    frames whose crossed wall is on the map, if the camera sees the opening),
    else the most visible occupied blob between the route and its chord, else none.
    """
    sd = Path(sample_dir)
    meta = json.loads((sd / "meta.json").read_text())
    cands = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cands)) or len(cands[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    if not (sd / "state.npz").exists():
        return _none("no_state")
    route = np.asarray(cands[candidate_index]["polyline_xz"], float)
    exc = crossing_excursion(route)
    if exc > MAX_CROSSING_M:
        return _none("route_crosses_chord", excursion_m=round(exc, 3))
    grid = Grid.load(sd / "state.npz")
    cam = Camera(meta["pose_xz"], meta["heading_xz"])

    wall = crossed_blob(grid, meta.get("chord", {}).get("crossing_xz")) if meta.get("flavor") == "doorway" else None
    if wall is not None:
        op = find_opening(grid, route, wall)
        if op is not None:
            c = op["centre"]
            uvc, fc = cam.uv([c])
            a, b = op["flanks"]
            _, fab = cam.uv([a, b])
            if (cam.in_frame(uvc, fc)[0] and (fab > MIN_FWD_M).all()
                    and grid.first_occupied(cam.pos, c) is None):
                quad = np.vstack([cam.uv([a])[0], cam.uv([b])[0],
                                  cam.uv([b], DOOR_HEIGHT_M)[0], cam.uv([a], DOOR_HEIGHT_M)[0]])
                return _finish("opening", quad, c, width_m=round(op["width_m"], 3))

    blobs = [b for b in enclosed_blobs(grid, cam, route) if len(b["visible_xz"]) >= MIN_VISIBLE_CELLS]
    if not blobs:
        return _none("no_visible_target")
    if len({np.sign(b["side"]) for b in blobs if abs(b["side"]) > 0.05}) > 1:
        return _none("both_sides", n_blobs=len(blobs))
    best = max(blobs, key=lambda b: len(b["visible_xz"]))
    xz = best["visible_xz"]
    pts = np.vstack([cam.uv(xz)[0], cam.uv(xz, OBJECT_TOP_M)[0]])
    return _finish("object", convex_hull(pts), xz.mean(0),
                   visible_cells=len(xz), blob_cells=best["cells"])
