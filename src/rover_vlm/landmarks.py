"""Grounded landmarks: which thing an instruction should name, found by geometry.

A v2 human audit (2026-09-28) found Cosmos names real objects well but picks the wrong
one ~20% of the time (75% on the second route of a pair): choosing which object a drawn
route goes around is a geometric judgment VLMs fail. Here geometry chooses the target on
the rover's own occupancy map (`state.npz`, built from this very frame) and projects it
into the image; Cosmos only names it (prompt i4, instructions.py).

Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

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


OBJECT_TOP_M = 1.0          # top of an object's outline (the 2D map has no heights)
MAX_PASS_DIST_M = 1.0       # an obstacle this close to the route is something it passes
START_SKIP_M = 0.3          # ignore what the route passes in its first metres (the rover's own spot)
KEEP_NEAR_ROUTE_M = 1.5     # outline only the part of a long blob (a wall) near the route
MIN_VISIBLE_CELLS = 4       # visible occupied cells for an object to count
MIN_POLY_AREA = 0.002       # outline area as a fraction of the image
CROSSING_SNAP_M = 0.8       # chord.crossing_xz to the nearest occupied cell
# Audit r3 (2026-09-28): 16 of 18 false instructions were routes straying more than this to
# the other side of their straight line; no single side word describes them.
MAX_CROSSING_M = 0.3
REASONS = ("bad_candidate", "no_state", "side_mismatch", "route_bends_both_ways",
           "no_visible_target", "target_too_small")


def _none(reason, **diag):
    return {"target": "none", "reason": reason, "polygon": [], "bbox": None,
            "world_xz": None, "diag": diag}


def _chord_frame(route):
    a, g = route[0], route[-1]
    ch = (g - a) / max(np.linalg.norm(g - a), 1e-9)
    return a, np.array([-ch[1], ch[0]])


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


def column_in_frame(cam, xz):
    """True where some part of a floor-to-OBJECT_TOP_M column at xz is in the image. An
    obstacle close to the camera has its floor below the frame but its body in view."""
    lo, fwd = cam.uv(xz)
    hi, _ = cam.uv(xz, OBJECT_TOP_M)
    return (fwd > MIN_FWD_M) & (lo[:, 0] >= 0) & (lo[:, 0] <= 1) & (hi[:, 1] <= 1) & (lo[:, 1] >= 0)


def _seen_first(grid, cam, lab, xz, rc):
    first = grid.first_occupied(cam.pos, xz)
    return first is None or lab[first] == lab[tuple(rc)]


def _dilate(mask):
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out[max(dr, 0):mask.shape[0] + min(dr, 0), max(dc, 0):mask.shape[1] + min(dc, 0)] |= \
                mask[max(-dr, 0):mask.shape[0] + min(-dr, 0), max(-dc, 0):mask.shape[1] + min(-dc, 0)]
    return out


def _outline(cam, vis, source, **diag):
    outline = np.vstack([cam.uv(vis)[0], cam.uv(vis, OBJECT_TOP_M)[0]])
    return _finish("object", convex_hull(outline), vis.mean(0), source=source,
                   visible_cells=len(vis), **diag)


def crossing_target(grid, cam, crossing_xz):
    """The obstacle the straight start->goal line runs into (Habitat's chord.crossing_xz):
    what the route is going around by construction. Blobs one empty cell apart are one
    obstacle (the camera sees the front face; the crossing cell may be on the back one).
    Returns the outline dict, or None when it is not on the map or not in view."""
    if crossing_xz is None:
        return None
    occm = grid.s == OCC
    occ = np.argwhere(occm)
    if not len(occ):
        return None
    d = np.linalg.norm(grid.xz(occ) - np.asarray(crossing_xz, float), axis=1)
    if d.min() > CROSSING_SNAP_M:
        return None
    merged, _ = components(_dilate(occm))
    lab = np.where(occm, merged, 0)
    cells = np.argwhere(lab == lab[tuple(occ[int(d.argmin())])])
    bxz = grid.xz(cells)
    keep = (np.linalg.norm(bxz - np.asarray(crossing_xz, float), axis=1) <= KEEP_NEAR_ROUTE_M) \
        & column_in_frame(cam, bxz)
    vis = np.array([q for q, rc, ok in zip(bxz, cells, keep) if ok and _seen_first(grid, cam, lab, q, rc)])
    if len(vis) < MIN_VISIBLE_CELLS:
        return None
    return _outline(cam, vis, "crossing", blob_cells=len(cells))


def select_target(sample_dir, candidate_index, side=None):
    """The object an instruction for this route should name, or a reason there is none.

    `side` (the record's label side): when given, a route that does not bow that way from
    its straight line is dropped (`side_mismatch`, ~1.4% of records) — its side word could
    not be true of any obstacle the route goes around. A route straying more than
    MAX_CROSSING_M to the other side is dropped too (`route_bends_both_ways`).

    First choice is the obstacle the straight line runs into (crossing_target): audit r2
    showed that is what people name. Otherwise, the visible obstacle the route passes closest to (within MAX_PASS_DIST_M,
    past the first START_SKIP_M) that lies on the route's inner side there — toward the
    straight start->goal line, i.e. what the route bends around. That is exactly "the route
    passes it on the side its label says", so the side word stays the metadata side and is
    true of the named object even on routes that bend both ways; an obstacle passed the other
    way round is never named. (Audit 2026-09-28: openings and dropping two-way routes failed;
    the user names the object the route goes around.)
    """
    sd = Path(sample_dir)
    meta = json.loads((sd / "meta.json").read_text())
    cands = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cands)) or len(cands[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    if not (sd / "state.npz").exists():
        return _none("no_state")
    route = np.asarray(cands[candidate_index]["polyline_xz"], float)
    grid = Grid.load(sd / "state.npz")
    cam = Camera(meta["pose_xz"], meta["heading_xz"])

    pts = densify(route, 0.05)
    a, nrm = _chord_frame(route)
    main = np.sign(((pts - a) @ nrm).mean()) or 1.0
    if side is not None and ("right" if main > 0 else "left") != side:
        return _none("side_mismatch")
    excursion = float(max(0.0, (-main * ((pts - a) @ nrm)).max()))
    if excursion > MAX_CROSSING_M:
        return _none("route_bends_both_ways", excursion_m=round(excursion, 3))
    t = crossing_target(grid, cam, meta.get("chord", {}).get("crossing_xz"))
    if t is not None:
        return t
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    normals = np.stack([-d[:, 1], d[:, 0]], 1)          # same handedness as the chord normal
    usable = np.linalg.norm(pts - pts[0], axis=1) > START_SKIP_M

    lab, _ = components(grid.s == OCC)
    occ = np.argwhere(lab > 0)
    xz = grid.xz(occ)
    best = None
    for j in np.nonzero(column_in_frame(cam, xz))[0]:
        dist = np.linalg.norm(pts - xz[j], axis=1)
        dist[~usable] = np.inf
        i = int(dist.argmin())
        if dist[i] > MAX_PASS_DIST_M or np.sign((xz[j] - pts[i]) @ normals[i]) != -main:
            continue
        if (best is None or dist[i] < best[0]) and _seen_first(grid, cam, lab, xz[j], occ[j]):
            best = (float(dist[i]), lab[tuple(occ[j])])
    if best is None:
        return _none("no_visible_target")

    cells = np.argwhere(lab == best[1])
    bxz = grid.xz(cells)
    near = np.linalg.norm(bxz[:, None] - pts[None], axis=2).min(1) <= KEEP_NEAR_ROUTE_M
    inview = column_in_frame(cam, bxz)
    vis = np.array([q for q, rc, ok in zip(bxz, cells, near & inview)
                    if ok and _seen_first(grid, cam, lab, q, rc)])
    if len(vis) < MIN_VISIBLE_CELLS:
        return _none("no_visible_target", visible_cells=len(vis))
    return _outline(cam, vis, "nearest", pass_dist_m=round(best[0], 3), blob_cells=len(cells))
