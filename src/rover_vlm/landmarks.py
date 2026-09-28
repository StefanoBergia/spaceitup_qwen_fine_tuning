"""Grounded landmarks: which thing an instruction should name, found by geometry.

A v2 human audit (2026-09-28) found Cosmos names real objects well but picks the wrong
one ~20% of the time (75% on the second route of a pair): choosing which object a drawn
route goes around is a geometric judgment VLMs fail. Here geometry chooses the target on
the rover's own occupancy map (`state.npz`, built from this very frame) and projects it
into the image; Cosmos only names it (prompt i4, instructions.py).

Pure numpy + PIL, CPU only. Spec: docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md
"""

from collections import deque

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
