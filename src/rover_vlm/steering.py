"""Steering eval pairs: two routes from one entry whose texts differ only in what is tested.

The d1 eval pairs could not test steering (2026-10-07): both routes of every pair end at the
frame's goal and diverge only behind obstacles, so one route satisfied the other's events 94%
of the time. Here routes are planned on the frame's own map (state.npz): known free floor the
camera sees, at least CLEAR_M from every obstacle.

- side slice: same entry and stop, the route passes one named object O on opposite sides;
- destination slice: same entry, two "stop in front of" targets clearly apart in the image.

Every pair is kept only when each route fails the other's description under the eval's own
checker. Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import heapq
import math

import numpy as np

from .descriptions import desc_check
from .eval import description_follow, image_to_floor, resample_polyline
from .habitat_data import clip_polyline_unit, resample_with_transitions
from .landmarks import FREE, OCC, Camera

CLEAR_M = 0.45      # d1 / round-4 GT routes keep 0.44-0.51 m from map obstacles (p10-p75)
SNAP_M = 0.3        # a waypoint may move this far onto the nearest traversable cell
_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_CHUNK = 1024


def traversable(grid, cam, clear_m=CLEAR_M):
    """Cells a route may use: known free, at least clear_m from every occupied cell (centre to
    centre), and in the image (the cell centre projects inside the frame). Unknown cells never
    are. The in-frame floor is convex (a 90 degree wedge beyond the bottom edge), so a straight
    move between two such cells stays in view."""
    rc = np.argwhere(grid.s == FREE)
    ok = np.zeros(grid.s.shape, bool)
    if not len(rc):
        return ok
    uv, fwd = cam.uv(grid.xz(rc))
    keep = Camera.in_frame(uv, fwd)
    occ = np.argwhere(grid.s == OCC).astype(np.int32)
    if len(occ):
        r2 = (clear_m / grid.mpp) ** 2
        rc32 = rc.astype(np.int32)
        for i in range(0, len(rc32), _CHUNK):
            d2 = ((rc32[i:i + _CHUNK, None, :] - occ[None]) ** 2).sum(-1).min(1)
            keep[i:i + _CHUNK] &= d2 >= r2
    ok[rc[keep, 0], rc[keep, 1]] = True
    return ok


def astar(mask, a, b):
    """8-connected shortest cell path a -> b over True cells of `mask`, or None."""
    if not (mask[a] and mask[b]):
        return None
    h_, w_ = mask.shape

    def h(p):
        return math.hypot(p[0] - b[0], p[1] - b[1])

    g = {a: 0.0}
    prev = {}
    heap = [(h(a), 0.0, a)]
    while heap:
        _, gc, p = heapq.heappop(heap)
        if p == b:
            out = [p]
            while p in prev:
                p = prev[p]
                out.append(p)
            return out[::-1]
        if gc > g[p]:
            continue
        for dr, dc in _NEIGH:
            q = (p[0] + dr, p[1] + dc)
            if not (0 <= q[0] < h_ and 0 <= q[1] < w_) or not mask[q]:
                continue
            ng = gc + (math.sqrt(2.0) if dr and dc else 1.0)
            if ng < g.get(q, math.inf):
                g[q] = ng
                prev[q] = p
                heapq.heappush(heap, (ng + h(q), ng, q))
    return None


def line_clear(mask, a, b):
    """True when every cell on the straight segment a -> b (half-cell steps) is in `mask`."""
    n = int(math.ceil(2 * math.hypot(b[0] - a[0], b[1] - a[1]))) + 1
    rr = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cc = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(mask[rr, cc].all())


def string_pull(mask, cells):
    """Corner cells of a cell path: from each kept cell, jump to the farthest later cell in
    straight line of sight on `mask`."""
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_clear(mask, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def snap(grid, mask, xz, max_m=SNAP_M):
    """The traversable cell nearest to world xz when within max_m, else None."""
    cells = np.argwhere(mask)
    if not len(cells):
        return None
    rc = grid.rc(xz)[0]
    d = np.hypot(cells[:, 0] - rc[0], cells[:, 1] - rc[1])
    k = int(d.argmin())
    return (int(cells[k, 0]), int(cells[k, 1])) if d[k] * grid.mpp <= max_m else None


def plan_route(grid, mask, waypoints_xz):
    """World xz corner polyline through the waypoints in order (each snapped onto `mask`):
    one A* leg per consecutive pair, each string-pulled on its own so every waypoint stays a
    corner (a pull across a via point could swap the side of the obstacle it rounds). None
    when a waypoint cannot snap or a leg has no path."""
    cells = [snap(grid, mask, p) for p in waypoints_xz]
    if any(c is None for c in cells):
        return None
    out = []
    for a, b in zip(cells[:-1], cells[1:]):
        leg = astar(mask, a, b)
        if leg is None:
            return None
        leg = string_pull(mask, leg)
        out += leg if not out else leg[1:]
    return grid.xz(np.array(out, float))


def route_length(xz):
    xz = np.asarray(xz, float)
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum()) if len(xz) > 1 else 0.0


SEP_POINTS = 50     # resampling for the visible-separation test


def entry_floor(meta, start_uv):
    """Floor xz of the frame's entry point (level camera), or None at/above the horizon."""
    xz = image_to_floor([[start_uv[0], start_uv[1], 1]], meta["pose_xz"], meta["heading_xz"])
    return xz[0] if len(xz) else None


def answer_path(cam, route_xz):
    """[x, y, 1] waypoints of a fully visible floor route, built as the d1 answers are (clip to
    the unit square, resample with transitions). Floor lines project to image lines, so the
    route's corners are enough."""
    uv, _ = cam.uv(np.asarray(route_xz, float))
    clipped = clip_polyline_unit([(float(u), float(v), False) for u, v in uv])
    return [list(w) for w in resample_with_transitions(clipped)]


def separation(path_a, path_b):
    """Largest image distance between the two paths, both resampled by arclength."""
    a = resample_polyline(np.array([p[:2] for p in path_a], float), SEP_POINTS)
    b = resample_polyline(np.array([p[:2] for p in path_b], float), SEP_POINTS)
    return float(np.linalg.norm(a - b, axis=1).max())


def exclusive(ra, rb):
    """Each route follows its own description and fails the other's (eval.description_follow,
    stop judged against each description's own route end)."""
    ca, cb = desc_check(ra["desc"], ra["desc"]["objects"]), desc_check(rb["desc"], rb["desc"]["objects"])
    ea, eb = ra["path"][-1], rb["path"][-1]
    return (description_follow(ra["path"], ca, ea)["follows"] == 1
            and description_follow(rb["path"], cb, eb)["follows"] == 1
            and description_follow(ra["path"], cb, eb)["follows"] == 0
            and description_follow(rb["path"], ca, ea)["follows"] == 0)
