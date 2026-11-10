"""Where a route ends when no object is close: a place read from the frame's occupancy grid (d3 stops).

The d3 stop (spec docs/superpowers/specs/2026-11-10-stop-phrases-d3-design.md) names an object within 1.5 m of
the end when there is one (route_events.object_relation); otherwise this module picks the place from a fixed list,
first fit wins: in a doorway, in a corner, at the end of the hallway, against a wall, in the middle of the room.

The grid (state.npz) is the rover's *explored* map: around a visible eval stop a median 39% of the 1.5 m disc is
unknown (2026-11-10). Unknown cells are no evidence either way: a ray that meets one stops with outcome "unknown",
never counted as free or as a wall, and "the middle of the room" needs most of its disc actually seen free.

Rays fan out from the goal in N_DIRS directions; angle 0 is the route's final heading and +90 its right
(right = (-h_z, h_x), as route_events.stop_phrase). A flat wall seen by several rays is one wall: walls are the
local minima of the hit distance over the fan. Pure numpy, CPU only; thresholds are starting values (plan Task 9
calibrates them on the d3 dry run).
"""

import numpy as np

from .landmarks import FREE, OCC, UNKNOWN

PLACES = ("doorway", "corner", "hallway_end", "wall", "room_middle")
N_DIRS = 16
RAY_MAX_M = 3.0
DOORWAY_HALF_M = 0.9          # each jamb this close to the goal (an opening at most 1.8 m wide)
DOORWAY_DEPTH_M = 0.6         # the jambs must end within this along the heading (a gap, not a corridor)
CORNER_M = 1.5
CORNER_DEG = (60.0, 120.0)    # the two walls' directions differ by this much
CORNER_OPEN_DEG = 30.0        # no wall within CORNER_M this close to either wall's opposite direction
CORNER_WEDGE = 1.5            # the ray between the two walls hits within CORNER_WEDGE * CORNER_M (closed wedge)
HALL_MAX_W = 2.2
HALL_BACK_M = (0.0, 1.0, 2.0)
HALL_END_M = 1.5
HALL_OPEN_W = 3.2             # wider than this ahead = the hallway opens into a room
WALL_M = 1.0
ROOM_CLEAR_M = 1.5
ROOM_MIN_FREE = 0.6
HEADING_M = 0.5               # the final heading is read over the route's last this-many metres


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _state(grid, p):
    """Cell state at world xz `p`; off the array is UNKNOWN (landmarks.Grid.cell)."""
    return grid.cell(grid.rc(p)[0])


def _march(grid, xz, d, max_m):
    """(outcome, dist) of one ray: the first OCC ("occ") or UNKNOWN ("unknown") cell, else ("free", max_m)."""
    step = grid.mpp / 2
    for t in np.arange(step, max_m + 1e-9, step):
        s = _state(grid, xz + t * d)
        if s == OCC:
            return "occ", float(t)
        if s == UNKNOWN:
            return "unknown", float(t)
    return "free", float(max_m)


def ray_hits(grid, xz, head_xz, n_dirs=N_DIRS, max_m=RAY_MAX_M):
    """[(angle_deg, outcome, dist_m)] for n_dirs rays from xz; angle 0 = head_xz, +90 = its right, in (-180, 180]."""
    xz, h = np.asarray(xz, float), _unit(head_xz)
    right = np.array([-h[1], h[0]])
    out = []
    for k in range(n_dirs):
        a = 360.0 * k / n_dirs
        a = a - 360.0 if a > 180.0 else a
        r = np.radians(a)
        outcome, dist = _march(grid, xz, np.cos(r) * h + np.sin(r) * right, max_m)
        out.append((a, outcome, dist))
    return out


def _walls(hits, max_m):
    """[(angle, dist)] of the local minima of the OCC hit distance within max_m (cyclic; non-hits count as far)."""
    d = [dist if o == "occ" and dist <= max_m else np.inf for _, o, dist in hits]
    n, out, i = len(d), [], 0
    while i < n:
        if np.isfinite(d[i]) and d[i] <= d[i - 1] and d[i] <= d[(i + 1) % n]:
            out.append((hits[i][0], d[i]))
            while d[(i + 1) % n] == d[i] and i + 1 < n:      # a plateau is one wall
                i += 1
        i += 1
    return out


def _gap(a, b):
    """Absolute angle between two directions in degrees, 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _side(grid, p, h, sign):
    """Distance to OCC straight to the right (sign +1) or left (-1) of heading h from p, or None."""
    right = np.array([-h[1], h[0]])
    o, d = _march(grid, np.asarray(p, float), sign * right, RAY_MAX_M)
    return d if o == "occ" else None


def width_across(grid, xz, head_xz):
    """Left + right distance to OCC across head_xz at xz, when both side rays hit OCC; else None."""
    h = _unit(head_xz)
    r, l = _side(grid, xz, h, 1), _side(grid, xz, h, -1)
    return None if r is None or l is None else r + l


def _point_back(route, back_m):
    """The route point back_m before its end (along the polyline), or None when the route is shorter."""
    pts = np.asarray(route, float)
    left = back_m
    for i in range(len(pts) - 1, 0, -1):
        seg = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if left <= seg + 1e-9:
            return pts[i] + (pts[i - 1] - pts[i]) * (left / max(seg, 1e-9))
        left -= seg
    return None


def _heading(route):
    pts = np.asarray(route, float)
    back = _point_back(pts, HEADING_M)
    return _unit(pts[-1] - (back if back is not None else pts[0]))


def _doorway(grid, goal, h, walls):
    sides = [(a, d) for a, d in walls if _gap(a, 90.0) <= 30.0 or _gap(a, -90.0) <= 30.0]
    for i, (a, da) in enumerate(sides):
        for b, db in sides[i + 1:]:
            if da <= DOORWAY_HALF_M and db <= DOORWAY_HALF_M and _gap(a, b) >= 150.0:
                for along in (DOORWAY_DEPTH_M, -DOORWAY_DEPTH_M):
                    p = goal + along * h
                    r, l = _side(grid, p, h, 1), _side(grid, p, h, -1)
                    if r is not None and l is not None and r <= DOORWAY_HALF_M and l <= DOORWAY_HALF_M:
                        return None                             # the jambs go on: a corridor, not a doorway
                return {"jamb_a_m": round(da, 2), "jamb_b_m": round(db, 2)}
    return None


def _corner(grid, goal, h, walls):
    near = [(a, d) for a, d in walls if d <= CORNER_M]
    for i, (a, da) in enumerate(near):
        for b, db in near[i + 1:]:
            if not CORNER_DEG[0] <= _gap(a, b) <= CORNER_DEG[1]:
                continue
            if any(_gap(c, a + 180.0) <= CORNER_OPEN_DEG or _gap(c, b + 180.0) <= CORNER_OPEN_DEG for c, _ in near):
                continue                                        # a wall opposite one of them: a corridor
            mid = np.radians(a + ((b - a + 180.0) % 360.0 - 180.0) / 2)
            right = np.array([-h[1], h[0]])
            o, dm = _march(grid, goal, np.cos(mid) * h + np.sin(mid) * right, CORNER_WEDGE * CORNER_M)
            if o != "occ":
                continue                                        # the wedge is open (a gap between two stretches)
            return {"wall_a_m": round(da, 2), "wall_b_m": round(db, 2), "angle_deg": round(_gap(a, b), 1),
                    "vertex_m": round(dm, 2)}
    return None


def _hallway_end(grid, route, goal, h):
    widths = []
    for back in HALL_BACK_M:
        p = _point_back(route, back)
        w = None if p is None else width_across(grid, p, h)
        if w is None or w > HALL_MAX_W:
            return None
        widths.append(w)
    o, ahead = _march(grid, goal, h, HALL_END_M)
    if o == "occ":
        return {"width_m": round(max(widths), 2), "ahead_m": round(ahead, 2)}
    w_ahead = width_across(grid, goal + HALL_END_M * h, h)
    if w_ahead is not None and w_ahead >= HALL_OPEN_W:
        return {"width_m": round(max(widths), 2), "width_ahead_m": round(w_ahead, 2)}
    return None


def _wall(walls):
    near = [d for _, d in walls if d <= WALL_M]
    return {"wall_m": round(near[0], 2)} if len(near) == 1 else None


def _room_middle(grid, goal):
    rc = np.round(grid.rc(goal)[0]).astype(int)
    rad = int(round(ROOM_CLEAR_M / grid.mpp))
    rr, cc = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    m = rr ** 2 + cc ** 2 <= rad ** 2
    rows, cols = rc[0] + rr[m], rc[1] + cc[m]
    inside = (rows >= 0) & (rows < grid.s.shape[0]) & (cols >= 0) & (cols < grid.s.shape[1])
    vals = np.full(len(rows), UNKNOWN)
    vals[inside] = grid.s[rows[inside], cols[inside]]
    free = float(np.mean(vals == FREE))
    if (vals == OCC).any() or free < ROOM_MIN_FREE:
        return None
    return {"free_share": round(free, 2), "clear_m": float(ROOM_CLEAR_M)}


def place_at(grid, route_xz):
    """{"place": one of PLACES, "measures": {name: float}} for the end of route_xz, or None when no place fits."""
    route = np.asarray(route_xz, float)
    goal, h = route[-1], _heading(route)
    walls = _walls(ray_hits(grid, goal, h), RAY_MAX_M)
    for place, found in (("doorway", lambda: _doorway(grid, goal, h, walls)),
                         ("corner", lambda: _corner(grid, goal, h, walls)),
                         ("hallway_end", lambda: _hallway_end(grid, route, goal, h)),
                         ("wall", lambda: _wall(walls)),
                         ("room_middle", lambda: _room_middle(grid, goal))):
        m = found()
        if m is not None:
            return {"place": place, "measures": {k: float(v) for k, v in m.items()}}
    return None
