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
import itertools
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from .descriptions import desc_check
from .eval import description_follow, image_to_floor, resample_polyline
from .habitat_data import clip_polyline_unit, resample_with_transitions
from .landmarks import FREE, OCC, Camera, densify
from .route_events import MIN_VISIBLE_CELLS, _blob_cells, describe_xz

CLEAR_M = 0.45      # d1 / round-4 GT routes keep 0.44-0.51 m from map obstacles (p10-p75)
SNAP_M = 0.3        # a waypoint may move this far onto the nearest traversable cell
MIN_ROUTE_M = 1.5
MIN_SEP_UV = 0.1
VIA_EXTRA_M = 0.1                 # via point: CLEAR_M + this beyond O's side
STOP_BEYOND_M = (0.5, 1.0, 1.5)   # shared stop: this far past O's far edge, first that plans
MIN_AHEAD_M = 1.0                 # an anchor nearer the entry leaves no room to go round it
INSIDE_FRAC = 0.9                 # share of O's cells that must lie between the two routes
MIN_DEST_DU = 0.15                # destination endpoints at least this far apart in image x
# A destination stop sits this far before its object: beyond route_events.PASS_M (0.7), so the
# target is not also a "pass" event of its own route, and inside STOP_NEAR_M (1.0), so the
# stop phrase is "in front of" that object.
DEST_GAP_M = 0.8
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


def _rounded(xz):
    return [[round(float(x), 3), round(float(z), 3)] for x, z in np.asarray(xz, float)]


def enclosed_blobs(frame, a_xz, b_xz):
    """{blob: cells inside the closed loop a + reversed(b)} (PIL even-odd polygon fill on the
    grid): two routes with the same ends differ in which way they go round exactly the blobs
    between them."""
    mpp = frame.grid.mpp
    pts = np.vstack([densify(np.asarray(a_xz, float), mpp / 2), densify(np.asarray(b_xz, float), mpp / 2)[::-1]])
    rc = frame.grid.rc(pts)
    img = Image.new("1", (frame.lab.shape[1], frame.lab.shape[0]), 0)
    ImageDraw.Draw(img).polygon([(float(c), float(r)) for r, c in rc], fill=1)
    inside = np.asarray(img, bool) & (frame.lab > 0)
    labs, counts = np.unique(frame.lab[inside], return_counts=True)
    return {int(b): int(n) for b, n in zip(labs, counts)}


def two_sided(frame, a_xz, b_xz, blob):
    """True when the two routes go round `blob` on opposite sides and round nothing else:
    at least INSIDE_FRAC of its cells lie between them, and no other blob with
    MIN_VISIBLE_CELLS or more cells does."""
    inside = enclosed_blobs(frame, a_xz, b_xz)
    n = len(_blob_cells(frame)[blob][0])
    return (inside.get(blob, 0) >= INSIDE_FRAC * n
            and not any(b != blob and k >= MIN_VISIBLE_CELLS for b, k in inside.items()))


def _event_key(e):
    return e["type"], e["side"], tuple(e["objects"])


def side_difference(ev_l, ev_r, oid):
    """(index of O's pass event, None) when the two event lists are identical except that the
    left route passes O alone on the left and the right route on the right; else
    (None, "anchor_missing" | "events_differ")."""
    def at(events, side):
        return [i for i, e in enumerate(events) if e["type"] == "pass" and e["objects"] == [oid] and e["side"] == side]

    il, ir = at(ev_l, "left"), at(ev_r, "right")
    if len(il) != 1 or len(ir) != 1:
        return None, "anchor_missing"
    if il != ir or len(ev_l) != len(ev_r):
        return None, "events_differ"
    if any(_event_key(a) != _event_key(b) for i, (a, b) in enumerate(zip(ev_l, ev_r)) if i != il[0]):
        return None, "events_differ"
    return il[0], None


def stop_key(stop):
    return stop["type"], stop["relation"], stop.get("amount"), stop.get("object")


def side_routes(frame, mask, start, blob):
    """({"left": xz, "right": xz}, None): one route each way round `blob` to a shared stop past
    it; else (None, reason). A via on the route's right of O means the route goes RIGHT of O
    (route_events' pass side = the route's side)."""
    xz = _blob_cells(frame)[blob][1]
    c = xz.mean(0)
    d = c - start
    dist = float(np.linalg.norm(d))
    if dist < MIN_AHEAD_M:
        return None, "too_near"
    d /= dist
    right = np.array([-d[1], d[0]])              # Camera.right's handedness
    lat = (xz - c) @ right
    via = {"right": c + right * (lat.max() + CLEAR_M + VIA_EXTRA_M),
           "left": c + right * (lat.min() - CLEAR_M - VIA_EXTRA_M)}
    if any(snap(frame.grid, mask, v) is None for v in via.values()):
        return None, "no_free_side"
    far = start + d * float(((xz - start) @ d).max())
    for k in STOP_BEYOND_M:
        stop = far + d * k
        routes = {s: plan_route(frame.grid, mask, [start, via[s], stop]) for s in ("left", "right")}
        if all(r is not None for r in routes.values()):
            return routes, None
    return None, "plan_failed"


def side_pair(frame, meta, mask, start, blob):
    """(pair dict, None) for the side slice around one blob, else (None, reason)."""
    routes, why = side_routes(frame, mask, start, blob)
    if routes is None:
        return None, why
    if min(route_length(r) for r in routes.values()) < MIN_ROUTE_M:
        return None, "too_short"
    if not two_sided(frame, routes["left"], routes["right"], blob):
        return None, "not_two_sided"
    desc = {s: describe_xz(meta, frame, r) for s, r in routes.items()}
    if any(d["reason"] for d in desc.values()):
        return None, "no_stop"
    oid = f"{frame.name}:b{blob}"
    idx, why = side_difference(desc["left"]["events"], desc["right"]["events"], oid)
    if why:
        return None, why
    if stop_key(desc["left"]["stop"]) != stop_key(desc["right"]["stop"]):
        return None, "events_differ"
    out = []
    for tag, s in (("L", "left"), ("R", "right")):
        path = answer_path(frame.cam, routes[s])
        if len(path) < 2:
            return None, "too_short"
        out.append({"tag": tag, "side": s, "route_xz": _rounded(routes[s]), "path": path,
                    "desc": desc[s], "anchor_event": idx, "target": None})
    sep = separation(out[0]["path"], out[1]["path"])
    if sep < MIN_SEP_UV:
        return None, "too_close"
    return {"slice": "side", "anchor": oid, "targets": None, "separation": round(sep, 4), "routes": out}, None


def dest_route(frame, meta, mask, start, blob):
    """(route dict, None) from the entry to DEST_GAP_M in front of `blob`, whose stop phrase is
    "in front of" that blob; else (None, reason)."""
    xz = _blob_cells(frame)[blob][1]
    dd = np.linalg.norm(xz - start, axis=1)
    if dd.min() < MIN_AHEAD_M:
        return None, "too_near"
    q = xz[dd.argmin()]
    d = (q - start) / dd.min()
    route = plan_route(frame.grid, mask, [start, q - d * DEST_GAP_M])
    if route is None:
        return None, "plan_failed"
    if route_length(route) < MIN_ROUTE_M:
        return None, "too_short"
    desc = describe_xz(meta, frame, route)
    oid = f"{frame.name}:b{blob}"
    stop = desc.get("stop") or {}
    if desc["reason"] or stop.get("type") != "near" or stop.get("relation") != "in front of" \
            or stop.get("object") != oid:
        return None, "no_stop"
    if any(oid in e["objects"] for e in desc["events"]):
        return None, "target_passed"      # another cell of the blob lies within PASS_M of the route
    path = answer_path(frame.cam, route)
    if len(path) < 2:
        return None, "too_short"
    return {"tag": None, "side": None, "route_xz": _rounded(route), "path": path, "desc": desc,
            "anchor_event": None, "target": oid}, None


def dest_pairs(routes):
    """Every two destination routes whose endpoints are MIN_DEST_DU apart in image x, tagged
    A (image-left end) / B; (pairs, reasons)."""
    pairs, reasons = [], Counter()
    for a, b in itertools.combinations(routes, 2):
        a, b = sorted((a, b), key=lambda r: r["path"][-1][0])
        if b["path"][-1][0] - a["path"][-1][0] < MIN_DEST_DU:
            reasons["too_close"] += 1
            continue
        ra, rb = {**a, "tag": "A"}, {**b, "tag": "B"}
        pairs.append({"slice": "dest", "anchor": None, "targets": [ra["target"], rb["target"]],
                      "separation": round(separation(ra["path"], rb["path"]), 4), "routes": [ra, rb]})
    return pairs, reasons
