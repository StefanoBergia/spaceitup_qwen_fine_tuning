"""Steering eval pairs: two routes from one entry whose texts differ only in what is tested.

The d1 eval pairs could not test steering (2026-10-07): both routes of every pair end at the
frame's goal and diverge only behind obstacles, so one route satisfied the other's events 94%
of the time. Here the builders take an injected planner (Addendum A1): in production the scene
navmesh baked at CLEAR_M = 0.30 m, planned by scripts/navmesh_plan.py and fed back through
lookup_planner; the frame-map A* (map_planner) is the unit tests' stand-in. Routes may cross
floor the camera never saw and leave the view: they are clipped like d1's, and stretches behind
an obstacle carry v = 0 in the answer path.

- side slice: same entry and stop, the route passes one named object O on opposite sides;
- destination slice: same entry, two "stop in front of" targets clearly apart in the image.

Every pair is kept only when each route fails the other's description under the eval's own
checker. Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import heapq
import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .descriptions import choose_names, desc_check, name_rows_split, route_objects, template_description
from .eval import description_follow, image_to_floor, resample_polyline
from .habitat_data import clip_polyline_unit, resample_with_transitions
from .instructions import head_noun
from .landmarks import FREE, MIN_FWD_M, OCC, Camera, densify
from .route_events import (MIN_VISIBLE_CELLS, ROUTE_STEP_M, WALL_EXTENT_M, _blob_cells, describe_xz,
                           first_blocker, load_frame)

CLEAR_M = 0.30      # Addendum A1: navmesh bake radius and via clearance (d1 GT routes keep 0.44-0.51 m)
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


def answer_path(cam, route_xz, frame=None):
    """[x, y, v] waypoints of a floor route, built as the d1 answers are: densified at
    ROUTE_STEP_M, projected, v = 0 where the frame map (when given) has an occupied cell between
    the camera and the point (route_events.first_blocker), clipped to the unit square, resampled
    with transitions. None when any point is less than MIN_FWD_M in front of the camera."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    uv, fwd = cam.uv(pts)
    if (fwd < MIN_FWD_M).any():
        return None
    hidden = [frame is not None and first_blocker(frame, p) is not None for p in pts]
    clipped = clip_polyline_unit([(float(u), float(v), h) for (u, v), h in zip(uv, hidden)])
    return [list(w) for w in resample_with_transitions(clipped)]


def map_planner(clear_m=CLEAR_M):
    """plan(frame, waypoints_xz) on the frame's own map (Task 1's A*): the unit tests' stand-in
    for the navmesh (synthetic scenes have none). Masks are cached per frame."""
    masks = {}

    def plan(frame, waypoints_xz):
        if frame.name not in masks:
            masks[frame.name] = traversable(frame.grid, frame.cam, clear_m)
        return plan_route(frame.grid, masks[frame.name], waypoints_xz)
    return plan


def request_key(frame_name, waypoints_xz):
    """Stable id of one planning request: the frame and its waypoints to the millimetre."""
    return frame_name + "|" + ";".join(f"{x:.3f},{z:.3f}" for x, z in np.asarray(waypoints_xz, float))


def recording_planner(requests):
    """plan() that plans nothing and records each request in `requests` (key -> waypoints):
    steer-propose runs the builders with it to learn what navmesh_plan.py must plan."""
    def plan(frame, waypoints_xz):
        requests[request_key(frame.name, waypoints_xz)] = _rounded(waypoints_xz)
        return None
    return plan


def lookup_planner(routes):
    """plan() answering from navmesh routes {key: route_xz or None} (navmesh_plan.py output): a None
    value is "planned, no route"; a key that was never requested raises KeyError."""
    def plan(frame, waypoints_xz):
        key = request_key(frame.name, waypoints_xz)
        if key not in routes:      # never requested: propose/steer drift, not a planning failure
            raise KeyError(key)
        r = routes[key]
        return None if r is None else np.asarray(r, float)
    return plan


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


def side_routes(frame, start, blob, plan):
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
    far = start + d * float(((xz - start) @ d).max())
    for k in STOP_BEYOND_M:
        stop = far + d * k
        routes = {s: plan(frame, [start, via[s], stop]) for s in ("left", "right")}
        if all(r is not None for r in routes.values()):
            return routes, None
    return None, "plan_failed"


def side_pair(frame, meta, start, blob, plan):
    """(pair dict, None) for the side slice around one blob, else (None, reason)."""
    routes, why = side_routes(frame, start, blob, plan)
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
        path = answer_path(frame.cam, routes[s], frame)
        if path is None:
            return None, "behind_camera"
        if len(path) < 2:
            return None, "too_short"
        out.append({"tag": tag, "side": s, "route_xz": _rounded(routes[s]), "path": path,
                    "desc": desc[s], "anchor_event": idx, "target": None})
    sep = separation(out[0]["path"], out[1]["path"])
    if sep < MIN_SEP_UV:
        return None, "too_close"
    return {"slice": "side", "anchor": oid, "targets": None, "separation": round(sep, 4), "routes": out}, None


def dest_route(frame, meta, start, blob, plan):
    """(route dict, None) from the entry to DEST_GAP_M in front of `blob`, whose stop phrase is
    "in front of" that blob; else (None, reason)."""
    xz = _blob_cells(frame)[blob][1]
    dd = np.linalg.norm(xz - start, axis=1)
    if dd.min() < MIN_AHEAD_M:
        return None, "too_near"
    q = xz[dd.argmin()]
    d = (q - start) / dd.min()
    route = plan(frame, [start, q - d * DEST_GAP_M])
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
    path = answer_path(frame.cam, route, frame)
    if path is None:
        return None, "behind_camera"
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


def load_names(primary_rows, fallback_rows):
    """(names, fallback, kinds) from the two name-row files of the mixed namer (A5): Gemma rows
    are primary, Cosmos rows the fallback; kinds = each object's "need" (wall / object / opening)."""
    names, _ = name_rows_split(primary_rows)
    fallback, _ = name_rows_split(fallback_rows)
    kinds = {r["id"]: r.get("need") for r in primary_rows}
    for r in fallback_rows:
        kinds.setdefault(r["id"], r.get("need"))
    return names, fallback, kinds


def frame_clash(oid, chosen, names=None, fallback=None):
    """True when another object of the same frame has a name with the same head noun as `oid`'s
    chosen name. Another object's names count from `chosen` and from both name files (either name
    of it counts), so a clash named only in the other file is still found."""
    frame, h = oid.split(":")[0], head_noun(chosen.get(oid))
    if not h:
        return False
    for src in (chosen, names or {}, fallback or {}):
        for o, n in src.items():
            if o != oid and o.split(":")[0] == frame and head_noun(n) == h:
                return True
    return False


def name_pair(pair, names, fallback, kinds):
    """Pick each route's names (A5 rule), refuse anchors whose name is ambiguous in the frame and
    destination targets sharing a head noun, and write each route's template sentence. Returns
    the drop reason, or None after adding namer / names / template to both route dicts."""
    heads = []
    for r in pair["routes"]:
        row = {"events": r["desc"]["events"], "stop": r["desc"]["stop"]}
        chosen, namer, reject = choose_names(row, names, fallback)
        if chosen is None:
            return "unnamed_object" if reject == "unnamed_object" else "name_clash"
        anchor = pair["anchor"] or r["target"]
        if frame_clash(anchor, chosen, names, fallback):
            return "name_clash"
        heads.append(head_noun(chosen[anchor]))
        r.update(namer=namer, names={o: chosen[o] for o in route_objects(row)},
                 template=template_description(row["events"], row["stop"], chosen, kinds))
    if pair["slice"] == "dest" and heads[0] == heads[1]:
        return "name_clash"
    return None


def anchor_blobs(frame, names, fallback, kinds):
    """Blobs that may anchor a pair: named (either file), not used as a wall, shorter than
    WALL_EXTENT_M."""
    out = []
    for b, (_, xz) in sorted(_blob_cells(frame).items()):
        oid = f"{frame.name}:b{b}"
        if not (names.get(oid) or (fallback or {}).get(oid)) or kinds.get(oid) == "wall":
            continue
        if float(np.hypot(*np.ptp(xz, axis=0))) >= WALL_EXTENT_M:
            continue
        out.append(int(b))
    return out


def frame_pairs(sample_dir, start_uv, names, fallback, kinds, plan):
    """Every candidate pair of one frame, both slices, named and mutually exclusive, with the
    drop reasons counted as "<slice>:<reason>" (frame-level reasons unprefixed). `plan` is a planner
    callable (map_planner in tests, lookup_planner in the steer stage)."""
    sd = Path(sample_dir)
    reasons = Counter()
    if not (sd / "state.npz").exists():
        reasons["no_state"] += 1
        return [], reasons
    meta, frame = load_frame(sd)
    pre = f"{frame.name}:"      # naming only needs this frame's objects
    names = {o: n for o, n in names.items() if o.startswith(pre)}
    fallback = {o: n for o, n in (fallback or {}).items() if o.startswith(pre)}
    start = entry_floor(meta, start_uv)
    if start is None:
        reasons["no_start"] += 1
        return [], reasons
    blobs = anchor_blobs(frame, names, fallback, kinds)
    if not blobs:
        reasons["no_named_object"] += 1
        return [], reasons
    pairs = []

    def keep(pair):
        why = name_pair(pair, names, fallback, kinds) or (None if exclusive(*pair["routes"]) else "not_exclusive")
        reasons[f"{pair['slice']}:{why or 'kept'}"] += 1
        if why is None:
            pairs.append(pair)

    for b in blobs:
        pair, why = side_pair(frame, meta, start, b, plan)
        if pair is None:
            reasons[f"side:{why}"] += 1
        else:
            pair["pair_id"] = f"steer_side_{frame.name}_b{b}"
            keep(pair)
    routes = []
    for b in blobs:
        r, why = dest_route(frame, meta, start, b, plan)
        if r is None:
            reasons[f"dest:{why}"] += 1
        else:
            routes.append(r)
    dpairs, dwhy = dest_pairs(routes)
    reasons.update({f"dest:{k}": v for k, v in dwhy.items()})
    for p in dpairs:
        p["pair_id"] = f"steer_dest_{frame.name}_" + "_".join(t.split(":")[-1] for t in p["targets"])
        keep(p)
    return pairs, reasons


def steer_frames(records):
    """One item per eval frame (sample_id) from the d1 entry records; the frame's GT-route
    record ("instructed") gives the entry point when present."""
    by = {}
    for r in records:
        hm = r["habitat_meta"]
        cur = by.get(hm["sample_id"])
        if cur is None or (hm.get("kind") == "instructed" and cur["habitat_meta"].get("kind") != "instructed"):
            by[hm["sample_id"]] = r
    return [{"sample_id": s, "sample_dir": str(Path(r["image"][0]).parent), "image": r["image"][0],
             "start_uv": list(r["habitat_meta"]["start_uv_norm"]), "scene_id": r["habitat_meta"]["scene_id"],
             "viewpoint": r["habitat_meta"].get("parent_sample_id") or s,
             "is_aug": r["habitat_meta"].get("source") == "augmented"} for s, r in sorted(by.items())]


def floor_height(sample_dir):
    """World Y of the frame's floor (state.npz floor_height), or None."""
    p = Path(sample_dir) / "state.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return float(z["floor_height"]) if "floor_height" in z.files else None
