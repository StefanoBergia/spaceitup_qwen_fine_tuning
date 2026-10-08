"""Steering eval pairs (stop slice, spec Addendum A2): one ground-truth route cut at two points.

The d1 eval pairs could not test steering (2026-10-07): both routes of a pair end at the frame's
goal and diverge behind obstacles. Side pairs (one object passed on either side) and destination
pairs (two objects to stop in front of) proved not viable on these frames (A1/A2: no room beside
objects, nothing nameable far enough from the entry). A stop pair keeps the frame's d1
ground-truth route (planned by the generator on the navmesh) and cuts it at two points whose stop
phrases differ: same image, same entry, same route, two texts, two different ends. Each cut is
described from the rover's pose, as d1's routes; its answer starts where the route reaches the
image (hidden stretches v = 0). A pair is kept only when each cut fails the other's description
under the eval's own checker. Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import itertools
from collections import Counter
from pathlib import Path

import numpy as np

from .descriptions import choose_names, desc_check, name_rows_split, route_objects, template_description
from .eval import description_follow, resample_polyline
from .habitat_data import clip_polyline_unit, resample_with_transitions
from .instructions import head_noun
from .landmarks import CAM_HEIGHT_M, MIN_FWD_M, densify
from .route_events import (MIN_AREA, MIN_VISIBLE_CELLS, ROUTE_STEP_M, WALL_EXTENT_M, _blob_cells, describe_xz,
                           first_blocker, load_frame, object_outline, visible_cells)

MIN_ROUTE_M = 1.5
MIN_SEP_UV = 0.1
CUT_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 1.0)   # where a route is cut, as shares of its length


def route_length(xz):
    xz = np.asarray(xz, float)
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum()) if len(xz) > 1 else 0.0


SEP_POINTS = 50     # resampling for the visible-separation test


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


def stop_key(stop):
    return stop["type"], stop["relation"], stop.get("amount"), stop.get("object")


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


def in_view(cam, route_xz):
    """The route from where it reaches the image's bottom edge (CAM_HEIGHT_M ahead of the camera):
    the generator's routes start at the rover, below the frame. Empty when it never gets there."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    _, fwd = cam.uv(pts)
    ok = fwd >= CAM_HEIGHT_M
    return pts[int(np.argmax(ok)):] if ok.any() else pts[:0]


def cuts(route_xz, fractions=CUT_FRACTIONS, min_m=MIN_ROUTE_M):
    """(fraction, prefix) of the route at each share of its length that keeps at least min_m."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    out = []
    for f in fractions:
        s = f * arc[-1]
        if s < min_m:
            continue
        out.append((f, pts[: max(2, int(np.searchsorted(arc, s - 1e-9)) + 1)]))
    return out


def stop_pairs(frame, meta, route_xz):
    """(pairs, reasons): every two cuts of one route whose stop phrases differ, whose answer ends lie
    at least MIN_SEP_UV apart, and whose answers each fail the other's description. Tags A (the
    earlier stop) / B."""
    reasons, routes = Counter(), []
    for f, cut in cuts(route_xz):
        desc = describe_xz(meta, frame, cut)
        if desc["reason"]:
            reasons[f"cut:{desc['reason']}"] += 1
            continue
        view = in_view(frame.cam, cut)
        path = answer_path(frame.cam, view, frame) if len(view) >= 2 else None
        if path is None or len(path) < 2:
            reasons["cut:out_of_view"] += 1
            continue
        routes.append({"tag": None, "side": None, "route_xz": _rounded(cut), "path": path, "desc": desc,
                       "anchor_event": None, "target": desc["stop"]["object"], "fraction": f})
    pairs = []
    for a, b in itertools.combinations(routes, 2):
        if stop_key(a["desc"]["stop"]) == stop_key(b["desc"]["stop"]):
            reasons["same_stop"] += 1
            continue
        sep = float(np.linalg.norm(np.subtract(a["path"][-1][:2], b["path"][-1][:2])))
        if sep < MIN_SEP_UV:
            reasons["too_close"] += 1
            continue
        ra, rb = {**a, "tag": "A"}, {**b, "tag": "B"}
        if not exclusive(ra, rb):
            reasons["not_exclusive"] += 1
            continue
        pairs.append({"slice": "stop", "anchor": None, "targets": [ra["target"], rb["target"]],
                      "separation": round(sep, 4), "routes": [ra, rb]})
    return pairs, reasons


def name_pair(pair, names, fallback, kinds):
    """Pick each route's names (A5 rule), refuse a stop object whose name is ambiguous in the frame,
    and write each route's template sentence. Returns the drop reason, or None after adding namer /
    names / template to both route dicts. Two stops at one object ("just past" / "well past the
    chair") are fine; two identical sentences are not."""
    for r in pair["routes"]:
        row = {"events": r["desc"]["events"], "stop": r["desc"]["stop"]}
        chosen, namer, reject = choose_names(row, names, fallback)
        if chosen is None:
            return "unnamed_object" if reject == "unnamed_object" else "name_clash"
        anchor = pair["anchor"] or r["target"]
        if anchor and frame_clash(anchor, chosen, names, fallback):
            return "name_clash"
        r.update(namer=namer, names={o: chosen[o] for o in route_objects(row)},
                 template=template_description(row["events"], row["stop"], chosen, kinds))
    if pair["routes"][0]["template"] == pair["routes"][1]["template"]:
        return "same_text"
    return None


def frame_pairs(sample_dir, candidate_index, names, fallback, kinds):
    """The stop pairs of one frame's ground-truth route, named, with drop reasons ("no_state",
    "bad_candidate", "stop:<reason>", "stop:kept")."""
    sd = Path(sample_dir)
    reasons = Counter()
    if not (sd / "state.npz").exists():
        reasons["no_state"] += 1
        return [], reasons
    meta, frame = load_frame(sd)
    cm = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cm)) or len(cm[candidate_index].get("polyline_xz") or []) < 2:
        reasons["bad_candidate"] += 1
        return [], reasons
    pre = f"{frame.name}:"      # naming only needs this frame's objects
    names = {o: n for o, n in names.items() if o.startswith(pre)}
    fallback = {o: n for o, n in (fallback or {}).items() if o.startswith(pre)}
    pairs, why = stop_pairs(frame, meta, cm[candidate_index]["polyline_xz"])
    reasons.update({f"stop:{k}": v for k, v in why.items()})
    kept = []
    for p in pairs:
        p["pair_id"] = f"steer_stop_{frame.name}_" + "_".join(f"{round(r['fraction'] * 100):03d}" for r in p["routes"])
        w = name_pair(p, names, fallback, kinds)
        reasons[f"stop:{w or 'kept'}"] += 1
        if w is None:
            kept.append(p)
    return kept, reasons


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
             "is_aug": r["habitat_meta"].get("source") == "augmented",
             "candidate_index": r["habitat_meta"]["candidate_index"]} for s, r in sorted(by.items())]


def object_rows(sample_dir, image, skip_ids=()):
    """One naming row per visible non-wall object of a frame (Addendum A1), as d1's object rows:
    blobs shorter than WALL_EXTENT_M, with >= MIN_VISIBLE_CELLS visible cells and an outline of
    at least MIN_AREA; ids in skip_ids (already named) are left out."""
    sd = Path(sample_dir)
    if not (sd / "state.npz").exists():
        return []
    _, frame = load_frame(sd)
    out = []
    for b, (cells, xz) in sorted(_blob_cells(frame).items()):
        oid = f"{frame.name}:b{b}"
        if oid in skip_ids or float(np.hypot(*np.ptp(xz, axis=0))) >= WALL_EXTENT_M:
            continue
        vis = visible_cells(frame, cells, int(b))
        if len(vis) < MIN_VISIBLE_CELLS:
            continue
        o = object_outline(frame, vis)
        if o["area"] < MIN_AREA:
            continue
        out.append({"object_id": oid, "kind_hint": "object", "image": image,
                    **{k: o[k] for k in ("polygon", "bbox", "crop_box", "area", "depth_fallback")}})
    return out
