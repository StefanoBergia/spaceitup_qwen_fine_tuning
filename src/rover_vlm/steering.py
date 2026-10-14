"""Steering eval pairs (stop slice, spec Addendum A2): one ground-truth route cut at two points.

The d1 eval pairs could not test steering (2026-10-07): both routes of a pair end at the frame's
goal and diverge behind obstacles. Side pairs (one object passed on either side) and destination
pairs (two objects to stop in front of) proved not viable on these frames (A1/A2: no room beside
objects, nothing nameable far enough from the entry). A stop pair keeps the frame's d1
ground-truth route (planned by the generator on the navmesh) and cuts it at two points whose stop
phrases differ: same image, same entry, same route, two texts, two different ends. Each cut is
described from the rover's pose, as d1's routes; its answer starts where the route reaches the
image (hidden stretches v = 0). A pair is kept only when each cut fails the other's description
under the eval's own checker.
d2 (2026-10-13): a turning GT route also gets two turn cuts (just before
the turn, 1 m after it); that pair is the viewpoint's pair, slice "turn".
d2 T2 (2026-10-14): a cut that ends out of sight ends "behind" what hides it (or with its hidden turn), so a pair before and after a hidden turn is no longer exclusive. Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import copy
import itertools
from collections import Counter
from pathlib import Path

import numpy as np

from .descriptions import (choose_names, desc_check, name_rows_split, route_objects, template_description,
                           with_description)
from .eval import description_follow, resample_polyline
from .habitat_data import clip_polyline_unit, format_answer_goal, resample_with_transitions
from .instructions import head_noun
from .landmarks import CAM_HEIGHT_M, MIN_FWD_M, densify
from .route_events import (MIN_AREA, MIN_VISIBLE_CELLS, ROUTE_STEP_M, WALL_EXTENT_M, _blob_cells, describe_xz,
                           detect_turn, first_blocker, hidden_tail_m, load_frame, object_outline, visible_cells)

MIN_ROUTE_M = 1.5
MIN_SEP_UV = 0.1
CUT_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 1.0)   # where a route is cut, as shares of its length
TURN_CUT_BEFORE_M = 0.3    # A_turn stops this far before the turn starts ("stop before the turn")
TURN_CUT_AFTER_M = 1.0     # B_turn stops this far after it ends ("turn ... and stop past X")
GATE0_MIN_TURN_PAIRS = 50


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
    stop judged against each description's own route end). With MIN_SEP_UV (0.1) > STOP_TOL_UV (0.03)
    the cross-check (each route fails the other's text) is automatic for stop pairs, so this mostly
    re-checks that each cut follows its own text."""
    ca, cb = desc_check(ra["desc"], ra["desc"]["objects"]), desc_check(rb["desc"], rb["desc"]["objects"])
    ea, eb = ra["path"][-1], rb["path"][-1]
    return (description_follow(ra["path"], ca, ea)["follows"] == 1
            and description_follow(rb["path"], cb, eb)["follows"] == 1
            and description_follow(ra["path"], cb, eb)["follows"] == 0
            and description_follow(rb["path"], ca, ea)["follows"] == 0)


def _rounded(xz):
    return [[round(float(x), 3), round(float(z), 3)] for x, z in np.asarray(xz, float)]


def stop_key(stop):
    """A cut's ending key (d2 T2): two cuts with the same key and the same steps say the same thing."""
    if stop["type"] == "behind":
        return "behind", stop["object"]
    if stop["type"] == "turn":
        return ("turn",)
    return "stop", stop["relation"], stop.get("amount"), stop.get("object")


def hidden_flags(frame, route_xz):
    """One flag per point of the route densified at ROUTE_STEP_M (True = an occupied cell between it and the camera),
    the flags answer_path gives the answer's waypoints: a cut's ending is decided on the same flags (d2 T2)."""
    return [first_blocker(frame, p) is not None for p in densify(np.asarray(route_xz, float), ROUTE_STEP_M)]


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
    # keep the last point before the edge so clip_polyline_unit puts the entry exactly on the edge
    return pts[max(0, int(np.argmax(ok)) - 1):] if ok.any() else pts[:0]


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


def turn_cuts(route_xz, turn, min_m=MIN_ROUTE_M):
    """(fraction, prefix) of the two turn cuts of a turning route (spec "Steering"): A TURN_CUT_BEFORE_M before
    the turn starts, B TURN_CUT_AFTER_M after it ends (or at the route's end). Both or none: a cut shorter
    than min_m drops the pair."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    out = []
    for s in (turn["start_arc_m"] - TURN_CUT_BEFORE_M, min(turn["end_arc_m"] + TURN_CUT_AFTER_M, arc[-1])):
        if s < min_m:
            return []
        out.append((round(float(s / arc[-1]), 3), pts[: max(2, int(np.searchsorted(arc, s - 1e-9)) + 1)]))
    return out


def _event_keys(events):
    return [(e["type"], e["side"], tuple(e["objects"])) for e in events]


def stop_pairs(frame, meta, route_xz, turn=None):
    """(pairs, reasons): every two cuts of one route whose texts differ (stop or events), whose answer ends lie
    at least MIN_SEP_UV apart, and whose answers each fail the other's description. Tags A (the
    earlier stop) / B. prefix_events: A's events are the first ones of B's (a pure stop pair; False when
    the texts also differ in a step). Frame-level reasons: route_too_short (no cut keeps MIN_ROUTE_M),
    lt2_cuts (fewer than two cuts survive). With `turn` (route_events.detect_turn) the route also gets its two
    turn cuts; the pair of those two is slice "turn", every other pair "stop"."""
    reasons, routes = Counter(), []
    made = [(f, cut, False) for f, cut in cuts(route_xz)] + \
           [(f, cut, True) for f, cut in (turn_cuts(route_xz, turn) if turn else [])]
    if not made:
        reasons["route_too_short"] += 1
    for f, cut, is_turn in made:
        desc = describe_xz(meta, frame, cut, turn=turn, hidden_tail=hidden_tail_m(hidden_flags(frame, cut)))
        if desc["reason"]:
            reasons[f"cut:{desc['reason']}"] += 1
            continue
        view = in_view(frame.cam, cut)
        path = answer_path(frame.cam, view, frame) if len(view) >= 2 else None
        if path is None or len(path) < 2:
            reasons["cut:out_of_view"] += 1
            continue
        routes.append({"tag": None, "side": None, "route_xz": _rounded(cut), "path": path, "desc": desc,
                       "anchor_event": None, "target": desc["stop"]["object"], "fraction": f,
                       "turn_cut": is_turn})
    if made and len(routes) < 2:
        reasons["lt2_cuts"] += 1
    pairs = []
    routes.sort(key=lambda r: route_length(r["route_xz"]))      # A is always the shorter cut
    for a, b in itertools.combinations(routes, 2):
        if (stop_key(a["desc"]["stop"]), _event_keys(a["desc"]["events"])) == \
                (stop_key(b["desc"]["stop"]), _event_keys(b["desc"]["events"])):
            reasons["same_text"] += 1
            continue
        sep = float(np.linalg.norm(np.subtract(a["path"][-1][:2], b["path"][-1][:2])))
        if sep < MIN_SEP_UV:
            reasons["too_close"] += 1
            continue
        ra, rb = {**a, "tag": "A"}, {**b, "tag": "B"}
        if not exclusive(ra, rb):
            reasons["not_exclusive"] += 1
            continue
        ea, eb = (r["desc"]["events"] for r in (ra, rb))
        pairs.append({"slice": "turn" if a["turn_cut"] and b["turn_cut"] else "stop", "anchor": None, "targets": [ra["target"], rb["target"]],
                      "separation": round(sep, 4), "prefix_events": _event_keys(ea) == _event_keys(eb[: len(ea)]),
                      "routes": [ra, rb]})
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
    if pair["routes"][0]["namer"] != pair["routes"][1]["namer"]:
        return "mixed_namer"        # one object could be named differently in the two texts
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
    route = cm[candidate_index]["polyline_xz"]
    pairs, why = stop_pairs(frame, meta, route, turn=detect_turn(route))
    reasons.update({f"stop:{k}": v for k, v in why.items()})
    kept = []
    for p in pairs:
        p["pair_id"] = f"steer_{p['slice']}_{frame.name}_" + "_".join(f"{round(r['fraction'] * 100):03d}" for r in p["routes"])
        w = name_pair(p, names, fallback, kinds)
        reasons[f"stop:{w or 'kept'}"] += 1
        if w is None:
            kept.append(p)
    return kept, reasons


def steer_frames(records):
    """One item per eval frame (sample_id) from the d1 entry records; the frame's GT-route
    record ("instructed") supplies the candidate_index (the route) when present."""
    by = {}
    for r in records:
        hm = r["habitat_meta"]
        cur = by.get(hm["sample_id"])
        if cur is None or (hm.get("kind") == "instructed" and cur["habitat_meta"].get("kind") != "instructed"):
            by[hm["sample_id"]] = r
    return [{"sample_id": s, "sample_dir": str(Path(r["image"][0]).parent), "image": r["image"][0],
             "scene_id": r["habitat_meta"]["scene_id"],
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


GATE0_MIN_PAIRS = 150


def select_pairs(candidates):
    """One pair per viewpoint: a turn pair when the viewpoint has one (spec "Steering"), then the original frame
    before its yaw variants, then the largest end separation (pair_id breaks ties). Sorted by pair_id."""
    best = {}
    for p in candidates:
        rank = (p["slice"] != "turn", bool(p["is_aug"]), -p["separation"], p["pair_id"])
        if p["viewpoint"] not in best or rank < best[p["viewpoint"]][0]:
            best[p["viewpoint"]] = (rank, p)
    return sorted((v[1] for v in best.values()), key=lambda p: p["pair_id"])


def pair_rows(pair):
    """The two route rows of a selected pair, in the d1 events-row schema (so compose,
    desc_check and the audit take them) plus the steering fields."""
    ids = [f"{pair['pair_id']}_{r['tag']}" for r in pair["routes"]]
    rows = []
    for k, r in enumerate(pair["routes"]):
        d = r["desc"]
        rows.append({
            "id": ids[k], "pair_id": pair["pair_id"], "sibling_id": ids[1 - k], "slice": pair["slice"],
            "kind": "steer", "reason": None,
            **{f: pair[f] for f in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")},
            **{f: d[f] for f in ("events", "stop", "goal_xz", "pose_xz", "heading_xz", "route_len_m")},
            "route_xz": r["route_xz"], "path": r["path"], "desc_check": desc_check(d, d["objects"]),
            "anchor": pair["anchor"] or r["target"], "anchor_event": r["anchor_event"], "side": r["side"],
            "template": r["template"], "namer": r["namer"], "names": r["names"],
            "style_key": pair["pair_id"], "separation": pair["separation"], "prefix_events": pair["prefix_events"],
            "outlines": {oid: o["polygon"] for oid, o in d["objects"].items()},
        })
    return rows


STEER_FILES = [f"eval_steer_stop_{t}_{f}" for t in ("template", "composed") for f in ("entry", "goal")]
# d1 record fields that still hold for a steering route (everything route-specific is rebuilt)
STEER_META_KEEP = ("scene_id", "sample_id", "parent_sample_id", "source", "yaw_deg", "flavor", "framing", "entry_edge")


def steer_record(base, row, text, source, form):
    """One eval record for a steering route: the frame's d1 record with the route's answer, fresh
    habitat_meta and the description prompt in `form` ("entry" | "goal")."""
    rec = copy.deepcopy(base)
    rec["id"] = row["id"]
    rec["conversations"][1]["value"] = format_answer_goal([tuple(w) for w in row["path"]])
    hm = {k: base["habitat_meta"].get(k) for k in STEER_META_KEEP}
    hm.update({"pair_id": row["pair_id"], "sibling_id": row["sibling_id"], "kind": "steer",
               "start_uv_norm": row["path"][0][:2], "goal_uv_norm": row["path"][-1][:2],
               "goal_visible": int(row["path"][-1][2]),
               "steer": {"slice": row["slice"], "anchor": row["anchor"], "anchor_event": row["anchor_event"],
                         "side": row["side"], "viewpoint": row["viewpoint"],
                         "prefix_events": row.get("prefix_events")}})
    rec["habitat_meta"] = hm
    return with_description(rec, text, source, form, row["desc_check"])


def finalize_steer(base_by_sample, rows, desc_by_id):
    """{"files": {stem: records}, "stats"} for the 4 eval files. Template text for every pair;
    composed text only when both routes' compose rows passed as Cosmos text (a template fallback
    drops the pair from the composed files only)."""
    files = {k: [] for k in STEER_FILES}
    stats = Counter()
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for pid in sorted(by_pair):
        pair = sorted(by_pair[pid], key=lambda r: r["id"])
        if len(pair) != 2:
            stats["incomplete_pair"] += 1
            continue
        sl, base = pair[0]["slice"], base_by_sample[pair[0]["sample_id"]]
        stats[f"{sl}:pairs"] += 1
        for form in ("entry", "goal"):
            files[f"eval_steer_stop_template_{form}"] += [steer_record(base, r, r["template"], "template", form)
                                                          for r in pair]
        labels = [desc_by_id.get(r["id"]) for r in pair]
        if any(lab is None for lab in labels):
            stats[f"{sl}:unlabelled"] += 1
            continue
        if not all(lab.get("passed") and lab.get("instruction_source") == "cosmos" for lab in labels):
            stats[f"{sl}:composed_dropped"] += 1
            continue
        stats[f"{sl}:composed"] += 1
        for form in ("entry", "goal"):
            files[f"eval_steer_stop_composed_{form}"] += [steer_record(base, r, lab["description"], "cosmos", form)
                                                          for r, lab in zip(pair, labels)]
    return {"files": files, "stats": dict(stats)}
