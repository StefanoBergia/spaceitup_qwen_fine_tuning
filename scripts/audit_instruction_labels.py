"""Build a human audit page for v2 instruction labels, or for their grounded-landmark targets.

Draws N kept instructed records from train_general.json (fixed seed, stratified between
single-route "instructed" records and the "opposite" side of two-route pairs), renders
each frame with the labelled route in magenta (alternative in grey, as Cosmos saw it), and
writes a self-contained HTML page where each label is marked OK / wrong landmark / wrong
side / unsure. Marks are stored in the published artifact's db (collection "marks", doc
id = record id), so they can be read back with the Artifact tool's read_db.

--mode geometry (2026-09-28): the page shows each labelset record's route, its straight
entry->goal line and the target rover_vlm.landmarks chose (cyan outline), and asks whether
that target is right. 30 pair records + 70 single-route records from labelset_train.json.

--labels (i4 gate, 2026-09-28): labels mode reads label rows instead of train_general.json —
the passed rows' instructions on their labelset_eval records, target outline drawn.

--mode semantic: no page; for passed i4 rows in scenes with HM3D semantics
(outputs/landmark_semantics/, from scripts/render_semantic_frames.py) compare the name with
the HM3D category under the target; writes outputs/landmark_name_check.json and
outputs/landmark_name_mismatches.jsonl.

--mode events (d1 geometry gate, 2026-09-30): 100 routes from events_train.jsonl (30 from
pairs), each drawn with its route, the straight line, numbered event outlines (orange) and the
stop object (green), plus the template sentence with "object N" for names. Question: is the
description TRUE of this route, and does it name what a person would? Gate >= 90% true.

    uv run scripts/audit_instruction_labels.py --mode events \
        --events data/prepared_habitat_instr_v2/events_train.jsonl \
        --objects data/prepared_habitat_instr_v2/objects_train.jsonl     # -> outputs/instr_events_audit.html

--mode d1 (label gate): 100 passed descriptions from desc_*_d1.jsonl drawn with their numbered
objects. Gate >= 90% correct.
    uv run scripts/audit_instruction_labels.py --mode d1 \
        --events data/prepared_habitat_instr_v2/events_eval.jsonl \
        --objects data/prepared_habitat_instr_v2/objects_eval.jsonl \
        --desc outputs/instr_labels_v3/desc_gate_d1.jsonl

--mode steer (steering Gate 1, 2026-10-08): stop pairs from steer/events_steer_eval.jsonl; both cuts of the
route on the frame (magenta = first text, cyan = second), the stop objects outlined yellow, both template
texts. Is each text true of its own route (steps and stop), and are the two stops clearly apart?
Gate >= 90% of pairs true.
    uv run scripts/audit_instruction_labels.py --mode steer --n 50     # -> outputs/steer_audit.html

Run (CPU, login node is fine):
    uv run scripts/audit_instruction_labels.py            # -> outputs/instr_label_audit.html
    uv run scripts/audit_instruction_labels.py --n 150 --seed 1
    uv run scripts/audit_instruction_labels.py --mode geometry \
        --landmarks data/prepared_habitat_instr_v2/landmarks_train.jsonl   # -> outputs/instr_geometry_audit.html
    uv run scripts/audit_instruction_labels.py --labels outputs/instr_labels_v2/gate_i4.jsonl \
        --landmarks data/prepared_habitat_instr_v2/landmarks_eval.jsonl --out outputs/instr_i4_audit.html
    uv run scripts/audit_instruction_labels.py --mode semantic --labels outputs/instr_labels_v2/gate_i4.jsonl \
        --landmarks data/prepared_habitat_instr_v2/landmarks_eval.jsonl
"""

import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

import numpy as np

from rover_vlm.descriptions import template_description
from rover_vlm.habitat_choice import _draw_badge, _font
from rover_vlm.instructions import _answer_path, render_generation_jpeg, with_instruction
from rover_vlm.landmarks import Camera
from rover_vlm.landmark_check import OPENING_WORDS, beside_categories, name_matches, target_categories
from rover_vlm.overlay import draw_polyline
from rover_vlm.route_events import check_events_schema

REPO_ROOT = Path(__file__).resolve().parent.parent
V2 = REPO_ROOT / "data" / "prepared_habitat_instr_v2"
DATA = V2 / "train_general.json"
OUT = REPO_ROOT / "outputs" / "instr_label_audit.html"
TEMPLATE = Path(__file__).with_name("audit_instruction_labels.html")
V3 = REPO_ROOT / "data" / "prepared_habitat_instr_v3"
STEER_A, STEER_B, ANCHOR_COLOR = (255, 0, 200), (0, 200, 255), (255, 230, 0)
IMG_PX = 640
TARGET_OUTLINE = (0, 230, 255)
VERDICTS = {
    "steer": [
        {"v": "ok", "label": "True pair", "help": "Both texts are true of their own route, and the two stops are clearly apart"},
        {"v": "stop", "label": "Wrong stop", "help": "A stop phrase does not match where its route ends"},
        {"v": "event", "label": "Wrong step", "help": "A step (pass / between / through / keep on your side) is not true of its route"},
        {"v": "object", "label": "Wrong object", "help": "A name does not fit the thing it refers to"},
        {"v": "same", "label": "Not visibly different", "help": "The two stops are too close to tell apart in the image"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
    "labels": [
        {"v": "ok", "label": "Correct", "help": "Landmark is there and the route goes that side of it"},
        {"v": "landmark", "label": "Wrong landmark", "help": "Object not in the frame, misnamed, or not what the route bends around"},
        {"v": "side", "label": "Wrong side", "help": "Route goes the other way round the named object"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous to judge from this frame"},
    ],
    # round 3 (2026-09-28): the question is whether the instruction would be TRUE, not whether
    # the outline is the one object a person would pick — several are often defensible
    "geometry": [
        {"v": "ok", "label": "True instruction", "help": "\"Go <side> around the outlined object\" is correct for this route"},
        {"v": "wrong", "label": "False instruction", "help": "Wrong object (not something the route goes around) or wrong side"},
        {"v": "none_ok", "label": "No target, correctly", "help": "Nothing nameable; dropping this record is right"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
    "events": [
        {"v": "ok", "label": "True", "help": "Every step and the stop are true, and it names what you would"},
        {"v": "object", "label": "Wrong object", "help": "A step names something the route does not pass that way"},
        {"v": "missing", "label": "Misses an obstacle", "help": "An obvious obstacle the route goes around is left out"},
        {"v": "stop", "label": "Wrong stop", "help": "The stop phrase does not match where the route ends"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
    "d1": [
        {"v": "ok", "label": "Correct", "help": "Names, sides and the stop are all right, and it reads naturally"},
        {"v": "name", "label": "Wrong name", "help": "An object is misnamed or is not the numbered thing"},
        {"v": "side", "label": "Wrong side", "help": "A side word does not match how the route passes"},
        {"v": "stop", "label": "Wrong stop", "help": "The stop phrase is wrong"},
        {"v": "unnatural", "label": "Unnatural", "help": "True, but nobody would say it like that"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
}


VERDICTS["turns"] = [
    {"v": "ok", "label": "True", "help": "Every step (the turn included) and the stop are true; for a dropped route, dropping it is right"},
    {"v": "turn", "label": "Wrong turn", "help": "There is no such turn, or it goes the other way"},
    {"v": "anchor", "label": "Wrong turn object", "help": "'behind / round' the outlined object is not true"},
    {"v": "step", "label": "Other step or stop wrong", "help": "A pass / wall / between / through step or the stop is wrong"},
    {"v": "keep", "label": "Should not be dropped", "help": "A spur or bent drop of a route that is fine"},
    {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
]
TURN_MARK = (255, 60, 60)


def _turn(row):
    return next(e for e in row["events"] if e["type"] == "turn")


def sample_turns(rows_d2, rows_d1, seed, n_visible=20, n_hidden=30, n_bent=20, n_spur=30):
    """G1 (spec "Success criteria"): ~20 visible + ~30 hidden turn routes (hidden fills what visible lacks),
    20 routes d1 dropped as no_events_bent that d2 describes, 30 routes d2 drops as spur. Shuffled."""
    rng = random.Random(seed)
    d1 = {r["id"]: r for r in rows_d1}
    turns = [r for r in rows_d2 if r["reason"] is None and any(e["type"] == "turn" for e in r["events"])]
    hidden = [r for r in turns if _turn(r)["relation"] == "behind"]
    visible = [r for r in turns if _turn(r)["relation"] != "behind"]
    bent = [r for r in rows_d2 if r["reason"] is None and (d1.get(r["id"]) or {}).get("reason") == "no_events_bent"]
    spur = [r for r in rows_d2 if r["reason"] == "spur"]
    k_vis = min(n_visible, len(visible))
    k_hid = min(n_hidden + n_visible - k_vis, len(hidden))
    picked = ([("turn_visible", r) for r in rng.sample(visible, k_vis)]
              + [("turn_hidden", r) for r in rng.sample(hidden, k_hid)]
              + [("bent", r) for r in rng.sample(bent, min(n_bent, len(bent)))]
              + [("spur", r) for r in rng.sample(spur, min(n_spur, len(spur)))])
    rng.shuffle(picked)
    return picked, {"turn_visible": len(visible), "turn_hidden": len(hidden), "rescued_bent": len(bent), "spur": len(spur)}


def sample(records, n, opposite_share, seed):
    by_kind = {"instructed": [], "opposite": []}
    for r in records:
        kind = r["habitat_meta"].get("kind")
        if kind in by_kind and r["habitat_meta"].get("instruction"):
            by_kind[kind].append(r)
    rng = random.Random(seed)
    n_opp = round(n * opposite_share)
    picked = rng.sample(by_kind["opposite"], n_opp) + rng.sample(by_kind["instructed"], n - n_opp)
    rng.shuffle(picked)
    pop = {k: len(v) for k, v in by_kind.items()}
    return picked, pop


def draw_reference_line(img, path, dash=12):
    """Dashed straight line entry -> goal: the side label says which way the route bows from it."""
    (x0, y0, _), (x1, y1, _) = path[0], path[-1]
    W, H = img.size
    a, b = (x0 * W, y0 * H), (x1 * W, y1 * H)
    n = max(1, int(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 // dash))
    draw = ImageDraw.Draw(img)
    for i in range(0, n, 2):
        p = [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in (i, min(i + 1, n))]
        draw.line(p, fill=(0, 0, 0), width=5)
        draw.line(p, fill=(255, 255, 255), width=3)
    for (x, y) in (a, b):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 255, 255), outline=(0, 0, 0), width=2)


def sample_geometry(records, landmarks, n_pair, n_single, seed):
    """Labelset records that have a landmark row: n_pair from pairs (both kinds), n_single others."""
    have = [r for r in records if r["id"] in landmarks]
    pairs = [r for r in have if r["habitat_meta"].get("pair_id")]
    single = [r for r in have if not r["habitat_meta"].get("pair_id")]
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n_pair, len(pairs))) + rng.sample(single, min(max(0, n_single), len(single)))
    rng.shuffle(picked)
    return picked, {"pair": len(pairs), "single": len(single)}


def item(rec, landmark=None):
    m = rec["habitat_meta"]
    img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw_reference_line(img, _answer_path(rec))
    if landmark and landmark.get("polygon"):
        ImageDraw.Draw(img).polygon([(u * IMG_PX, v * IMG_PX) for u, v in landmark["polygon"]],
                                    outline=TARGET_OUTLINE, width=4)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    out = {"id": rec["id"], "instruction": m.get("instruction"), "side": m["side"], "kind": m["kind"],
           "has_alt": bool(m.get("alt_path")), "scene": m["scene_id"],
           "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
    if landmark:
        out.update(target=landmark["target"], reason=landmark["reason"])
    return out


def label_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sample_labelled(records, rows, n, seed):
    """A uniform random sample of the passed i4 rows, as records carrying their instruction.

    (Pairs used to come first so both sides showed; after the route_bends_both_ways drop
    almost no pair keeps both sides, and over-sampling them would skew a spot check.)
    """
    by_id = {r["id"]: r for r in records}
    ok = [r for r in rows if r.get("passed") and r["id"] in by_id]
    rng = random.Random(seed)
    picked = [with_instruction(by_id[r["id"]], r["instruction"], r.get("instruction_source") or "cosmos")
              for r in rng.sample(ok, min(n, len(ok)))]
    n_pair = sum(1 for r in ok if by_id[r["id"]]["habitat_meta"].get("pair_id"))
    return picked, {"passed_pair": n_pair, "passed_single": len(ok) - n_pair, "rows": len(rows)}


def semantic_report(rows, landmarks, sem_dir):
    """Agreement between i4 names and HM3D categories, by target type and pair/single."""
    table, mismatches, no_frame = {}, [], 0
    for r in rows:
        lm = landmarks.get(r["id"])
        if not (r.get("passed") and lm and r.get("reply")):
            continue
        f = sem_dir / f"{Path(lm['sample_dir']).name}.npz"
        if not f.exists():
            no_frame += 1
            continue
        z = np.load(f)
        ids, cats = z["ids"], str(z["categories"])
        if not json.loads(cats):     # scene replayed without HM3D labels (some val scenes)
            no_frame += 1
            continue
        name, beside = r["reply"]["name"], r["reply"].get("beside")
        key = f"{lm['target']}/{'pair' if r.get('pair_id') else 'single'}"
        if lm["target"] == "object":
            # any sizeable object in the outline counts: composite objects (a table ringed by
            # chairs, an island with a dishwasher) are rarely the majority-pixel category
            found = target_categories(ids, cats, lm["polygon"], name=name)
            ok, ref = any(name_matches(name, c) for c in found), sorted(found) or None
        else:
            near = sorted(beside_categories(ids, cats, lm["bbox"]))
            ok = any(w in OPENING_WORDS for w in name.split()) and (
                not beside or any(name_matches(beside, c) for c in near))
            ref = near
        cell = table.setdefault(key, {"n": 0, "agree": 0})
        cell["n"] += 1
        cell["agree"] += int(ok)
        if not ok:
            mismatches.append({"id": r["id"], "target": lm["target"], "name": name, "beside": beside,
                               "hm3d": ref, "instruction": r.get("instruction")})
    for cell in table.values():
        cell["rate"] = round(cell["agree"] / cell["n"], 3)
    return {"cells": table, "no_semantic_frame": no_frame}, mismatches


def load_landmarks_rows(paths):
    rows = {}
    for p in paths or []:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


EVENT_COLOR, STOP_COLOR = (255, 170, 0), (60, 220, 60)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sample_events(rows, n_pair, n_single, seed):
    ok = [r for r in rows if r["reason"] is None]
    pairs = [r for r in ok if r["pair_id"]]
    single = [r for r in ok if not r["pair_id"]]
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n_pair, len(pairs))) + rng.sample(single, min(max(0, n_single), len(single)))
    rng.shuffle(picked)
    return picked, {"described_pair": len(pairs), "described_single": len(single), "rows": len(rows)}


def placeholder_names(row):
    """{object_id: "object N"} in drawing order (events, then the stop object)."""
    ids = [o for e in row["events"] for o in e["objects"]] + ([row["stop"]["object"]] if row["stop"]["object"] else [])
    return {oid: f"object {i + 1}" for i, oid in enumerate(dict.fromkeys(ids))}


def row_kinds(row, objects):
    """{object_id: kind_hint} for the ids this row uses (the template marks walls with it)."""
    return {o: objects[o]["kind_hint"] for o in placeholder_names(row)}


def template_text(row, objects):
    """The template sentence with "object N" names and the row's kinds."""
    return template_description(row["events"], row["stop"], placeholder_names(row), kinds=row_kinds(row, objects))


def draw_numbered(img, objects, names, stop_id=None):
    """Outline each named object with its number; the stop object in green, events orange."""
    draw = ImageDraw.Draw(img)
    font = _font(max(16, IMG_PX // 30))
    for oid, label in names.items():
        colour = STOP_COLOR if oid == stop_id else EVENT_COLOR
        pts = [(u * IMG_PX, v * IMG_PX) for u, v in objects[oid]["polygon"]]
        if len(pts) >= 3:
            draw.polygon(pts, outline=colour, width=4)
            cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
            _draw_badge(draw, (cx, cy), label.split()[-1], colour, font, 14)


def events_item(rec, row, objects, text=None):
    names = placeholder_names(row)
    img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw_reference_line(img, _answer_path(rec))
    draw_numbered(img, objects, names, stop_id=row["stop"]["object"])
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    m = rec["habitat_meta"]
    return {"id": rec["id"], "instruction": text or template_text(row, objects),
            "side": m["side"], "kind": m["kind"], "has_alt": bool(m.get("alt_path")), "scene": m["scene_id"],
            "target": "object", "reason": None,
            "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}


def turns_item(rec, row, objects, group):
    """One G1 item: the GT route, numbered objects and the turn point; a spur drop shows the route alone."""
    if group == "spur":
        img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
        draw_reference_line(img, _answer_path(rec))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        m = rec["habitat_meta"]
        return {"id": rec["id"], "instruction": "DROPPED as an out-and-back spur. Is dropping this route right?",
                "side": m["side"], "kind": m["kind"], "has_alt": False, "scene": m["scene_id"], "target": "spur",
                "reason": "spur", "group": group,
                "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
    item = events_item(rec, row, objects)
    t = next((e for e in row["events"] if e["type"] == "turn"), None)
    if t is not None:
        img = Image.open(io.BytesIO(base64.b64decode(item["img"].split(",", 1)[1])))
        uv, fwd = Camera(row["pose_xz"], row["heading_xz"]).uv(np.array([t["turn_xz"]], float))
        if fwd[0] > 0:
            x, y = float(uv[0, 0]) * IMG_PX, float(uv[0, 1]) * IMG_PX
            ImageDraw.Draw(img).ellipse([x - 10, y - 10, x + 10, y + 10], outline=TURN_MARK, width=4)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        item["img"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    item["group"] = group
    return item


def select_d2_rows(ok, events_by_id, n, seed):
    """G3: n passed d2 rows composed with the current prompt (reused d1 rows left out), half with a turn."""
    rng = random.Random(seed)
    new = [r for r in ok if r.get("prompt_version") == "d4" and r["id"] in events_by_id]
    has = [r for r in new if any(e["type"] == "turn" for e in events_by_id[r["id"]]["events"])]
    has_ids = {r["id"] for r in has}
    no = [r for r in new if r["id"] not in has_ids]
    k = min(n // 2, len(has))
    rows = rng.sample(has, k) + rng.sample(no, min(n - k, len(no)))
    rng.shuffle(rows)
    return rows


def read_ids(path):
    """Route ids from a text file, one per line; blank lines and surrounding whitespace ignored."""
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def select_d1_rows(ok, ids, n, seed):
    """The passed d1 rows to audit: exactly `ids` in order (n ignored) if given, else a random n.
    Returns (rows, number of ids that are not among the passed rows)."""
    if ids is None:
        return random.Random(seed).sample(ok, min(n, len(ok))), 0
    by_id = {r["id"]: r for r in ok}
    rows = [by_id[i] for i in ids if i in by_id]
    return rows, len(ids) - len(rows)


def sample_steer(rows, n, seed, n_turn=0):
    """Up to n pairs (both rows, id order), shuffled; and the number of pairs available.
    n_turn of them come from the turn slice first, the rest from the other pairs."""
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], []).append(r)
    pairs = sorted((sorted(v, key=lambda r: r["id"]) for v in by.values() if len(v) == 2), key=lambda p: p[0]["pair_id"])
    rng = random.Random(seed)
    turn = [p for p in pairs if p[0].get("slice") == "turn"]
    other = [p for p in pairs if p[0].get("slice") != "turn"]
    first = rng.sample(turn, min(n_turn, len(turn)))
    picked = first + rng.sample(other, min(max(0, n - len(first)), len(other)))
    rng.shuffle(picked)
    return picked, len(pairs)


def steer_item(a, b):
    """One audit item: the frame with both routes and the stop objects outlined, both texts."""
    img = Image.open(a["image"]).convert("RGB").resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = _font(max(16, IMG_PX // 30))
    for k, oid in enumerate(o for o in dict.fromkeys([a["anchor"], b["anchor"]] + [o for e in b.get("events", []) if e["type"] == "turn" for o in e["objects"]]) if o):
        poly = a["outlines"].get(oid) or b["outlines"].get(oid) or []
        pts = [(u * IMG_PX, v * IMG_PX) for u, v in poly]
        if len(pts) >= 3:
            draw.polygon(pts, outline=ANCHOR_COLOR, width=4)
            _draw_badge(draw, (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)),
                        str(k + 1), ANCHOR_COLOR, font, 14)
    # A is a prefix of B: draw B first so A stays visible on top, and mark each route's end
    for row, colour in ((b, STEER_B), (a, STEER_A)):
        draw_polyline(draw, [(p[0], p[1]) for p in row["path"]], colour, IMG_PX, IMG_PX, width=6)
    for row, colour in ((b, STEER_B), (a, STEER_A)):
        x, y = row["path"][-1][0] * IMG_PX, row["path"][-1][1] * IMG_PX
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=colour, outline=(255, 255, 255), width=2)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return {"id": a["pair_id"], "instruction": f"MAGENTA: {a['template']}   \u00b7   CYAN: {b['template']}",
            "slice": a["slice"], "side": a["slice"], "kind": "steer", "has_alt": False, "scene": a["scene_id"],
            "target": "object", "reason": None,
            "pure": bool(a.get("prefix_events", True) and b.get("prefix_events", True)),
            "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}


def d1_title(desc):
    """Page title for --mode d1: names the desc file unless it is the default gate file."""
    stem = Path(desc).stem
    return "Route Description Audit" if Path(desc).name == "desc_gate_d1.jsonl" else f"Route Description Audit \u2014 {stem}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--opposite-share", type=float, default=0.2,
                    help="fraction drawn from the 'opposite' side of pairs (weighted back when scoring)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["labels", "geometry", "semantic", "events", "d1", "steer", "turns"], default="labels")
    ap.add_argument("--labels", type=Path, default=None,
                    help="i4 label rows (JSONL): labels mode shows them; semantic mode checks them")
    ap.add_argument("--split", type=Path, default=V2 / "labelset_eval.json",
                    help="labelset the --labels rows came from")
    ap.add_argument("--semantics", type=Path, default=REPO_ROOT / "outputs" / "landmark_semantics")
    ap.add_argument("--landmarks", type=Path, nargs="+", default=None,
                    help="landmarks_*.jsonl: draw each record's target (required for --mode geometry)")
    ap.add_argument("--events", type=Path, default=None, help="events_*.jsonl (--mode events)")
    ap.add_argument("--objects", type=Path, default=None, help="objects_*.jsonl (--mode events)")
    ap.add_argument("--desc", type=Path, default=None, help="desc_*_d1.jsonl (--mode d1)")
    ap.add_argument("--ids", type=Path, default=None,
                    help="route ids, one per line (--mode d1): audit exactly these passed routes, in order; ignores --n")
    ap.add_argument("--d1-events", type=Path, default=None, help="d1 events_*.jsonl (--mode turns: the rescued-bent group)")
    ap.add_argument("--labelset-dir", type=Path, default=V2, help="dir holding labelset_<split>.json")
    ap.add_argument("--round", default="d1", help="--mode d1: 'd2' samples half turn routes, d4 prompt only")
    ap.add_argument("--n-turn", type=int, default=0, help="--mode steer: pairs taken from the turn slice")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    landmarks = load_landmarks_rows(args.landmarks)
    if args.mode == "steer":
        events = args.events or V3 / "steer" / "events_steer_eval.jsonl"
        check_events_schema(events)
        picked, pop = sample_steer(jsonl(events), args.n, args.seed, n_turn=args.n_turn)
        items = [steer_item(a, b) for a, b in picked]
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "steer",
                   "verdicts": VERDICTS["steer"], "source": str(events)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Steering Pair Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "steer_audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"{len(items)} pairs (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
    if args.mode == "events":
        if not (args.events and args.objects):
            raise SystemExit("--mode events needs --events events_train.jsonl --objects objects_train.jsonl")
        check_events_schema(args.events)
        by_id = {r["id"]: r for r in json.loads((V2 / "labelset_train.json").read_text())}
        objects = {o["object_id"]: o for o in jsonl(args.objects)}
        rows, pop = sample_events(jsonl(args.events), 30, args.n - 30, args.seed)
        items = [events_item(by_id[r["id"]], r, objects) for r in rows]
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "events",
                   "verdicts": VERDICTS["events"], "source": str(args.events)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Route Events Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "instr_events_audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
    if args.mode == "turns":
        if not (args.events and args.objects and args.d1_events):
            raise SystemExit("--mode turns needs --events, --objects and --d1-events (the d1 events_train.jsonl)")
        check_events_schema(args.events)
        split = "eval" if "eval" in args.events.name else "train"
        by_id = {r["id"]: r for r in json.loads((args.labelset_dir / f"labelset_{split}.json").read_text())}
        objects = {o["object_id"]: o for o in jsonl(args.objects)}
        picked, pop = sample_turns(jsonl(args.events), jsonl(args.d1_events), args.seed)
        items = [turns_item(by_id[r["id"]], r, objects, g) for g, r in picked]
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "turns",
                   "verdicts": VERDICTS["turns"], "source": str(args.events)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Turn Events Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "turn_events_audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
    if args.mode == "d1":
        if not (args.events and args.objects and args.desc):
            raise SystemExit("--mode d1 needs --events, --objects and --desc desc_*_d1.jsonl")
        check_events_schema(args.events)
        split = args.labelset_dir / ("labelset_eval.json" if "eval" in args.events.name else "labelset_train.json")
        by_id = {r["id"]: r for r in json.loads(split.read_text())}
        ev = {r["id"]: r for r in jsonl(args.events)}
        objects = {o["object_id"]: o for o in jsonl(args.objects)}
        ok = [r for r in jsonl(args.desc) if r.get("passed")]
        if args.round == "d2":
            rows, missing = select_d2_rows(ok, ev, args.n, args.seed), 0
        else:
            rows, missing = select_d1_rows(ok, read_ids(args.ids) if args.ids else None, args.n, args.seed)
        if args.ids:
            print(f"--ids: {len(rows)} shown, {missing} id(s) missing or not passed in {args.desc}")
        items = [events_item(by_id[r["id"]], ev[r["id"]], objects, text=r["description"]) for r in rows]
        for it, r in zip(items, rows):
            it["source"] = r["instruction_source"]
        pop = {"passed": len(ok), "template": sum(r["instruction_source"] == "template" for r in ok)}
        title = "d2 Label Audit" if args.round == "d2" else d1_title(args.desc)
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "d1",
                   "verdicts": VERDICTS["d1"], "source": str(args.desc), "title": title}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", f"<title>{title}</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "instr_d1_audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    if args.mode == "semantic":
        if not (args.labels and landmarks):
            raise SystemExit("--mode semantic needs --labels <i4 rows> and --landmarks <landmarks jsonl>")
        report, mismatches = semantic_report(label_rows(args.labels), landmarks, args.semantics)
        out_dir = REPO_ROOT / "outputs"
        (out_dir / "landmark_name_check.json").write_text(json.dumps(report, indent=2))
        (out_dir / "landmark_name_mismatches.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mismatches))
        for key, cell in sorted(report["cells"].items()):
            print(f"{key:16s} n={cell['n']:4d}  agree {cell['rate']:.1%}")
        print(f"{report['no_semantic_frame']} passed rows have no semantic frame (scene without HM3D semantics)")
        print(f"{len(mismatches)} mismatches -> outputs/landmark_name_mismatches.jsonl")
        return
    if args.mode == "geometry":
        if not landmarks:
            raise SystemExit("--mode geometry needs --landmarks data/prepared_habitat_instr_v2/landmarks_train.jsonl")
        source = V2 / "labelset_train.json"
        picked, pop = sample_geometry(json.loads(source.read_text()), landmarks, 30, args.n - 30, args.seed)
        out = args.out or REPO_ROOT / "outputs" / "instr_geometry_audit.html"
    elif args.labels:
        source = args.split.resolve()
        picked, pop = sample_labelled(json.loads(source.read_text()), label_rows(args.labels), args.n, args.seed)
        out = args.out or REPO_ROOT / "outputs" / "instr_i4_audit.html"
    else:
        source = DATA
        picked, pop = sample(json.loads(source.read_text()), args.n, args.opposite_share, args.seed)
        out = args.out or OUT
    items = [item(r, landmarks.get(r["id"])) for r in picked]
    payload = {"items": items, "population": pop, "seed": args.seed, "mode": args.mode,
               "verdicts": VERDICTS[args.mode], "source": str(source.relative_to(REPO_ROOT))}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
    if args.mode == "geometry":   # the gallery names an artifact by its <title>
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Landmark Geometry Audit</title>", 1)
    elif args.labels:             # i4 labels: Cosmos's sentence on the geometry-chosen target
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Grounded Label Audit</title>", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
