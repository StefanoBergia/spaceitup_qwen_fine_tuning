"""Route descriptions (prompt d1): turning geometry's events into checked sentences.

rover_vlm.route_events finds what a route passes and where it stops. This module holds the
canonical template sentence, the text rules every description must pass (all names present
in route order, each pass with its own side word, the stop phrase present), the Cosmos
prompts for the two labelling stages (name one object on a crop; compose the sentence from
text only) and the finalize helpers that turn labels into training/eval records.

Pure and CPU-only. Spec: docs/superpowers/specs/2026-09-30-route-descriptions-design.md
"""

import copy
import io
import random
import re
from collections import Counter

from PIL import Image, ImageDraw

from .habitat_data import description_entry_prompt, instruction_endpoints_prompt
from .instructions import (
    _DRAWING, _EGOCENTRIC, STYLES, TARGET_OUTLINE_COLOR, _answer_block, _answer_path, _stable_index, choose_style,
    head_noun, normalize_name,
)
from .traces import SCENE_INTRO, answer_payload, parse_trace

DESC_WORDS = (5, 45)
_STOP = re.compile(r"\b(?:stop|stops|stopping|halt|park|come to rest|end up)\b", re.I)
_AFTER = re.compile(r"\b(?:just|right) (?:after|past|beyond)\b", re.I)
_SIDE = re.compile(r"\b(left|right)\b", re.I)
_BREAK = re.compile(r"[,;.]|\bthen\b|\band\b", re.I)
GENERIC_NAMES = {"object", "thing", "item", "area", "region", "stuff", "something", "space", "structure",
                 "furniture", "obstacle", "piece", "unit"}   # head_noun("piece of furniture") == "piece"
_CONJUNCTION = re.compile(r"\b(?:and|then|or)\b", re.I)     # would split the name at _BREAK
# "right" meaning "exactly", not a side: blanked out before any side-word check. "next to",
# "by" and "beside" are no longer stop relations (Addendum A) but stay here: a description may
# still say "right next to" in passing, and that "right" is never a side word.
_RIGHT_ADVERB = re.compile(r"\bright(?= (?:after|past|beyond|before|in front of|next to|by|beside)\b)", re.I)
_PRONOUN = re.compile(r"\b(?:it|them)\b", re.I)
# The words that say each stop relation. The stop clause (the text after the last stop word)
# must hold one of its own relation's words, so a swapped relation ("a little before the
# armchair" for a past stop) is rejected as stop:relation. after_last is checked by _AFTER.
STOP_RELATION_WORDS = {
    ("near", "in front of"): ("in front of", "right in front of", "just in front of"),
    ("near", "just past"): ("just past", "right past", "just beyond", "just after"),
    ("distance", "before"): ("before", "short of"),
    ("distance", "past"): ("past", "after", "beyond"),
}
# The words that say each coarse distance-stop amount (route_events.AMOUNT_STEPS); the stop
# clause must hold one of its own amount's words (Addendum A 1).
STOP_AMOUNT_WORDS = {
    "just": ("just", "right"),
    "a little": ("a little", "a bit", "slightly", "shortly"),
    "well": ("well", "far", "a good way"),
}


def _words_re(words):
    return re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.I)


def event_phrase(ev, names):
    """One step. Object passes name the ROUTE's side ("go right of the table"); walls name the
    ROVER's side the wall is on ("keep the white wall on your left")."""
    if ev["type"] == "pass":
        return f"go {ev['side']} of the " + " and the ".join(names)
    if ev["type"] == "wall":
        return f"keep the {names[0]} on your {ev['side']}"
    if ev["type"] == "between":
        return f"go between the {names[0]} and the {names[1]}"
    return f"go through the {names[0]}"


def stop_text(stop, name, wall=False):
    """The stop clause. A distance stop needs its coarse `amount` (rows from before Addendum A
    carry only distance_m: they raise instead of being mis-rendered). A "past" stop measured
    from a wall (`wall`) is past the END of the wall."""
    if stop["type"] == "near":
        return f"stop {stop['relation']} the {name}"
    if stop["type"] == "after_last":
        return "stop just after it"
    if not stop.get("amount"):
        raise ValueError("distance stop without amount")
    end = "the end of " if wall and stop["relation"] == "past" else ""
    return f"stop {stop['amount']} {stop['relation']} {end}the {name}"


def _is_wall(oid, events, kinds=None):
    """A wall: kind "wall" in `kinds` (object id -> kind_hint), or the object of a wall event."""
    return bool(oid) and ((kinds or {}).get(oid) == "wall"
                          or any(e["type"] == "wall" and oid in e["objects"] for e in events))


def _stop_line(events, stop, names, kinds=None):
    return stop_text(stop, names.get(stop["object"]), _is_wall(stop["object"], events, kinds))


def template_description(events, stop, names, kinds=None):
    """The canonical sentence for a route; always passes description_rejects. `kinds` (object
    id -> kind_hint) marks walls inside a `between` for the "past the end of the wall" stop."""
    parts = [event_phrase(e, [names[o] for o in e["objects"]]) for e in events]
    body = ", then ".join(parts) if parts else "go straight ahead"
    text = f"{body}, and {_stop_line(events, stop, names, kinds)}."
    return text[0].upper() + text[1:]


def _noun_forms(noun):
    """Singular plus plurals: +s/es, f->ves, fe->ves, consonant+y->ies."""
    forms = [noun, noun + "s", noun + "es"]
    if noun.endswith("fe"):
        forms.append(noun[:-2] + "ves")
    elif noun.endswith("f"):
        forms.append(noun[:-1] + "ves")
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in "aeiou":
        forms.append(noun[:-1] + "ies")
    return forms


def _find_name(text, name, start):
    noun = head_noun(name)
    if not noun:
        return None
    forms = "|".join(re.escape(f) for f in _noun_forms(noun))
    return re.compile(rf"\b(?:{forms})\b", re.I).search(text, start)


def _owner(text, start, end, spans):
    """Whose a side word (at start:end) is: the next name in its clause, else the previous name."""
    after = [s for s in spans if s[0] >= end]
    nxt = min(after, key=lambda s: s[0]) if after else None
    if nxt is not None and not _BREAK.search(text[end:nxt[0]]):
        return nxt[2]
    before = [s for s in spans if s[1] <= start]
    return max(before, key=lambda s: s[1])[2] if before else None


def _side_rejects(text, events, spans):
    """Each side word belongs to the next name in its clause, else to the previous name. A pass
    (one or two objects) and a wall need their own side word and no other.
    "on/to your left/right" is allowed only when a wall owns it and it says the wall's side; a
    wall's side word counts ONLY in that egocentric form. A bare one ("pass the wall on the
    left", "keep left of the wall") reads as the route's side for objects -- the opposite
    geometry -- so it is side:wall_form and does not count (the wall then lacks its side)."""
    out, got, ego = [], {}, []
    for m in _EGOCENTRIC.finditer(text):
        word = m.group(0).split()[-1].lower()
        owner = _owner(text, m.start(), m.end(), spans)
        wall = isinstance(owner, int) and events[owner]["type"] == "wall"
        if not (wall and events[owner]["side"] == word):
            out.append("side:egocentric")
        if wall:   # counted here: the _RIGHT_ADVERB blanking below must not hide it
            got.setdefault(owner, set()).add(word)
        ego.append((m.start(), m.end(), wall))
    text = _RIGHT_ADVERB.sub(lambda m: " " * len(m.group(0)), text)   # same length: spans stay valid
    for m in _SIDE.finditer(text):
        in_ego = next((w for a, b, w in ego if a <= m.start() and m.end() <= b), None)
        if in_ego:                                       # a wall's egocentric word, already counted
            continue
        owner = _owner(text, m.start(), m.end(), spans)
        if in_ego is None and isinstance(owner, int) and events[owner]["type"] == "wall":
            out.append(f"side:wall_form:{owner}")
            continue
        got.setdefault(owner, set()).add(m.group(1).lower())
    for i, e in enumerate(events):
        words = got.get(i, set())
        if e["type"] in ("pass", "wall"):
            if e["side"] not in words:
                out.append(f"side:missing:{i}")
            if words - {e["side"]}:
                out.append(f"side:opposite:{i}")
        elif words:
            out.append(f"side:stray:{i}")
    if got.get("stop") or got.get(None):
        out.append("side:stray")
    return list(dict.fromkeys(out))


def description_rejects(text, events, stop, names):
    """Why a description is unusable for these events; [] when it is fine."""
    if not text:
        return ["empty"]
    out = [f"drawing:{m.group(0).lower()}" for m in _DRAWING.finditer(text)]
    if re.search(r"\d", text):
        out.append("digits")
    n = len(text.split())
    if not (DESC_WORDS[0] <= n <= DESC_WORDS[1]):
        out.append(f"length:{n}")
    spans, pos = [], 0
    for i, e in enumerate(events):
        for oid in e["objects"]:
            m = _find_name(text, names[oid], pos)
            if m is None:
                out.append(f"name:missing:{i}")
                continue
            spans.append((m.start(), m.end(), i))
            pos = m.end()
        if e["type"] == "between" and not re.search(r"\bbetween\b", text, re.I):
            out.append(f"between:missing:{i}")
        if e["type"] == "through" and not re.search(r"\bthrough\b", text, re.I):
            out.append(f"through:missing:{i}")
    stops = list(_STOP.finditer(text))
    clause = stops[-1].end() if stops else len(text)        # the stop clause: text after the stop word
    if stop["object"]:
        m = _find_name(text, names[stop["object"]], pos)
        if m is None and events and stop["object"] in events[-1]["objects"] and clause >= pos:
            m = _PRONOUN.search(text, clause)                # "...right of the sofa, then stop past it"
        if m is None:
            out.append("stop:name")
        else:
            spans.append((m.start(), m.end(), "stop"))
    if not stops:
        out.append("stop:missing")
    else:
        words = STOP_RELATION_WORDS.get((stop["type"], stop["relation"]))
        if words and not _words_re(words).search(text, clause):
            out.append("stop:relation")
    if stop["type"] == "after_last" and not _AFTER.search(text):
        out.append("stop:after")
    if stop["type"] == "distance":
        words = STOP_AMOUNT_WORDS.get(stop.get("amount"))
        if not words or not stops or not _words_re(words).search(text, clause):
            out.append("stop:amount")
    return out + _side_rejects(text, events, spans)


def name_rejects(name, kind, need):
    """Why a Cosmos object name is unusable; [] when it is fine."""
    if not name:
        return ["name:empty"]
    out = []
    if len(name.split()) > 4:
        out.append("name:long")
    if head_noun(name) in GENERIC_NAMES or head_noun(name) is None:
        out.append("name:generic")
    if _DRAWING.search(name):
        out.append("name:drawing")
    if _SIDE.search(name):
        out.append("name:side")
    if _CONJUNCTION.search(name):
        out.append("name:conjunction")
    if re.search(r"\d", name):
        out.append("name:digits")
    if re.search(r"[,/;()\"'`\u2018\u2019\u201c\u201d]", name):
        out.append("name:punct")
    if (need == "opening") != (kind == "opening"):   # need "wall": a wall, or a door in it ("object")
        out.append("kind:mismatch")
    return out


PROMPT_VERSION = "d1"
KINDS = {"object": "object", "wall": "wall", "opening": "opening", "doorway": "opening", "door": "opening",
         "arch": "opening", "archway": "opening", "passage": "opening", "gap": "opening"}
SHAPES = ["one sentence", "two short sentences", "one sentence joined with 'then'",
          "a sentence that starts with the first thing to pass"]


def naming_image(image_path, polygon, crop_box, px=512):
    """One JPEG, two panels: the frame with the object outlined and the rest dimmed, and a
    close-up of the object's box (multi-image requests are untested on the server)."""
    orig = Image.open(image_path).convert("RGB")
    img = orig.resize((px, px), Image.LANCZOS)
    pts = [(u * px, v * px) for u, v in polygon]
    ctx = img.copy()
    if len(pts) >= 3:
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        ctx = Image.composite(img, Image.eval(img, lambda p: int(p * 0.45)), mask)
        ImageDraw.Draw(ctx).polygon(pts, outline=TARGET_OUTLINE_COLOR, width=4)
    W, H = orig.size
    fx0, fy0, fx1, fy1 = (min(max(c, 0.0), 1.0) for c in (crop_box or [0, 0, 1, 1]))
    x0, y0 = min(int(fx0 * W), W - 8), min(int(fy0 * H), H - 8)
    x1, y1 = max(int(fx1 * W), x0 + 8), max(int(fy1 * H), y0 + 8)
    crop = orig.crop((x0, y0, min(x1, W), min(y1, H)))
    scale = px / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (2 * px, px), (0, 0, 0))
    canvas.paste(ctx, (0, 0))
    canvas.paste(crop, (px + (px - crop.width) // 2, (px - crop.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def naming_prompt(need):
    what = {"opening": "The outlined region is an opening the rover drives through (a doorway, an arch, a gap "
                       "between walls).",
            # final review I3: kind "wall" is any long blob the route runs alongside (median 2.7 m:
            # also sofas, counters, beds), and a near "in front of" anchor that is wall-like
            "wall": "The outlined region is something long that the rover drives alongside or stops in front "
                    "of: a wall, a partition, a door in a wall, or a long piece of furniture such as a sofa "
                    "or a counter.",
            }.get(need, "The outlined region is something the rover drives past.")
    return f"""{SCENE_INTRO}

The picture has two panels. Left: the camera view with one region outlined and everything
else dimmed. Right: a close-up of that region. {what}

1. Name what is inside the outline in 1 to 4 words, the way a person would say it ("grey
   sofa", "stone column", "kitchen island", "open doorway"). Never use the words left or
   right, and never describe the picture itself (outline, panel, close-up, dimming).
2. Say what kind of thing it is: "object" (furniture, an appliance, a column), "wall" (part of
   a wall) or "opening" (a doorway, an arch, a gap between walls).

{_answer_block('{"name": "<name>", "kind": "object|wall|opening"}')}"""


_QUOTES = "\"'`\u2018\u2019\u201c\u201d "
_TRAIL = ".,;:!? "


def parse_name_reply(raw):
    """{"name", "kind"} from a naming reply, or None when there is no usable name."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is None or not isinstance(payload.get("name"), str) or not payload["name"].strip():
        return None
    kind = payload.get("kind")
    kind = KINDS.get(kind.strip().lower(), "object") if isinstance(kind, str) else "object"
    name = " ".join(payload["name"].lower().split()).strip(_QUOTES).rstrip(_TRAIL).strip(_QUOTES).strip()
    name = re.sub(r"^(?:a|an|the)\s+", "", name)
    return {"name": normalize_name(name), "kind": kind}


def choose_shape(route_id):
    return SHAPES[_stable_index("shape:" + route_id, len(SHAPES))]


def compose_prompt(events, stop, names, style, shape, kinds=None):
    steps = "\n".join(f"{i + 1}. {event_phrase(e, [names[o] for o in e['objects']])}" for i, e in enumerate(events))
    steps = steps or "1. go straight ahead"
    return f"""You are writing a route description for a small rover, the way a person standing behind
it would say it. The route, in order:
{steps}
Then: {_stop_line(events, stop, names, kinds)}.

Write it as {style}, in {shape}. Keep every step in this order, keep each object's name and
the word left or right that goes with it, and keep where to stop. You may add small words
("then", "carefully", "keep going"), but add no other objects, sides or directions.
For walls, keep the words 'on your left/right' exactly as given; for other objects never say 'your left/right'.
Write numbers as words. At most 45 words.

{_answer_block('{"description": "<the description>"}')}"""


def parse_compose_reply(raw):
    """The description string from a compose reply, or None (a bare sentence in <answer> is kept)."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is not None:
        text = payload.get("description")
        return " ".join(text.split()) if isinstance(text, str) and text.strip() else None
    if answer and "{" not in answer and len(answer.split()) <= 60:
        return " ".join(answer.strip().strip('"').split()) or None
    return None


def label_object(ask, obj, seed, image=None):
    """Name one object: up to two Cosmos calls (seed, seed + 1). Never raises on a bad reply."""
    need = obj["kind_hint"]
    image = image if image is not None else naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])
    row = {"id": obj["object_id"], "need": need, "prompt_version": PROMPT_VERSION}
    rejects, raws, fin, reply = ["parse"], [], None, None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(image, naming_prompt(need), s)
        raws.append(raw)
        reply = parse_name_reply(raw)
        rejects = ["parse"] if reply is None else name_rejects(reply["name"], reply["kind"], need)
        if not rejects:
            return {**row, "name": reply["name"], "kind": reply["kind"], "passed": True, "rejects": [],
                    "attempts": attempt, "finish": fin}
    return {**row, "name": None, "kind": reply and reply["kind"], "passed": False, "rejects": rejects,
            "attempts": 2, "finish": fin, "raw": raws}


def route_objects(row):
    ids = [o for e in row["events"] for o in e["objects"]]
    return ids + ([row["stop"]["object"]] if row["stop"]["object"] else [])


def label_route(ask_text, row, names, seed, kinds=None):
    """Compose one route's description: two Cosmos attempts, then the template. `kinds` (object
    id -> kind_hint, e.g. the name rows' "need") lets the template say "past the end of" a wall."""
    out = {"id": row["id"], "pair_id": row.get("pair_id"), "kind": row.get("kind"), "prompt_version": PROMPT_VERSION}
    missing = [o for o in route_objects(row) if not names.get(o)]
    if missing:
        return {**out, "passed": False, "description": None, "instruction_source": None,
                "rejects": ["unnamed_object"], "missing": missing}
    style, shape = choose_style(row["id"]), choose_shape(row["id"])
    prompt = compose_prompt(row["events"], row["stop"], names, style, shape, kinds)
    rejects, raws = [], []
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, _ = ask_text(prompt, s)
        raws.append(raw)
        text = parse_compose_reply(raw)
        rejects = description_rejects(text, row["events"], row["stop"], names)
        if not rejects:
            return {**out, "description": text, "instruction_source": "cosmos", "passed": True, "rejects": [],
                    "cosmos_rejects": [], "style": style, "shape": shape, "attempts": attempt}
    tpl = template_description(row["events"], row["stop"], names, kinds)
    trej = description_rejects(tpl, row["events"], row["stop"], names)
    return {**out, "description": tpl if not trej else None, "instruction_source": "template",
            "passed": not trej, "rejects": trej, "cosmos_rejects": rejects, "style": style, "shape": shape,
            "attempts": 2, "raw": raws}


def select_routes(rows, limit=None):
    ok = [r for r in rows if r.get("reason") is None]
    return ok[:limit] if limit else ok


def summarize_d1(rows, stage):
    ok = [r for r in rows if not r.get("error")]
    out = {"n": len(rows), "errors": len(rows) - len(ok),
           "accept_rate": round(sum(bool(r.get("passed")) for r in ok) / len(ok), 4) if ok else 0.0,
           "rejects": dict(Counter(x for r in ok for x in r.get("rejects", [])).most_common(15))}
    if stage == "compose":
        passed = [r for r in ok if r.get("passed")]
        out["template_share"] = round(sum(r.get("instruction_source") == "template" for r in passed)
                                      / len(passed), 4) if passed else 0.0
        out["cosmos_rejects"] = dict(Counter(x for r in ok for x in r.get("cosmos_rejects", [])).most_common(15))
    return out


def d1_gate_failures(summary, stage, min_accept=0.9, max_template=0.1):
    fails = []
    if summary["accept_rate"] < min_accept:
        fails.append(f"accept rate {summary['accept_rate']:.3f} < {min_accept}")
    if stage == "compose" and summary.get("template_share", 0) > max_template:
        fails.append(f"template share {summary['template_share']:.3f} > {max_template}")
    if summary.get("errors", 0) > 0:
        fails.append(f"request errors {summary['errors']} > 0")
    return fails


def name_rows_split(rows):
    """(named {id: name}, rejected {ids that ran but failed}); error rows are in neither."""
    named = {r["id"]: r["name"] for r in rows if r.get("passed") and not r.get("error")}
    rejected = {r["id"] for r in rows if not r.get("passed") and not r.get("error")}
    return named, rejected


def names_missing(row, named, rejected):
    """Route objects with no name row or only an error row (a retryable gap, not a rejection)."""
    return [o for o in route_objects(row) if o not in named and o not in rejected]


ENTRY_SHARE = 70   # percent of frames whose records get the entry-only prompt (user, 2026-09-30)


def prompt_form(sample_id):
    """'entry' (no goal in the prompt) or 'goal', fixed per frame so a pair shares it."""
    return "entry" if _stable_index("form:" + sample_id, 100) < ENTRY_SHARE else "goal"


def desc_check(ev, objects):
    """What eval.description_follow needs, carried in the record (no file access at eval time)."""
    return {"pose_xz": ev["pose_xz"], "heading_xz": ev["heading_xz"], "goal_xz": ev["goal_xz"],
            "events": [{"type": e["type"], "side": e["side"],
                        "obstacles": [objects[o]["cells_xz"] for o in e["objects"]],
                        "gap_xz": e.get("gap_xz")} for e in ev["events"]]}


def with_description(rec, text, source, form, check):
    new = copy.deepcopy(rec)
    path = _answer_path(new)
    new["conversations"][0]["value"] = (description_entry_prompt(path[0], text) if form == "entry"
                                        else instruction_endpoints_prompt(path[0], path[-1], text))
    new["habitat_meta"].update({"instruction": text, "instruction_source": source, "prompt_form": form,
                                "desc_check": check, "goal_in_prompt": form == "goal"})
    return new


def finalize_descriptions(label_recs, events_by_id, desc_by_id, objects):
    kept, geo, unlabelled, rejected = [], 0, 0, 0
    for rec in label_recs:
        ev, row = events_by_id.get(rec["id"]), desc_by_id.get(rec["id"])
        if not ev or ev.get("reason") is not None:
            geo += 1
            continue
        if row is None:
            unlabelled += 1
            continue
        if not row.get("passed"):
            rejected += 1
            continue
        form = prompt_form(rec["habitat_meta"]["sample_id"])
        kept.append(with_description(rec, row["description"], row["instruction_source"], form, desc_check(ev, objects)))
    groups = {}
    for r in kept:
        if r["habitat_meta"].get("pair_id"):
            groups.setdefault(r["habitat_meta"]["pair_id"], []).append(r)
    pairs = [r for g in groups.values() if len(g) == 2 for r in g]
    return {"kept": kept, "pairs": pairs,
            "stats": {"records": len(label_recs), "kept": len(kept), "dropped_geometry": geo,
                      "unlabelled": unlabelled, "rejected": rejected, "pairs_kept": len(pairs) // 2,
                      "entry_form": sum(r["habitat_meta"]["prompt_form"] == "entry" for r in kept),
                      "template": sum(r["habitat_meta"]["instruction_source"] == "template" for r in kept)}}


def as_form(rec, form):
    hm = rec["habitat_meta"]
    return with_description(rec, hm["instruction"], hm["instruction_source"], form, hm["desc_check"])


def shuffled_texts(recs, seed=0):
    """Entry-form copies whose description comes from a record of another frame (the check
    and the answer stay the record's own): a text-reading model should get worse here."""
    order = list(recs)
    random.Random(seed).shuffle(order)
    n = len(order)
    for k in range(1, n):
        src = [order[(i + k) % n] for i in range(n)]
        if all(a["habitat_meta"]["sample_id"] != b["habitat_meta"]["sample_id"] for a, b in zip(order, src)):
            break
    else:
        raise ValueError("cannot shuffle: every record shares one frame")
    return [with_description(a, b["habitat_meta"]["instruction"], "shuffled", "entry", a["habitat_meta"]["desc_check"])
            for a, b in zip(order, src)]
