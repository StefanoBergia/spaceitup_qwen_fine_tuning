"""Route descriptions (prompt d4 since d2; d2 / d1 before): turning geometry's events into checked sentences.

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
    _DRAWING, _EGOCENTRIC, _TRAILING, STYLES, TARGET_OUTLINE_COLOR, _answer_block, _answer_path, _stable_index,
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
UNIT_MODIFIERS = {"shelving", "shelf", "storage", "kitchen", "wall", "tv", "display", "corner", "drawer", "cabinet"}
_CONJUNCTION = re.compile(r"\b(?:and|then|or)\b", re.I)     # would split the name at _BREAK
# "right" meaning "exactly", not a side: blanked out before any side-word check. "next to",
# "by" and "beside" are no longer stop relations (Addendum A) but stay here: a description may
# still say "right next to" in passing, and that "right" is never a side word.
_RIGHT_ADVERB = re.compile(r"\bright(?= (?:after|past|beyond|before|in front of|next to|by|beside)\b)", re.I)
_PRONOUN = re.compile(r"\b(?:it|them)\b", re.I)
# d2: "turn left", "turns right", "turning to the left", "turn to your right". Its side word belongs to the turn
# event, never to the next name, and is claimed BEFORE _RIGHT_ADVERB blanks "right after" (Review Focus 1).
_TURN = re.compile(r"\bturn(?:s|ed|ing)?\s+(?:to\s+(?:the|your)\s+)?(left|right)\b", re.I)
TURN_RELATION_WORDS = {"behind": r"\bbehind\b", "round": r"\b(?:a)?round\b"}
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
    ROVER's side the wall is on ("keep the white wall on your left"); a turn names the rover's own turn and,
    when anchored, the object it turns behind / round (d2)."""
    if ev["type"] == "pass":
        return f"go {ev['side']} of the " + " and the ".join(names)
    if ev["type"] == "wall":
        return f"keep the {names[0]} on your {ev['side']}"
    if ev["type"] == "between":
        return f"go between the {names[0]} and the {names[1]}"
    if ev["type"] == "turn":
        rel = ev.get("relation")
        return f"turn {ev['side']}" + (f" {rel} the {names[0]}" if rel and names else "")
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
    if events and events[0]["type"] == "turn":
        body = "go ahead, then " + body              # the route runs straight before it turns (route_events)
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


def turn_rejects(text, events):
    """The turn rules, which need no names (finalize re-checks reused d1 texts with them): a turn phrase only
    where the events have a turn, saying the turn's side and no other, and an anchored turn's relation word
    ("behind" / "round") in the clause after the turn phrase."""
    found = list(_TURN.finditer(text))
    turns = [i for i, e in enumerate(events) if e["type"] == "turn"]
    if not turns:
        return ["turn:stray"] if found else []
    i, e = turns[0], events[turns[0]]
    sides = {m.group(1).lower() for m in found}
    out = []
    if e["side"] not in sides:
        out.append(f"turn:side:missing:{i}")
    if sides - {e["side"]}:
        out.append(f"turn:side:opposite:{i}")
    rel = e.get("relation")
    if rel and found:
        m = next((m for m in found if m.group(1).lower() == e["side"]), found[0])
        b = _BREAK.search(text, m.end())
        if not re.search(TURN_RELATION_WORDS[rel], text[m.end(): b.start() if b else len(text)], re.I):
            out.append(f"turn:relation:{i}")
    return out


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
            if m is None and e["type"] == "turn" and i > 0 and oid in events[i - 1]["objects"]:
                m = _PRONOUN.search(text, pos)       # "...right of the table, then turn left behind it"
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
    # the turn phrase's side word is the turn's: blank it (same length, spans stay valid) before side ownership
    side_text = _TURN.sub(lambda m: " " * len(m.group(0)), text)
    return out + turn_rejects(text, events) + _side_rejects(side_text, events, spans)


def name_rejects(name, kind, need):
    """Why a Cosmos object name is unusable; [] when it is fine."""
    if not name:
        return ["name:empty"]
    out = []
    if len(name.split()) > 4:
        out.append("name:long")
    # Rule N2: "unit" is not generic when preceded by a modifier from UNIT_MODIFIERS
    noun = head_noun(name)
    if noun in GENERIC_NAMES or noun is None:
        # Use the same word extraction as head_noun to strip trailing phrases correctly
        words = re.findall(r"[a-z]+", _TRAILING.sub("", (name or "").lower()))
        is_valid_unit = noun == "unit" and len(words) >= 2 and words[-2] in UNIT_MODIFIERS
        if not is_valid_unit:
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
    # Rule N1: kind "opening" for need "wall" is NOT a kind:mismatch (doors/windows in a wall)
    # Reject only: need "opening" + kind != "opening", or need "object" + kind "opening"
    if kind == "opening" and need not in ("opening", "wall"):
        out.append("kind:mismatch")
    elif kind != "opening" and need == "opening":
        out.append("kind:mismatch")
    return out


PROMPT_VERSION = "d4"

# d1 speaking styles: the shared STYLES minus the "why or where it leads" one (audit of gate r2: both
# unnatural labels came from it), plus a step-by-step one. The shared STYLES (i4) is unchanged.
D1_STYLES = [s for s in STYLES if s != "a command that also says briefly why or where it leads"] + [
    "a calm, step-by-step instruction"]


def choose_d1_style(record_id):
    return D1_STYLES[_stable_index("style:" + record_id, len(D1_STYLES))]
KINDS = {"object": "object", "wall": "wall", "opening": "opening", "doorway": "opening", "door": "opening",
         "arch": "opening", "archway": "opening", "passage": "opening", "gap": "opening"}
SHAPES = ["one sentence", "two short sentences", "one sentence joined with 'then'",
          "a sentence that starts with the first thing to pass"]


def naming_image(image_path, polygon, crop_box, px=512):
    """One square JPEG (black bands above and below), two panels: the frame with the object outlined and the rest dimmed, and a
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
    # letterboxed to square: Gemma's processor resizes to 896x896 without keeping the aspect, which
    # would stretch each panel 2x (final review I1)
    square = Image.new("RGB", (2 * px, 2 * px), (0, 0, 0))
    square.paste(canvas, (0, px // 2))
    buf = io.BytesIO()
    square.save(buf, "JPEG", quality=90)
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
   Name the whole thing the outline covers, not a small part on it (a handle, a switch, a socket, a leg).
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

Write it as {style}, in {shape}. Keep every step in this order and keep each object's name. Copy each step's direction words exactly
as written: in a 'keep the ... on your left/right' step, 'on your left' / 'on your right' stays that
way, whatever the thing is (a wall, a sofa, a counter); in a 'go left of / right of' step, 'left of'
/ 'right of' stays that way; a step with 'between' or 'through' has no left or right, so do not add
one. In a 'turn left' / 'turn right' step, keep 'turn left' / 'turn right' exactly (it is the rover's own
turn) and keep its 'behind' / 'round' and its object; never add a turn the steps do not have. Keep where
to stop. You may add small words
("then", "carefully", "keep going"), but add no other objects, sides or directions.
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


def name_clash(row, names):
    """True when two DIFFERENT route objects (event members + stop) share a head noun under `names`.
    Unnamed objects never clash (that is `unnamed_object`); the same id twice is one object."""
    seen = set()
    for o in dict.fromkeys(route_objects(row)):
        h = head_noun(names[o]) if names.get(o) else None
        if h:
            if h in seen:
                return True
            seen.add(h)
    return False


def _usable(row, names):
    return all(names.get(o) for o in route_objects(row)) and not name_clash(row, names)


def choose_names(row, names, fallback_names=None):
    """The A5 naming rule for one route: (names, "primary", None) when `names` serves it (every
    route object named, no two sharing a head noun), else (fallback_names, "fallback", None)
    under the same test, else (None, None, "unnamed_object" | "name:clash")."""
    if _usable(row, names):
        return names, "primary", None
    if fallback_names is not None and _usable(row, fallback_names):
        return fallback_names, "fallback", None
    tried = [names] + ([fallback_names] if fallback_names is not None else [])
    if any(not any(n.get(o) for n in tried) for o in route_objects(row)):
        return None, None, "unnamed_object"
    return None, None, "name:clash"


def label_route(ask_text, row, names, seed, kinds=None, fallback_names=None):
    """Compose one route's description: two Cosmos attempts, then the template. `kinds` (object
    id -> kind_hint, e.g. the name rows' "need") lets the template say "past the end of" a wall.
    `names` is tried first; if it leaves an object unnamed or two objects share a head noun,
    `fallback_names` is tried; the row's "namer" says which one was used."""
    out = {"id": row["id"], "pair_id": row.get("pair_id"), "kind": row.get("kind"), "prompt_version": PROMPT_VERSION}
    chosen, namer, reject = choose_names(row, names, fallback_names)
    if chosen is None:
        if reject == "unnamed_object":
            tried = [names] + ([fallback_names] if fallback_names is not None else [])
            missing = [o for o in route_objects(row) if not any(n.get(o) for n in tried)]
            return {**out, "passed": False, "description": None, "instruction_source": None,
                    "rejects": ["unnamed_object"], "missing": missing}
        return {**out, "passed": False, "description": None, "instruction_source": None,
                "rejects": ["name:clash"]}
    names = chosen
    key = row.get("style_key") or row["id"]     # steering pairs share one style and shape
    style, shape = choose_d1_style(key), choose_shape(key)
    prompt = compose_prompt(row["events"], row["stop"], names, style, shape, kinds)
    rejects, raws = [], []
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, _ = ask_text(prompt, s)
        raws.append(raw)
        text = parse_compose_reply(raw)
        rejects = description_rejects(text, row["events"], row["stop"], names)
        if not rejects:
            return {**out, "description": text, "instruction_source": "cosmos", "passed": True, "rejects": [],
                    "cosmos_rejects": [], "style": style, "shape": shape, "attempts": attempt,
                    "namer": namer}
    tpl = template_description(row["events"], row["stop"], names, kinds)
    trej = description_rejects(tpl, row["events"], row["stop"], names)
    return {**out, "description": tpl if not trej else None, "instruction_source": "template",
            "passed": not trej, "rejects": trej, "cosmos_rejects": rejects, "style": style, "shape": shape,
            "attempts": 2, "raw": raws, "namer": namer}


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
        out["namer"] = dict(Counter(r["namer"] for r in passed if r.get("namer")))
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


def objects_needing_fallback(routes, name_rows):
    """Objects of the routes NOT usable under `name_rows` (the primary / Gemma name rows): an
    object that is unnamed, rejected or only an error row there counts as unnamed, and a route
    with a name_clash is unusable. These are the only objects the fallback namer must name."""
    named, _ = name_rows_split(name_rows)
    return {o for r in routes if not _usable(r, named) for o in route_objects(r)}


def names_missing(row, named, rejected, fallback_named=None, fallback_rejected=None):
    """Route objects with no name row or only an error row (a retryable gap, not a rejection).
    A name in either file counts as present; an unnamed object is a rejection only when every
    file given ran it and rejected it."""
    files = [(named, rejected)] + ([(fallback_named, fallback_rejected or set())]
                                   if fallback_named is not None else [])
    return [o for o in route_objects(row)
            if not any(o in n for n, _ in files) and any(o not in r for _, r in files)]


ENTRY_SHARE = 70   # percent of frames whose records get the entry-only prompt (user, 2026-09-30)


def prompt_form(sample_id):
    """'entry' (no goal in the prompt) or 'goal', fixed per frame so a pair shares it."""
    return "entry" if _stable_index("form:" + sample_id, 100) < ENTRY_SHARE else "goal"


def desc_check(ev, objects):
    """What eval.description_follow needs, carried in the record (no file access at eval time). A turn
    event also carries its turn point, pre-turn heading and the GT end's lateral offset (d2)."""
    return {"pose_xz": ev["pose_xz"], "heading_xz": ev["heading_xz"], "goal_xz": ev["goal_xz"],
            "events": [_check_event(e, objects) for e in ev["events"]]}


def _check_event(e, objects):
    out = {"type": e["type"], "side": e["side"], "obstacles": [objects[o]["cells_xz"] for o in e["objects"]],
           "gap_xz": e.get("gap_xz")}
    if e["type"] == "turn":
        out.update({k: e[k] for k in ("turn_xz", "pre_heading_xz", "gt_lateral_m")})
    return out


def with_description(rec, text, source, form, check):
    new = copy.deepcopy(rec)
    path = _answer_path(new)
    new["conversations"][0]["value"] = (description_entry_prompt(path[0], text) if form == "entry"
                                        else instruction_endpoints_prompt(path[0], path[-1], text))
    new["habitat_meta"].update({"instruction": text, "instruction_source": source, "prompt_form": form,
                                "desc_check": check, "goal_in_prompt": form == "goal"})
    return new


def finalize_descriptions(label_recs, events_by_id, desc_by_id, objects, recheck=None):
    kept, geo, unlabelled, rejected, rechecked = [], 0, 0, 0, 0
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
        if recheck is not None and recheck(row["description"], ev["events"]):
            rechecked += 1                    # d2: a reused d1 text that breaks a d2 rule (spec: finalize re-checks)
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
                      "unlabelled": unlabelled, "rejected": rejected, "recheck_failed": rechecked, "pairs_kept": len(pairs) // 2,
                      "entry_form": sum(r["habitat_meta"]["prompt_form"] == "entry" for r in kept),
                      "template": sum(r["habitat_meta"]["instruction_source"] == "template" for r in kept)}}


def _geometry_key(row):
    st = row["stop"]
    return ([(e["type"], e["side"], tuple(e["objects"])) for e in row["events"]],
            (st["type"], st["relation"], st.get("amount"), st["object"]))


def same_description_geometry(old, new):
    """True when a d2 events row describes its route as the d1 row did: both described, same events (type,
    side, objects; arcs may move) and the same stop. Its d1 text then stays true: no d1 text says "turn"."""
    if old.get("reason") is not None or new.get("reason") is not None:
        return False
    return _geometry_key(old) == _geometry_key(new)


def seed_d2_labels(old_events, new_events, new_objects, old_names, old_desc):
    """What the d2 label job can start from (spec "Seeding"): the d1 name rows of objects that still exist
    ({"gemma": rows, "cosmos": rows}, error rows left out so they are retried), the d1 desc row of every
    unchanged described route (tagged "reused_from": "d1"), and the described routes that need composing."""
    ids = {o["object_id"] for o in new_objects}
    names = {tag: [r for r in rows if r["id"] in ids and not r.get("error")] for tag, rows in old_names.items()}
    old = {r["id"]: r for r in old_events}
    desc = {r["id"]: r for r in old_desc if not r.get("error")}
    reused, changed = [], []
    for r in new_events:
        if r.get("reason") is not None:
            continue
        o = old.get(r["id"])
        if o is not None and r["id"] in desc and same_description_geometry(o, r):
            reused.append({**desc[r["id"]], "reused_from": "d1"})
        else:
            changed.append(r)
    return {"names": names, "desc": reused, "changed": changed}


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
