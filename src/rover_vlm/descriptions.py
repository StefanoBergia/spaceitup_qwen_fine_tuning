"""Route descriptions (prompt d5 since d2 T2; d4 / d1 before): turning geometry's events into checked sentences.

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
from .route_events import stop_kind
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
_BEHIND = re.compile(r"\bbehind\b", re.I)
# The words that say each stop relation. The stop clause (the text after the last stop word)
# must hold one of its own relation's words, so a swapped relation ("a little before the
# armchair" for a past stop) is rejected as stop:relation. after_last is checked by _AFTER.
STOP_RELATION_WORDS = {
    ("near", "in front of"): ("in front of", "right in front of", "just in front of"),
    ("near", "just past"): ("just past", "right past", "just beyond", "just after"),
    ("distance", "before"): ("before", "short of"),
    ("distance", "past"): ("past", "after", "beyond"),
    ("object", "in front of"): ("in front of",),            # d3 (spec 2026-11-10): an object within 1.5 m
    ("object", "behind"): ("behind",),
    ("object", "next to"): ("next to", "beside"),
    ("wall", "in front of"): ("in front of",),              # d3 Addendum A1: the wall ahead, when no place fits
}
# d3 places (rover_vlm.stop_places.PLACES): the canonical clause, and the words a text may use for it
PLACE_TEXT = {"doorway": "in the doorway", "corner": "in the corner", "hallway_end": "at the end of the hallway",
              "hallway": "in the hallway", "wall": "by the wall", "room_middle": "in the middle of the room"}
PLACE_WORDS = {"doorway": ("doorway", "door opening"), "corner": ("corner",),
               "hallway_end": ("end of the hallway", "end of the corridor", "end of the hall"),
               "hallway": ("in the hallway", "in the corridor", "in the hall"),
               "wall": ("by the wall", "beside the wall", "next to the wall", "against the wall", "along the wall"),
               "room_middle": ("middle of the room", "centre of the room", "center of the room")}
# a d3 visible stop says where the rover ends up, never how far: these distance words are rejected after the stop word
_DISTANCE = re.compile(r"\b(?:a little|a bit|slightly|well|far|a good way)\s+(?:past|before|beyond|after|short of)\b"
                       r"|\bjust\s+(?:past|after|beyond)\b|\bshort of\b|\b(?:metres?|meters?|steps?)\b", re.I)
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
    from a wall (`wall`) is past the END of the wall. d3: an object stop says its relation, a place stop its place."""
    if stop["type"] in ("object", "wall"):
        return f"stop {stop['relation']} the {name}"
    if stop["type"] == "place":
        return "stop " + PLACE_TEXT[stop["place"]]
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


def behind_clause(events, stop, names, verb="proceed"):
    """The behind ending (d2 T2): '<verb> behind it' when the ending object is the last object of the last step,
    else '<verb> behind the <name>'."""
    last = events[-1]["objects"] if events else []
    target = "it" if last and last[-1] == stop["object"] else f"the {names[stop['object']]}"
    return f"{verb} behind {target}"


REWRITE_VERBS = ("proceed", "continue", "carry on", "go on")   # d2 T2: the verb of a rewritten behind ending
# the connector just before a text's last stop word: ", then" / ", and then" / ", and" / " and" / " then" / ". Then" / ". "
_CONNECTOR = re.compile(r"(?:\s*[,;]\s*(?:and\s+then|then|and)?|\s+(?:and\s+then|then|and)|\s*\.\s*(?:then|and)?)"
                        r"\s*(?:please\s+)?$", re.I)


def rewrite_ending(text, events, stop, names, route_id):
    """(new text, None), or (None, reason). A d1 text with its final stop clause -- the last stop word to the end of
    the text plus the connector before it -- replaced by ", then <verb> behind it / the X" (behind_clause; the verb
    from REWRITE_VERBS by a stable hash of the route id), keeping the text's own end punctuation. Reasons:
    "no_stop_clause" (no stop word), "no_connector" (nothing before the stop clause, or no connector), "rejects"
    (the new text fails description_rejects)."""
    verb = REWRITE_VERBS[_stable_index("verb:" + route_id, len(REWRITE_VERBS))]
    new, why = _replace_final_clause(text, behind_clause(events, stop, names, verb))
    if new is None:
        return None, why
    return (new, None) if not description_rejects(new, events, stop, names) else (None, "rejects")


def _replace_final_clause(text, clause):
    """(text with its final stop clause -- the last stop word to the end plus the connector before it -- replaced by
    ", then <clause>", keeping the text's own end punctuation, None) or (None, "no_stop_clause" / "no_connector")."""
    stops = list(_STOP.finditer(text or ""))
    if not stops:
        return None, "no_stop_clause"
    c = _CONNECTOR.search(text[: stops[-1].start()])
    head = text[: c.start()].rstrip(" ,;") if c else ""
    if not head:
        return None, "no_connector"
    end = text.rstrip()[-1:]
    return f"{head}, then {clause}{end if end in '.?!' else '.'}", None


def rewrite_stop(text, events, stop, names):
    """(new text, None) or (None, reason): a d2 text whose final stop clause becomes the d3 stop line (stop_text), d3
    seeding. Reasons as rewrite_ending: "no_stop_clause", "no_connector", "rejects"."""
    new, why = _replace_final_clause(text, _stop_line(events, stop, names))
    if new is None:
        return None, why
    return (new, None) if not description_rejects(new, events, stop, names) else (None, "rejects")


def template_description(events, stop, names, kinds=None):
    """The canonical sentence for a route; always passes description_rejects. `kinds` (object
    id -> kind_hint) marks walls inside a `between` for the "past the end of the wall" stop."""
    parts = [event_phrase(e, [names[o] for o in e["objects"]]) for e in events]
    body = ", then ".join(parts) if parts else "go straight ahead"
    if events and events[0]["type"] == "turn":
        body = "go ahead, then " + body              # the route runs straight before it turns (route_events)
    if stop["type"] == "turn":
        text = f"{body}."                                    # the hidden turn is the ending (d2 T2)
    elif stop["type"] == "behind":
        text = f"{body}, then {behind_clause(events, stop, names)}."
    else:
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


def ending_rejects(text, stop):
    """The ending rules that need no names (finalize re-checks reused and rewritten texts with them, d2 T2): a behind
    or turn ending never says where the route stops, and a behind ending says "behind"."""
    if stop is None or stop["type"] not in ("behind", "turn"):
        return []
    out = ["ending:stop_word"] if _STOP.search(text or "") else []
    if stop["type"] == "behind" and not _BEHIND.search(text or ""):
        out.append("ending:behind")
    return out


def d2_rejects(text, ev):
    """finalize's re-check of a d2 text against its events row: the turn rules and the ending rules."""
    return turn_rejects(text, ev["events"]) + ending_rejects(text, ev.get("stop"))


def _d3_clause_rejects(text, stop, clause):
    """A d3 stop clause (the text after the last stop word, from `clause`): no distance words, and a place stop says
    one of its place's words."""
    out = ["stop:distance"] if _DISTANCE.search(text, clause) else []
    if stop["type"] == "place" and not _words_re(PLACE_WORDS[stop["place"]]).search(text, clause):
        out.append("stop:place")
    return out


def visible_stop_rejects(text, stop):
    """The names-free checks of a d3 visible stop (finalize re-checks reused texts with them): a stop word, then no
    distance words, the place's words (place) or the relation's words (object). [] for every other stop."""
    if stop_kind(stop) is None:
        return []
    stops = list(_STOP.finditer(text or ""))
    if not stops:
        return ["stop:missing"]
    clause = stops[-1].end()
    out = _d3_clause_rejects(text, stop, clause)
    if stop["type"] in ("object", "wall") and \
            not _words_re(STOP_RELATION_WORDS[(stop["type"], stop["relation"])]).search(text, clause):
        out.append("stop:relation")
    return out


def d3_rejects(text, ev):
    """finalize's re-check of a d3 text against its events row: d2_rejects plus the visible-stop checks."""
    return d2_rejects(text, ev) + visible_stop_rejects(text, ev.get("stop"))


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
    if stop["type"] in ("behind", "turn"):
        out += ending_rejects(text, stop)
        if stop["type"] == "behind":
            hits = list(_BEHIND.finditer(text))
            start = hits[-1].end() if hits else len(text)
            m = _find_name(text, names[stop["object"]], start)
            if m is None and events and stop["object"] in events[-1]["objects"]:
                m = _PRONOUN.search(text, start)                 # "...right of the sofa, then carry on behind it"
            if m is None:
                out.append("ending:name")
            else:
                spans.append((m.start(), m.end(), "stop"))
    else:
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
        if stop["type"] in ("object", "place", "wall") and stops:
            out += _d3_clause_rejects(text, stop, clause)
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


PROMPT_VERSION = "d6"

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


def naming_prompt(need, version="n1"):
    """The naming prompt: n1 (A5, today's) or n2 (naming fix, spec 2026-10-15: name what fills the outline, building parts
    first, no furniture example for long blobs)."""
    if version == "n1":
        return _naming_prompt_n1(need)
    if version == "n2":
        return _naming_prompt_n2(need)
    raise ValueError(f"unknown naming prompt version {version!r}")


def _naming_prompt_n1(need):
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


_N2_WHAT = {"opening": "an opening (a doorway, an arch, a gap between walls). The rover drives through it.",
            "wall": "something long that the rover drives alongside or stops in front of."}


def _naming_prompt_n2(need):
    what = _N2_WHAT.get(need, "something the rover drives past.")
    return f"""{SCENE_INTRO}

The picture has two panels. Left: the camera view with one region outlined and everything
else dimmed. Right: a close-up of that region. The outlined region is {what}

1. Name the single thing that fills most of the outline, in 1 to 4 words, the way a person would
   say it. Look at what is inside the outline, not at what is next to it or behind it.
   If the outline is mostly part of the building, say which part ("white wall", "brick wall",
   "stone column", "glass partition", "open doorway", "wooden door", "stair railing").
   Name furniture or an appliance only if the outline is clearly on it ("grey sofa",
   "kitchen island", "refrigerator"). Never use the words left or right, and never describe the
   picture itself (outline, panel, close-up, dimming). Name the whole thing the outline covers,
   not a small part on it (a handle, a switch, a socket, a leg).
2. Say what kind of thing it is: "wall" (part of the building: a wall, a column or pillar, a door,
   a partition, a railing), "object" (furniture, an appliance, any other object) or "opening"
   (a doorway, an arch, a gap between walls).

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


ENDING_RULES = {
    "behind": "End with the route going on behind the {name} (proceed, continue, carry on or go on behind it); "
              "never say where it stops.",
    "turn": "End with the turn; never say where it stops.",
}


def compose_prompt(events, stop, names, style, shape, kinds=None):
    steps = "\n".join(f"{i + 1}. {event_phrase(e, [names[o] for o in e['objects']])}" for i, e in enumerate(events))
    steps = steps or "1. go straight ahead"
    if stop["type"] == "turn":
        then, keep = "", ENDING_RULES["turn"]
    elif stop["type"] == "behind":
        name = names[stop["object"]]
        then, keep = f"\nThen: proceed behind the {name}.", ENDING_RULES["behind"].format(name=name)
    else:
        then, keep = f"\nThen: {_stop_line(events, stop, names, kinds)}.", "Keep where to stop."
    return f"""You are writing a route description for a small rover, the way a person standing behind
it would say it. The route, in order:
{steps}{then}

Write it as {style}, in {shape}. Keep every step in this order and keep each object's name. Copy each step's direction words exactly
as written: in a 'keep the ... on your left/right' step, 'on your left' / 'on your right' stays that
way, whatever the thing is (a wall, a sofa, a counter); in a 'go left of / right of' step, 'left of'
/ 'right of' stays that way; a step with 'between' or 'through' has no left or right, so do not add
one. In a 'turn left' / 'turn right' step, keep 'turn left' / 'turn right' exactly (it is the rover's own
turn) and keep its 'behind' / 'round' and its object; never add a turn the steps do not have. {keep} You may add small words
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


def label_object(ask, obj, seed, image=None, name_prompt="n1"):
    """Name one object: up to two calls (seed, seed + 1) with naming prompt `name_prompt`. Never raises on a bad reply."""
    need = obj["kind_hint"]
    image = image if image is not None else naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])
    prompt = naming_prompt(need, name_prompt)
    row = {"id": obj["object_id"], "need": need, "prompt_version": PROMPT_VERSION, "name_prompt": name_prompt}
    rejects, raws, fin, reply = ["parse"], [], None, None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(image, prompt, s)
        raws.append(raw)
        reply = parse_name_reply(raw)
        rejects = ["parse"] if reply is None else name_rejects(reply["name"], reply["kind"], need)
        if not rejects:
            return {**row, "name": reply["name"], "kind": reply["kind"], "passed": True, "rejects": [],
                    "attempts": attempt, "finish": fin}
    return {**row, "name": None, "kind": reply and reply["kind"], "passed": False, "rejects": rejects,
            "attempts": 2, "finish": fin, "raw": raws}


def _name_key(name):
    return " ".join((name or "").lower().split())


def candidate_names(oid, variant_names):
    """The chooser's candidates for one object (Addendum C): the distinct names (case-insensitive) the variants gave it,
    each {"name", "from": [variants]}; variants visited in sorted order, the first spelling kept; shuffled with a seed
    stable per object, so a re-run shows the same order."""
    by = {}
    for v in sorted(variant_names):
        n = variant_names[v].get(oid)
        if n:
            by.setdefault(_name_key(n), {"name": n, "from": []})["from"].append(v)
    out = list(by.values())
    random.Random("choose:" + oid).shuffle(out)
    return out


def choose_prompt(need, names, version="c1"):
    """The chooser prompt (Addendum C): the n2 guidance and per-need hint, the lettered candidate names, "none"."""
    if version != "c1":
        raise ValueError(f"unknown choose prompt version {version!r}")
    what = _N2_WHAT.get(need, "something the rover drives past.")
    lines = "\n".join(f"{chr(97 + i)}. {n}" for i, n in enumerate(names))
    return f"""{SCENE_INTRO}

The picture has two panels. Left: the camera view with one region outlined and everything
else dimmed. Right: a close-up of that region. The outlined region is {what}

Here are names other people gave the outlined thing:
{lines}

Pick the one name that best names the single thing that fills most of the outline. Look at what
is inside the outline, not at what is next to it or behind it. If the outline is mostly part of
the building, the right name says which part (a wall, a column, a door, a partition, a railing);
a furniture or appliance name is right only if the outline is clearly on it. If no name fits,
answer "none".

{_answer_block('{"choice": "<letter>|none"}')}"""


_choose_text = choose_prompt      # choose_object's parameter of the same name shadows the function


def parse_choice_reply(raw, n):
    """The chosen letter ("a".. the n-th letter), "none", or None: one letter, alone, in brackets, or followed by
    "." ":" ")" and text ("c. white wall"); a letter past the list, two letters ("b or c"), a name ("a white wall") or
    prose are None."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    choice = payload.get("choice") if payload is not None else (answer if answer and "{" not in answer else None)
    if not isinstance(choice, str):
        return None
    s = choice.strip().lower()
    if s.startswith("none"):
        return "none"
    m = re.fullmatch(r"\(?([a-z])\)?(?:[.:)]\s*.*)?", s, re.S)
    return m.group(1) if m and ord(m.group(1)) - 97 < n else None


def choose_object(ask, obj, candidates, fallback, seed, image=None, choose_prompt="c1"):
    """Choose one object's name among `candidates` (candidate_names) with up to two calls (seed, seed + 1); "none" or two
    unreadable replies give the object `fallback` (today's name) -- per object, written into this row (Addendum C).
    One candidate is the pick and no candidate means the fallback, both without a call. Never raises on a bad reply."""
    names = [c["name"] for c in candidates]
    row = {"id": obj["object_id"], "need": obj["kind_hint"], "candidates": names, "single": len(names) == 1,
           "choose_prompt": choose_prompt, "rejects": [], "attempts": 0, "finish": None}
    fall = {"name": fallback, "passed": fallback is not None, "name_source": ["V0_fallback"]}
    if not names:
        return {**row, "choice": None, **fall}
    if len(names) == 1:
        return {**row, "choice": "a", "name": names[0], "passed": True, "name_source": candidates[0]["from"]}
    image = image if image is not None else naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])
    prompt = _choose_text(obj["kind_hint"], names, choose_prompt)
    raws = []
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(image, prompt, s)
        raws.append(raw)
        choice = parse_choice_reply(raw, len(names))
        if choice == "none":
            return {**row, "choice": "none", "attempts": attempt, "finish": fin, **fall}
        if choice is not None:
            c = candidates[ord(choice) - 97]
            return {**row, "choice": choice, "name": c["name"], "passed": True, "name_source": c["from"],
                    "attempts": attempt, "finish": fin}
    return {**row, "choice": None, "rejects": ["parse"], "attempts": 2, "finish": fin, "raw": raws, **fall}


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
    if stage == "choose":     # Addendum C: how each object got its name
        out["choices"] = {"picked": sum(r.get("choice") not in (None, "none") and not r.get("single") for r in ok),
                          "none": sum(r.get("choice") == "none" for r in ok),
                          "failed": sum(r.get("choice") is None and "parse" in r.get("rejects", []) for r in ok),
                          "single": sum(bool(r.get("single")) for r in ok),
                          "no_candidates": sum(not r.get("candidates") for r in ok)}
        out["from"] = dict(Counter(v for r in ok for v in r.get("name_source") or []))
    if stage == "check":      # route checker (spec 2026-11-16): unreadable replies and the plain yes share
        parsed = [r for r in ok if r.get("passed")]
        out["parse_failed"] = sum("parse" in r.get("rejects", []) for r in ok)
        out["overall_true_share"] = round(sum(bool(r.get("overall_true")) for r in parsed) / len(parsed), 4) \
            if parsed else 0.0
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
    event also carries its turn point, pre-turn heading and the GT end's lateral offset (d2); the ending says how
    the route ends (d2 T2)."""
    return {"pose_xz": ev["pose_xz"], "heading_xz": ev["heading_xz"], "goal_xz": ev["goal_xz"],
            "events": [_check_event(e, objects) for e in ev["events"]],
            "ending": _check_ending(ev.get("stop"), objects)}


def _check_ending(stop, objects):
    """The ending the eval checks (d2 T2): a stop in view, out of sight behind X (X's image outline), or a turn.
    A d3 stop in view also carries its kind ("object:next to", "place:corner") for the eval's by_stop split."""
    if stop is None or stop["type"] not in ("behind", "turn"):
        kind = stop_kind(stop)
        return {"type": "stop", "stop": kind} if kind else {"type": "stop"}
    if stop["type"] == "turn":
        return {"type": "turn"}
    return {"type": "behind", "outline": objects[stop["object"]]["polygon"]}


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
        if recheck is not None and recheck(row["description"], ev):
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


def _rewritten(old, new, d1row, named, tag="d1"):
    """(rewritten desc row, None) or (None, reason) for a route whose stop became "behind" (d2 T2) or a d3 visible
    stop (tag "d2"): same events as before, every route object named by the namer the old text used, and a rewrite
    that passes the rules."""
    if old is None or old.get("reason") is not None or d1row is None:
        return None, f"no_{tag}_text"
    if _geometry_key(old)[0] != _geometry_key(new)[0]:
        return None, "events_changed"
    names = named["cosmos"] if d1row.get("namer") == "fallback" else named["gemma"]
    if not _usable(new, names):
        return None, "unnamed"
    if tag == "d1":
        text, why = rewrite_ending(d1row["description"], new["events"], new["stop"], names, new["id"])
    else:
        text, why = rewrite_stop(d1row["description"], new["events"], new["stop"], names)
    if text is None:
        return None, why
    return {**d1row, "description": text, "rewritten_from": tag, "passed": True, "rejects": []}, None


def _seed_labels(old_events, new_events, new_objects, old_names, old_desc, tag, rewrite_stop_of):
    """Shared by seed_d2_labels / seed_d3_labels: the old name rows of objects that still exist ({"gemma": rows,
    "cosmos": rows}, error rows left out so they are retried); the old desc row of every unchanged described route
    (tagged "reused_from": tag); a rewritten old row for every route with unchanged events whose stop
    `rewrite_stop_of(stop)` selects (tagged "rewritten_from": tag); and the described routes that need composing.
    rewrite_failed counts, by reason, the selected routes that could not be rewritten."""
    ids = {o["object_id"] for o in new_objects}
    names = {k: [r for r in rows if r["id"] in ids and not r.get("error")] for k, rows in old_names.items()}
    named = {k: name_rows_split(rows)[0] for k, rows in names.items()}
    named.setdefault("gemma", {})
    named.setdefault("cosmos", {})
    old = {r["id"]: r for r in old_events}
    desc = {r["id"]: r for r in old_desc if not r.get("error")}
    reused, rewritten, changed, failed = [], [], [], Counter()
    for r in new_events:
        if r.get("reason") is not None:
            continue
        o, d = old.get(r["id"]), desc.get(r["id"])
        if o is not None and d is not None and same_description_geometry(o, r):
            reused.append({**d, "reused_from": tag})
            continue
        if rewrite_stop_of(r["stop"]):
            row, why = _rewritten(o, r, d, named, tag=tag)
            if row is not None:
                rewritten.append(row)
                continue
            failed[why] += 1
        changed.append(r)
    return {"names": names, "desc": reused + rewritten, "changed": changed, "rewrite_failed": dict(failed)}


def seed_d2_labels(old_events, new_events, new_objects, old_names, old_desc):
    """What the d2 label job can start from (spec "Seeding", Addendum T2), from the d1 files (_seed_labels): routes
    whose stop became "behind" get the d1 text with its ending rewritten (rewrite_ending)."""
    return _seed_labels(old_events, new_events, new_objects, old_names, old_desc, "d1",
                        lambda stop: stop["type"] == "behind")


def seed_d3_labels(old_events, new_events, new_objects, old_names, old_desc):
    """What the d3 label job can start from (spec 2026-11-10 "Seeding"), from the d2 files (_seed_labels): hidden
    behind and turn endings are reused; a route with a d3 visible stop gets the d2 text with its final stop clause
    rewritten to the d3 stop (rewrite_stop)."""
    return _seed_labels(old_events, new_events, new_objects, old_names, old_desc, "d2",
                        lambda stop: stop_kind(stop) is not None)


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
