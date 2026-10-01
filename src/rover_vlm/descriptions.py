"""Route descriptions (prompt d1): turning geometry's events into checked sentences.

rover_vlm.route_events finds what a route passes and where it stops. This module holds the
canonical template sentence, the text rules every description must pass (all names present
in route order, each pass with its own side word, the stop phrase present), the Cosmos
prompts for the two labelling stages (name one object on a crop; compose the sentence from
text only) and the finalize helpers that turn labels into training/eval records.

Pure and CPU-only. Spec: docs/superpowers/specs/2026-09-30-route-descriptions-design.md
"""

import re

from .instructions import _DRAWING, _EGOCENTRIC, head_noun

DISTANCE_WORDS = {1.0: "one metre", 1.5: "one and a half metres", 2.0: "two metres",
                  3.0: "three metres", 4.0: "four metres"}
DESC_WORDS = (5, 45)
_METRE = r"met(?:re|er)s?"
_STOP = re.compile(r"\b(?:stop|stops|stopping|halt|park|come to rest|end up)\b", re.I)
_AFTER = re.compile(r"\b(?:just|right) (?:after|past|beyond)\b", re.I)
_SIDE = re.compile(r"\b(left|right)\b", re.I)
_BREAK = re.compile(r"[,;.]|\bthen\b|\band\b", re.I)
GENERIC_NAMES = {"object", "thing", "item", "area", "region", "stuff", "something", "space", "structure"}


def event_phrase(ev, names):
    if ev["type"] == "pass":
        return f"go {ev['side']} of the {names[0]}"
    if ev["type"] == "between":
        return f"go between the {names[0]} and the {names[1]}"
    return f"go through the {names[0]}"


def stop_text(stop, name):
    if stop["type"] == "near":
        return f"stop {stop['relation']} the {name}"
    if stop["type"] == "after_last":
        return "stop just after it"
    return f"stop about {DISTANCE_WORDS[stop['distance_m']]} {stop['relation']} the {name}"


def template_description(events, stop, names):
    """The canonical sentence for a route; always passes description_rejects."""
    parts = [event_phrase(e, [names[o] for o in e["objects"]]) for e in events]
    body = ", then ".join(parts) if parts else "go straight ahead"
    text = f"{body}, and {stop_text(stop, names.get(stop['object']))}."
    return text[0].upper() + text[1:]


def _find_name(text, name, start):
    noun = head_noun(name)
    if not noun:
        return None
    return re.compile(rf"\b{re.escape(noun)}(?:e?s)?\b", re.I).search(text, start)


def _side_rejects(text, events, spans):
    """Each side word belongs to the next name in its clause, else to the previous name."""
    got = {}
    for m in _SIDE.finditer(text):
        after = [s for s in spans if s[0] >= m.end()]
        nxt = min(after, key=lambda s: s[0]) if after else None
        if nxt is not None and not _BREAK.search(text[m.end():nxt[0]]):
            owner = nxt[2]
        else:
            before = [s for s in spans if s[1] <= m.start()]
            owner = max(before, key=lambda s: s[1])[2] if before else None
        got.setdefault(owner, set()).add(m.group(1).lower())
    out = []
    for i, e in enumerate(events):
        words = got.get(i, set())
        if e["type"] == "pass":
            if e["side"] not in words:
                out.append(f"side:missing:{i}")
            if words - {e["side"]}:
                out.append(f"side:opposite:{i}")
        elif words:
            out.append(f"side:stray:{i}")
    if got.get("stop") or got.get(None):
        out.append("side:stray")
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
    if _EGOCENTRIC.search(text):
        out.append("side:egocentric")
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
    if stop["object"]:
        m = _find_name(text, names[stop["object"]], pos)
        if m is None:
            out.append("stop:name")
        else:
            spans.append((m.start(), m.end(), "stop"))
    if not _STOP.search(text):
        out.append("stop:missing")
    if stop["type"] == "after_last" and not _AFTER.search(text):
        out.append("stop:after")
    if stop["type"] == "distance":
        number = DISTANCE_WORDS[stop["distance_m"]].rsplit(" ", 1)[0]
        if not re.search(rf"\b{number}\s+{_METRE}\b", text, re.I):
            out.append("stop:distance")
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
    if re.search(r"\d", name):
        out.append("name:digits")
    if (need == "opening") != (kind == "opening"):
        out.append("kind:mismatch")
    return out
