"""Route checker: an image-based check of d3 route texts for hard frames (spec
docs/superpowers/specs/2026-11-16-route-checker-design.md).

A vision-language model (Qwen3.6 via vLLM, `scripts/label_descriptions.py --stage check`) sees the frame as the
user saw it on the label audits (route in magenta, numbered outlines) plus the text, and answers only what the map
cannot see: do the names fit real things, does the route pass a big obstacle the text leaves out, is the stop phrase
natural. It never judges left/right (round i1: Cosmos-Nano was at chance on drawn-route sides; sides come exactly
from the metadata). The answers become a "probably wrong" score fitted on the user's audit marks; only d3 TRAIN
routes are ever dropped.
"""

import json
import re

from rover_vlm.traces import answer_payload, parse_trace

CHECK_PROMPT_VERSION = "k1"
_YN = {"yes": True, "true": True, "no": False, "false": False}
_FIT = ("yes", "no", "unsure")

_PROMPT_K1 = """You are checking a navigation instruction for a small ground rover against a picture of what the rover's \
camera sees.

The picture:
- The magenta line is the route the rover drives. It is solid where you can see it and dashed where it goes out of \
sight; a ring marks where it leaves the view.
- Numbered outlines mark objects the instruction may mention: the object the rover stops at is outlined in green, \
the objects it passes are outlined in orange.

The instruction:
"{text}"

Which side the rover passes things on (left or right) is already correct, taken from the map: do not judge left or \
right. Judge only these:
1. For every thing the instruction names: does the name fit what is in the picture ("fits"), and is it a real \
separate thing rather than part of a wall, floor or ceiling ("is_object")?
2. Does the route pass close to a big obstacle (a pillar, a column, a large piece of furniture) that the instruction \
does not mention ("missing_obstacle")?
3. Is the way the instruction says where the rover stops a natural way to describe where the route ends \
("stop_ok")? A person might rather say e.g. "in the middle of the room" or "in front of the sofa"; if so, write \
that in "better_stop".
4. Overall, would a person who follows the instruction end where the route ends ("overall_true"), and how sure are \
you, 0 to 100 ("confidence")?

Reply with only this JSON:
{{"names": [{{"name": "<as written in the instruction>", "fits": "yes|no|unsure", "is_object": "yes|no|unsure"}}], \
"missing_obstacle": "yes|no", "stop_ok": "yes|no", "better_stop": "<short phrase or empty>", \
"overall_true": "yes|no", "confidence": <0-100>}}"""


def check_prompt(text: str) -> str:
    """The k1 checker prompt for one instruction text."""
    return _PROMPT_K1.format(text=text)


def _yes_no(v):
    if isinstance(v, bool):
        return v
    return _YN.get(v.strip().lower()) if isinstance(v, str) else None


def _confidence(v):
    if v is None:
        return 50
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            return 50
        v = m.group(0)
    try:
        return int(round(min(100.0, max(0.0, float(v)))))
    except (TypeError, ValueError):
        return 50


def parse_check_reply(raw: str) -> dict | None:
    """The checker's JSON as a normalised dict, or None when it is absent or lacks a yes/no for missing_obstacle,
    stop_ok or overall_true. Tolerates a <think> block, a code fence and prose around the JSON."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is None:
        return None
    out = {}
    for k in ("missing_obstacle", "stop_ok", "overall_true"):
        out[k] = _yes_no(payload.get(k))
        if out[k] is None:
            return None
    names = []
    for n in payload.get("names") or []:
        if not isinstance(n, dict):
            continue
        fits, obj = (str(n.get(k, "unsure")).strip().lower() for k in ("fits", "is_object"))
        names.append({"name": str(n.get("name", "")), "fits": fits if fits in _FIT else "unsure",
                      "is_object": obj if obj in _FIT else "unsure"})
    better = payload.get("better_stop")
    return {"names": names, "missing_obstacle": out["missing_obstacle"], "stop_ok": out["stop_ok"],
            "better_stop": better.strip() if isinstance(better, str) else "",
            "overall_true": out["overall_true"], "confidence": _confidence(payload.get("confidence"))}


def check_route(ask, item: dict, seed: int) -> dict:
    """Check one manifest row ({id, image, text}) with up to two calls (seed, seed + 1). Two unreadable replies give a
    row with rejects ["parse"] (a result, not an error, so --resume does not retry it forever). Never raises on a bad
    reply; transport errors propagate to the caller."""
    prompt = check_prompt(item["text"])
    row = {"id": item["id"], "prompt_version": CHECK_PROMPT_VERSION, "rejects": [], "attempts": 0, "finish": None}
    raws, fin = [], None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(item["image"], prompt, s)
        raws.append(raw)
        parsed = parse_check_reply(raw)
        if parsed is not None:
            return {**row, "attempts": attempt, "finish": fin, "passed": True, **parsed}
    return {**row, "attempts": 2, "finish": fin, "passed": False, "rejects": ["parse"], "raw": raws}
