"""Instruction-conditioned paths: the route instruction is an INPUT, the path the output.

Pure and CPU-only. See docs/superpowers/specs/2026-09-24-instruction-conditioned-paths-design.md.

The experiment only means something where one image and one pair of endpoints admit two
correct routes, so the unit of data is a *pair*: a frame with a feasible route round each
side of the obstacle, both entering the frame at (almost) the same point. Each pair yields
two records that share image and endpoints and differ only in the instruction and the path;
a model that ignores the instruction can get at most one of them right.
"""

import copy
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from .habitat_choice import _draw_badge, _font, _point_at_fraction
from .eval import bend_side, frechet_distance
from .habitat_data import build_record, format_answer_goal, instruction_endpoints_prompt
from .overlay import draw_polyline
from .traces import SCENE_INTRO, SCRATCH_NOTE, answer_payload, parse_trace

SIDES = ("left", "right")
#: Max distance between the two routes' entry points. Beyond it the entry point handed in
#: the prompt already reveals the side (57% of two-sided frames enter 0.5-1.0 apart).
SHARED_ENTRY_TOL = 0.05
#: Min Fréchet distance between the two routes. Below it the pair is one route drawn twice:
#: whether a prediction lands nearer the instructed side is prediction noise (18% of pairs
#: fell under 0.05 on the first build, 3 were byte-identical), so it could only add label
#: noise to training and cap pair success in eval.
MIN_ROUTE_SEPARATION = 0.05
TEMPLATE_INSTRUCTIONS = {"left": "Go around to the left.", "right": "Go around to the right."}


# the quoted instruction inside a prompt (its own double quotes are made single on render)
_INSTR_SPAN = re.compile(r'instruction says: "[^"]*"')


def _skip(reasons, reason):
    if reasons is not None:
        reasons[reason] = reasons.get(reason, 0) + 1
    return None


def side_candidates(meta):
    """{'left': i, 'right': j}: one feasible accepted candidate per side, or None.

    The labelled candidate wins its side, so the record for that side is exactly the
    round-4 record; otherwise the first accepted candidate of the side is used.
    """
    cands = meta.get("candidates", [])
    accepted = list(meta.get("accepted", []))
    label = meta.get("label")
    order = ([label] if label in accepted else []) + [i for i in accepted if i != label]
    out = {}
    for i in order:
        if not (0 <= i < len(cands)):
            continue
        side = cands[i].get("label")
        if side in SIDES and cands[i].get("feasible") and side not in out:
            out[side] = i
    return out if len(out) == 2 else None


def _answer_path(rec):
    return json.loads(rec["conversations"][1]["value"])["path"]


def _set_path(rec, path, instruction):
    """Rewrite a record's answer and prompt from its (possibly snapped) path."""
    rec["conversations"][1]["value"] = format_answer_goal([tuple(w) for w in path])
    rec["conversations"][0]["value"] = instruction_endpoints_prompt(path[0], path[-1], instruction)


def build_pair(sample_dir, *, texts=None, meta_extra=None, reasons=None):
    """Both records of one two-sided frame, [left, right], or None (reason counted).

    `texts` maps side -> instruction; None means the template. When the two entry points
    differ by at most SHARED_ENTRY_TOL, the right route's first waypoint is snapped onto
    the left one's, so the two prompts are identical except for the instruction.
    """
    sample_dir = Path(sample_dir)
    meta = json.loads((sample_dir / "meta.json").read_text())
    sides = side_candidates(meta)
    if sides is None:
        return _skip(reasons, "no_two_sided")
    source = "template" if texts is None else "given"
    texts = texts or TEMPLATE_INSTRUCTIONS

    recs = {}
    for side in SIDES:
        rec = build_record(sample_dir, framing="endpoints", candidate=sides[side],
                           instruction=texts[side], meta_extra=meta_extra)
        if rec is None:
            return _skip(reasons, "record_failed")
        recs[side] = rec

    paths = {s: _answer_path(r) for s, r in recs.items()}
    if math.dist(paths["left"][-1][:2], paths["right"][-1][:2]) > 1e-6:
        return _skip(reasons, "goal_differs")
    gap = math.dist(paths["left"][0][:2], paths["right"][0][:2])
    if gap > SHARED_ENTRY_TOL:
        return _skip(reasons, "entry_differs")
    xy = {s: [w[:2] for w in p] for s, p in paths.items()}
    if frechet_distance(xy["left"], xy["right"]) < MIN_ROUTE_SEPARATION:
        return _skip(reasons, "routes_too_similar")
    # Compare the whole [x, y, v] start, not just xy: identical coordinates with a different
    # visibility flag would still tell the prompt's reader which side it is on.
    if paths["right"][0] != paths["left"][0]:
        paths["right"][0] = list(paths["left"][0])
        _set_path(recs["right"], paths["right"], texts["right"])
        recs["right"]["habitat_meta"]["start_uv_norm"] = paths["right"][0][:2]
    masked = {s: _INSTR_SPAN.sub("", recs[s]["conversations"][0]["value"]) for s in SIDES}
    assert masked["left"] == masked["right"], f"{sample_dir.name}: prompts differ beyond the instruction"

    out = []
    for side, other in (("left", "right"), ("right", "left")):
        rec = recs[side]
        rec["id"] = f"{sample_dir.name}_{side[0].upper()}"
        rec["habitat_meta"].update({
            "pair_id": sample_dir.name,
            "side": side,
            "alt_path": paths[other],
            "entry_gap": round(gap, 4),
            "instruction_source": source,
        })
        out.append(rec)
    return out


# --- Cosmos labelling ----------------------------------------------------------------
# Bump when a prompt or overlay changes, so label files from different revisions coexist.
#
# i1 asked Cosmos to work out which way the magenta route goes and had a second Cosmos call
# verify the instruction against two identically drawn routes. The 100-pair gate showed
# Cosmos3-Nano cannot judge left vs right on drawn routes: given the *template* instruction
# ("Go around to the left.") the verifier picked the right route only 63% of the time
# (pair survival 37%), and 48% of i1 instructions named no side while only 67% of the rest
# named the correct one. The side is known exactly from the metadata, so asking Cosmos for
# it bought nothing but noise.
# i2 hands the side over and asks Cosmos only for the landmark -- the same "justify the known
# answer" strategy as the reasoning traces -- and verification becomes a text check
# (instruction_rejects with side=...): correct side word present, opposite absent, something
# named, no drawing references. One Cosmos call per record instead of two.
# i3 (2026-09-25, v2 review): for the ~23k single-route records the grey "alternative" was
# usually a same-side variant or the `direct` line, yet i2 told Cosmos it goes to the other
# side. i3 draws a grey route only when a real opposite-side candidate exists and otherwise
# shows the target alone, with the grey sentence dropped from the prompt. Pair frames get
# exactly the i2 text.
INSTRUCTION_PROMPT_VERSION = "i3"
RENDER_PX = 768
# Magenta is rare indoors, so the drawn route is never mistaken for an object; grey keeps
# the alternative visible without competing. The check image draws both routes alike.
TARGET_COLOR = (255, 0, 200)
OTHER_COLOR = (175, 175, 175)
CHECK_COLOR = (255, 215, 0)


def check_numbering(record_id):
    """Which badge (1 or 2) the target route gets in the check image, fixed per record id."""
    target = 1 + hashlib.sha256(record_id.encode()).digest()[0] % 2
    return {"target": target, "other": 3 - target}


def _canvas(record):
    img = Image.open(record["image"][0]).convert("RGB").resize((RENDER_PX, RENDER_PX), Image.LANCZOS)
    return img, ImageDraw.Draw(img)


def _xy(path):
    return [(x, y) for x, y, _ in path]


def render_generation_jpeg(record):
    """Cosmos's generation overlay as JPEG bytes: the record's route bold magenta and, when
    the record has one, the alternative thin grey. Rendered in memory so the ~28k v2 requests
    need no overlay files on NFS."""
    img, draw = _canvas(record)
    W, H = img.size
    alt = record["habitat_meta"].get("alt_path")
    if alt:
        draw_polyline(draw, _xy(alt), OTHER_COLOR, W, H, width=3)
    draw_polyline(draw, _xy(_answer_path(record)), TARGET_COLOR, W, H, width=7)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def render_generation_image(record, out_path):
    """render_generation_jpeg written to out_path (v1 kept its overlays on disk)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(render_generation_jpeg(record))


def render_check_image(record, out_path):
    """Both routes in one colour with badges 1/2 at mid-route. Returns the target's badge."""
    img, draw = _canvas(record)
    W, H = img.size
    num = check_numbering(record["id"])
    routes = {"target": _xy(_answer_path(record)), "other": _xy(record["habitat_meta"]["alt_path"])}
    for pts in routes.values():
        draw_polyline(draw, pts, CHECK_COLOR, W, H, width=5)
    font, r = _font(max(14, int(W * 0.028))), max(12, int(W * 0.022))
    for key, pts in routes.items():
        x, y = _point_at_fraction(pts, 0.5)
        _draw_badge(draw, (x * W, y * H), str(num[key]), CHECK_COLOR, font, r)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return num["target"]


def _answer_block(shape):
    return (
        f"{SCRATCH_NOTE}\n\n"
        "Reply in exactly this format and nothing else:\n"
        "<think>\n(your private working — discarded)\n</think>\n\n"
        f"<answer>\n{shape}\n</answer>\n\n"
        "The <answer> block must hold exactly one JSON object and nothing else — no code "
        "fence, no sentences outside the braces."
    )


def generation_prompt(side, has_alt=True):
    """Ask Cosmos for an instruction taking the magenta route, which goes to `side`.

    has_alt=False (no opposite-side candidate drawn): the grey-route sentence is left out,
    so the prompt never describes a route that is not on the image.
    """
    other = "right" if side == "left" else "left"
    if has_alt:
        scene = f"""Two possible routes to the same goal are drawn on the image. The rover should take the
bold magenta route, which goes to the rover's {side} (the {side} side of the image) around
whatever lies between the two routes; the thin grey route goes to the {other} instead."""
    else:
        scene = f"""One route to the goal is drawn on the image in bold magenta. The rover should take it; it
goes to the rover's {side} (the {side} side of the image) around whatever is in the way."""
    return f"""{SCENE_INTRO}

{scene}

Write the instruction a person standing behind the rover would say, so that a driver who
sees only the camera image, with nothing drawn on it, would take the {side} route.

Rules for the instruction:
- One sentence of 6 to 25 words, as a command ("go ...", "pass ...", "keep ...").
- It must contain the word '{side}' and never the word '{other}'.
- Name the thing in the scene that the route passes — a piece of furniture, a wall, a
  doorway, an opening — and say that the route passes it on the {side}. Name only
  things that are really in this image.
- Never mention the drawing: no lines, colours of lines, magenta, grey, drawn routes,
  numbers, highlights or coordinates. The driver will not see any of them.

{_answer_block('{"instruction": "<the instruction>"}')}"""


def check_prompt(instruction):
    """Ask Cosmos which of two identically drawn routes an instruction describes."""
    text = " ".join(str(instruction).split()).replace('"', "'")
    return f"""{SCENE_INTRO}

Two possible routes to the same goal are drawn on the image in the same colour, each marked
with a numbered badge, 1 and 2.

Someone gave the rover this instruction: "{text}"

Which route does the instruction describe? If neither fits perfectly, pick the one it fits
better.

{_answer_block('{"choice": <1 or 2>}')}"""


def parse_instruction(raw):
    """The instruction string from a Cosmos reply, or None.

    A reply that drops the JSON envelope but leaves one short sentence in <answer> is
    salvaged: the sentence is the deliverable.
    """
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is not None:
        text = payload.get("instruction")
        return " ".join(text.split()) if isinstance(text, str) and text.strip() else None
    if answer and "{" not in answer and len(answer.split()) <= 40:
        return " ".join(answer.strip().strip('"').split()) or None
    return None


def parse_choice(raw):
    """1 or 2 from a Cosmos check reply, or None."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is not None:
        c = payload.get("choice")
        # Cosmos often quotes the number ({"choice": "1"}) -- seen on every row of the first gate
        if isinstance(c, str) and c.strip() in ("1", "2"):
            c = int(c.strip())
        return c if c in (1, 2) and not isinstance(c, bool) else None
    found = set(re.findall(r"\b([12])\b", answer or ""))
    return int(found.pop()) if len(found) == 1 else None


# References to the overlay itself: the driver at inference sees a bare frame, so an
# instruction that leans on the drawing teaches Qwen to look for something that is not there.
_DRAWING = re.compile(
    r"\b(?:lines?|stroke|overlay|drawn|drawing|highlight\w*|marked|arrows?|badges?|magenta|"
    r"numbered|coordinates?|options?|outlin\w*|contour|dimmed|cyan|(?:route|path) (?:1|2|one|two)|"
    r"(?:bold|thin|grey|gray) (?:route|path))\b",
    re.I,
)
# v2 (2026-09-25): minimum 6 -> 4 words. 77 of v1's 91 train rejects were short, correct
# instructions such as "Go left around the sofa."; v1's finalized data is unaffected.
_WORD_LIMITS = (4, 25)


def instruction_rejects(text, side=None):
    """Why an instruction is unusable; [] when it is fine.

    With `side` (prompt i2+) the text is also the verifier: it must say that side, must not
    say the other one, and must name something beyond the side (see is_side_only).
    """
    if not text:
        return ["empty"]
    out = [f"drawing:{m.group(0).lower()}" for m in _DRAWING.finditer(text)]
    if re.search(r"\d", text):
        out.append("digits")
    n = len(text.split())
    if not (_WORD_LIMITS[0] <= n <= _WORD_LIMITS[1]):
        out.append(f"length:{n}")
    if side is not None:
        other = "right" if side == "left" else "left"
        if not re.search(rf"\b{side}\b", text, re.I):
            out.append("side:missing")
        if re.search(rf"\b{other}\b", text, re.I):
            out.append("side:opposite")
        if is_side_only(text):
            out.append("no_landmark")
        if _EGOCENTRIC.search(text):
            out.append("side:egocentric")
    return out


# "Pass the bed on your right" places the landmark on that side, so the route goes the
# OTHER way round — the reverse of our convention (the side word is the route's side).
# ~4% of the v2 labels were phrased so (2026-09-28); dropped rather than flipped.
_EGOCENTRIC = re.compile(r"\b(?:on|to) your (?:left|right)\b", re.I)


_SIDE_FILLER = set(
    "go keep stay turn veer head move drive around to the left right of on your side and "
    "then a it obstacle obstacles pass take route way bear towards toward forward ahead "
    "straight past through along".split()
)


def is_side_only(text):
    """True when the instruction says nothing beyond a side — no better than the template."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    return not [w for w in words if w not in _SIDE_FILLER]


# --- label bookkeeping ----------------------------------------------------------------

def _by_pair(items, key):
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    return groups


def summarize_labels(rows):
    """The numbers the gate and the run summary are judged on.

    A pair counts towards the total even when one of its requests errored (error rows keep
    their pair_id): an unlabelled pair is a pair not kept, and hiding it would flatter the
    survival rate. Single-route rows (pair_id None, v2) count only towards accept_rate.
    """
    ok = [r for r in rows if not r.get("error")]
    checked = [r for r in ok if r.get("check_correct") is not None]
    pair_ids = {r["pair_id"] for r in rows if r.get("pair_id")}
    kept = {pid for pid, rs in _by_pair([r for r in ok if r.get("pair_id")], lambda r: r["pair_id"]).items()
            if len(rs) == 2 and all(r.get("passed") for r in rs)}
    return {
        "n_records": len(rows),
        "n_instruction": sum(bool(r.get("instruction")) for r in ok),
        "n_drawing_refs": sum(any(x.startswith("drawing") for x in r.get("rejects", [])) for r in ok),
        "n_rejected": sum(bool(r.get("rejects")) for r in ok),
        "n_checked": len(checked),
        "check_accuracy": (round(sum(bool(r["check_correct"]) for r in checked) / len(checked), 4)
                           if checked else None),
        "n_side_only": sum(bool(r.get("side_only")) for r in ok),
        "accept_rate": round(sum(bool(r.get("passed")) for r in ok) / len(ok), 4) if ok else 0.0,
        "pairs_total": len(pair_ids),
        "pairs_kept": len(kept),
        "pair_survival": round(len(kept) / len(pair_ids), 4) if pair_ids else 0.0,
    }


def gate_failures(summary, *, min_pair_survival, max_drawing_refs, min_accept=None):
    """[] when the labelling round is good enough to scale up."""
    out = []
    if min_accept is not None and summary["accept_rate"] < min_accept:
        out.append(f"accept rate {summary['accept_rate']:.3f} < {min_accept}")
    if summary["pair_survival"] < min_pair_survival:
        out.append(f"pair survival {summary['pair_survival']:.3f} < {min_pair_survival}")
    if summary["n_drawing_refs"] > max_drawing_refs:
        out.append(f"drawing references {summary['n_drawing_refs']} > {max_drawing_refs}")
    return out


def with_instruction(rec, text, source):
    """A copy of a pair record whose prompt carries `text` as its instruction."""
    new = copy.deepcopy(rec)
    path = _answer_path(new)
    new["conversations"][0]["value"] = instruction_endpoints_prompt(path[0], path[-1], text)
    new["habitat_meta"]["instruction"] = text
    new["habitat_meta"]["instruction_source"] = source
    return new


def finalize_pairs(records, rows_by_id):
    """Pairs whose two records both passed labelling, as matched template/Cosmos lists.

    The two lists hold the same ids in the same order and differ only in the instruction,
    which is what makes template-vs-Cosmos a controlled comparison.
    """
    template, cosmos = [], []
    groups = _by_pair(records, lambda r: r["habitat_meta"]["pair_id"])
    for pid, recs in groups.items():
        assert len(recs) == 2, f"pair {pid} has {len(recs)} records"
        rows = [rows_by_id.get(r["id"]) for r in recs]
        if not all(row and row.get("passed") for row in rows):
            continue
        for rec, row in zip(recs, rows):
            side = rec["habitat_meta"]["side"]
            template.append(with_instruction(rec, TEMPLATE_INSTRUCTIONS[side], "template"))
            cosmos.append(with_instruction(rec, row["instruction"], "cosmos"))
    return template, cosmos, {"pairs_in": len(groups), "pairs_kept": len(template) // 2}


def assert_scene_disjoint(train, eval_):
    """Fatal if any scene appears in both splits (round 4's split guarantees it; re-checked)."""
    shared = ({r["habitat_meta"]["scene_id"] for r in train}
              & {r["habitat_meta"]["scene_id"] for r in eval_})
    assert not shared, f"{len(shared)} scenes in both splits, e.g. {sorted(shared)[:3]}"


def rescore_row(row):
    """A label row re-judged from its saved raw verify reply, with the current parser.

    Lets a parser fix be applied to a finished run without paying for the Cosmos calls
    again. Rows without a saved reply (passed rows, errors) are returned unchanged.
    """
    if row.get("error"):
        return row
    if row.get("prompt_version") == "i4":
        if row.get("reply") is None and any(r.startswith("no_target") for r in row.get("rejects", [])):
            return row
        new = dict(row)
        new.update(resolve_i4(row.get("reply"), row["side"], row["template"]))
        return new
    if row.get("prompt_version", "i1") != "i1" and row.get("instruction"):
        # i2+: no Cosmos verify call; the text rules are the verifier
        new = dict(row)
        new["rejects"] = instruction_rejects(row["instruction"], side=row["side"])
        new["passed"] = not new["rejects"]
        return new
    if row.get("raw_check") is None:
        return row
    new = dict(row)
    new["check_choice"] = parse_choice(row["raw_check"])
    new["check_correct"] = None if new["check_choice"] is None else new["check_choice"] == row["check_expected"]
    new["passed"] = bool(row.get("instruction") and not row.get("rejects") and new["check_correct"])
    return new


# --- i4 (2026-09-28): geometry picks the target, Cosmos only names it ------------------
# The v2 human audit found i3's landmarks ~20% wrong (75% on pairs' second route): Cosmos
# named real objects but not the one the route goes around. i4 shows Cosmos only the target
# that rover_vlm.landmarks chose — outlined, rest dimmed, no route — and hands it the side
# and a sentence pattern. The name is Cosmos's; the side and the referent are geometry's.

TARGET_OUTLINE_COLOR = (0, 230, 255)
TARGET_TEMPLATES = {
    "object": [
        "Go {side} around the {name}.",
        "Pass the {name} on the {side}.",
        "Take the {side} way around the {name}.",
        "Keep {side} of the {name}.",
        "Go around the {name} on the {side} side.",
        "Steer {side} around the {name}.",
        "Pass {side} of the {name}.",
        "Head {side} around the {name}.",
    ],
    "opening": [
        "Go through the {name} on the {side}.",
        "Take the {name} on the {side}.",
        "Head {side} through the {name}.",
        "Use the {name} on the {side}.",
        "Go {side} through the {name}.",
        "Go through the {name} to the {side} of the {beside}.",
        "Take the {name} {side} of the {beside}.",
        "Head for the {name} on the {side}, by the {beside}.",
    ],
}


def _stable_index(key, n):
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n


def choose_template(record_id, target, has_beside):
    """A sentence pattern for the record, fixed by its id (reruns agree)."""
    opts = [t for t in TARGET_TEMPLATES[target] if has_beside or "{beside}" not in t]
    return opts[_stable_index(record_id, len(opts))]


def fill_template(template, side, name, beside):
    return template.format(side=side, name=name, beside=beside or "")


def head_noun(name):
    words = re.findall(r"[a-z]+", (name or "").lower())
    return words[-1] if words else None


def render_target_jpeg(record, polygon):
    """The frame with the target outlined and everything else dimmed; no route drawn."""
    img, _ = _canvas(record)
    W, H = img.size
    pts = [(u * W, v * H) for u, v in polygon]
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    dim = Image.eval(img, lambda p: int(p * 0.45))
    img = Image.composite(img, dim, mask)
    ImageDraw.Draw(img).polygon(pts, outline=TARGET_OUTLINE_COLOR, width=6)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def naming_prompt(target, side):
    """Ask Cosmos to name the outlined target and phrase the instruction (prompt i4)."""
    what = ("an opening the rover drives through — a doorway, a passage, a hallway or a gap "
            "between two things" if target == "opening" else "the object the rover drives around")
    pattern = TARGET_TEMPLATES[target][0].format(side=side, name="<name>", beside="<beside>")
    beside = ('"beside": "<the most prominent object right next to the opening, or null>", '
              if target == "opening" else '"beside": null, ')
    return f"""{SCENE_INTRO}

One region of the image is outlined; everything outside it is dimmed. The outlined region is
{what}. The rover will pass it on its {side}.

1. Name what is inside the outline in 1 to 5 words, the way a person would say it
   ("grey sofa", "open doorway", "kitchen island"). Never use the words left or right in
   the name.
2. Say whether the outline covers one single thing (true) or several unrelated things (false).
3. Write one instruction a person standing behind the rover would say, in your own words,
   meaning the same as: "{pattern}". It must contain the word '{side}' and the name, and must
   not mention the outline, the dimming, colours of the drawing, or "your left/right".

{_answer_block('{"name": "<name>", ' + beside + '"single": true, "instruction": "<the instruction>"}')}"""


def parse_target_reply(raw):
    """{"name", "beside", "single", "instruction"} from a Cosmos i4 reply, or None."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is None or not isinstance(payload.get("name"), str) or not payload["name"].strip():
        return None
    single = payload.get("single", True)
    if isinstance(single, str):
        single = single.strip().lower() not in ("false", "no", "0")
    beside = payload.get("beside")
    beside = " ".join(beside.lower().split()) if isinstance(beside, str) and beside.strip() else None
    text = payload.get("instruction")
    return {"name": " ".join(payload["name"].lower().split()), "beside": beside,
            "single": bool(single),
            "instruction": " ".join(text.split()) if isinstance(text, str) and text.strip() else None}


def _i4_rejects(text, side, name):
    out = instruction_rejects(text, side=side)
    noun = head_noun(name)
    if text and noun and not re.search(rf"\b{re.escape(noun)}(?:e?s)?\b", text, re.I):
        out.append("name:missing")
    return out


def resolve_i4(reply, side, template):
    """The instruction for a record from its i4 reply: Cosmos's paraphrase when it passes the
    rules, else the filled template (which must pass them too)."""
    if reply is None:
        return {"instruction": None, "instruction_source": None, "rejects": ["parse"], "passed": False}
    if not reply["single"]:
        return {"instruction": None, "instruction_source": None, "rejects": ["mask_ambiguous"], "passed": False}
    if len(reply["name"].split()) > 5 or _DRAWING.search(reply["name"]):
        return {"instruction": None, "instruction_source": None, "rejects": ["name:bad"], "passed": False}
    para = reply.get("instruction")
    if para and not _i4_rejects(para, side, reply["name"]):
        return {"instruction": para, "instruction_source": "cosmos", "rejects": [], "passed": True}
    filled = fill_template(template, side, reply["name"], reply.get("beside"))
    rejects = _i4_rejects(filled, side, reply["name"])
    return {"instruction": filled if not rejects else None, "instruction_source": "template",
            "rejects": rejects, "passed": not rejects}


# --- v2: every round-4 frame -----------------------------------------------------------

def route_side(meta):
    """Side label ("left" / "right" / "direct") of the labelled candidate, or None."""
    cands, label = meta.get("candidates", []), meta.get("label")
    return cands[label].get("label") if label is not None and 0 <= label < len(cands) else None


def other_side_candidate(meta, side):
    """Best candidate on the opposite side (feasible first), or None.

    Only a genuinely opposite-side route may be drawn as Cosmos's grey alternative: same-side
    variants and the `direct` line would contradict the prompt's "goes to the other side"
    (v2 review I1 — they were ~86% of single-route alternatives).
    """
    cands, label = meta.get("candidates", []), meta.get("label")
    other = "right" if side == "left" else "left"
    opposite = [i for i, c in enumerate(cands) if i != label and c.get("label") == other]
    opposite.sort(key=lambda i: not cands[i].get("feasible"))
    return opposite[0] if opposite else None


def _candidate_path(sample_dir, index):
    rec = build_record(sample_dir, framing="endpoints", candidate=index)
    return _answer_path(rec) if rec is not None else None


def _labelled(rec, side, kind, alt_path, sample_name, suffix=None):
    rec["id"] = f"{sample_name}_{suffix or side[0].upper()}"
    rec["habitat_meta"].update({"side": side, "kind": kind, "alt_path": alt_path,
                                "instruction_source": "template"})
    return rec


def build_frame(source_rec, *, meta_extra=None, reasons=None):
    """Plain copy + the records to label for one round-4 frame.

    Pair frames (build_pair accepts them) give both sides; the correct side is kind
    "instructed", the other "opposite". Single-route frames give one "instructed" record
    whose alt_path is the best other-side candidate (only for Cosmos's overlay; no pair_id,
    so no pair metric sees it). `direct` frames give the plain copy only.
    """
    sample_dir = Path(source_rec["image"][0]).parent
    meta = json.loads((sample_dir / "meta.json").read_text())
    out = {"plain": copy.deepcopy(source_rec), "labelset": []}
    side = route_side(meta)
    if side not in SIDES:
        _skip(reasons, "direct")
        return out
    pair = build_pair(sample_dir, meta_extra=meta_extra, reasons=reasons)
    if pair is not None:
        for rec in pair:
            rec["habitat_meta"]["kind"] = "instructed" if rec["habitat_meta"]["side"] == side else "opposite"
        out["labelset"] = pair
        return out
    rec = build_record(sample_dir, framing="endpoints", instruction=TEMPLATE_INSTRUCTIONS[side],
                       meta_extra=meta_extra)
    if rec is None:
        _skip(reasons, "record_failed")
        return out
    alt_idx = other_side_candidate(meta, side)
    alt = _candidate_path(sample_dir, alt_idx) if alt_idx is not None else None
    out["labelset"] = [_labelled(rec, side, "instructed", alt, sample_dir.name)]
    return out


def build_wrong_side(source_rec, *, meta_extra=None):
    """Probe record: a single-route frame told to go to the side where no route exists.

    The candidate generator almost never draws a route on the blocked side (the other
    candidates of a single-route frame are same-side variants or `direct`; an infeasible
    opposite-side route exists in ~1 of 200 frames), so the probe keeps the feasible route
    as ground truth and the eval reads which way the prediction bends
    (eval.wrong_side_metrics). Template wording: Cosmos cannot describe an undrawn route.
    None for pair frames, `direct` frames and frames whose record fails to build.
    """
    sample_dir = Path(source_rec["image"][0]).parent
    meta = json.loads((sample_dir / "meta.json").read_text())
    side = route_side(meta)
    if side not in SIDES or side_candidates(meta) is not None:
        return None
    other = "right" if side == "left" else "left"
    rec = build_record(sample_dir, framing="endpoints", instruction=TEMPLATE_INSTRUCTIONS[other],
                       meta_extra=meta_extra)
    # the probe reads which way the prediction bends, so its ground truth must itself bend
    # clearly to the feasible side (v2 review I2: 89 of the first 300 ran "straight")
    if rec is None or bend_side(_answer_path(rec)) != side:
        return None
    rec = _labelled(rec, other, "wrong_side", None, sample_dir.name, suffix="W")
    rec["habitat_meta"]["feasible_side"] = side
    return rec


def retry_ids(rows):
    """Rows worth one more Cosmos attempt: ran, were rejected, and were not retried yet."""
    return {r["id"] for r in rows if not r.get("error") and not r.get("passed") and not r.get("retried")}


def finalize_general(plain_recs, label_recs, rows_by_id):
    """Assemble v2 files from plain copies, labelled records and their label rows.

    A labelled record is kept when its row passed; a pair frame is kept only when both of
    its records passed (otherwise both go, the plain copy stays). Kept records carry the
    Cosmos instruction in their prompt (with_instruction).
    """
    passed = {i for i, r in rows_by_id.items() if r.get("passed")}
    missing = sum(r["id"] not in rows_by_id for r in label_recs)
    rejected = sum(r["id"] in rows_by_id and r["id"] not in passed for r in label_recs)
    groups = _by_pair([r for r in label_recs if r["habitat_meta"].get("pair_id")],
                      lambda r: r["habitat_meta"]["pair_id"])
    bad_pairs = {pid for pid, rs in groups.items() if len(rs) != 2 or not all(r["id"] in passed for r in rs)}
    kept = [with_instruction(r, rows_by_id[r["id"]]["instruction"], "cosmos") for r in label_recs
            if r["id"] in passed and r["habitat_meta"].get("pair_id") not in bad_pairs]

    def kind(k):
        return [r for r in kept if r["habitat_meta"]["kind"] == k]

    return {
        "train_or_all": ([copy.deepcopy(p) for p in plain_recs]
                         + [r for r in kept if r["habitat_meta"]["kind"] != "wrong_side"]),
        "instructed": kind("instructed"),
        "pairs": [r for r in kept if r["habitat_meta"].get("pair_id")],
        "wrong_side": kind("wrong_side"),
        "stats": {"labelled": len(label_recs), "missing": missing, "rejected": rejected, "kept": len(kept),
                  "pairs_kept": len(groups) - len(bad_pairs), "pairs_dropped": len(bad_pairs)},
    }
