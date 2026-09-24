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
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from .habitat_choice import _draw_badge, _font, _point_at_fraction
from .eval import frechet_distance
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
INSTRUCTION_PROMPT_VERSION = "i1"
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


def render_generation_image(record, out_path):
    """Frame with the record's route bold magenta and the alternative thin grey."""
    img, draw = _canvas(record)
    W, H = img.size
    draw_polyline(draw, _xy(record["habitat_meta"]["alt_path"]), OTHER_COLOR, W, H, width=3)
    draw_polyline(draw, _xy(_answer_path(record)), TARGET_COLOR, W, H, width=7)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)


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


def generation_prompt():
    """Ask Cosmos for the instruction that singles out the magenta route."""
    return f"""{SCENE_INTRO}

Two possible routes to the same goal are drawn on the image. The bold magenta route is the
one the rover should drive; the thin grey route is the alternative it must not take.

Write the instruction a person standing behind the rover would say, so that a driver who
sees only the camera image, with nothing drawn on it, would take the magenta route and not
the grey one.

Rules for the instruction:
- One sentence of 6 to 25 words, as a command ("go ...", "pass ...", "keep ...").
- Refer to things that are really in the scene — furniture, walls, doorways, openings,
  floor areas. Name at least one of them and say on which side of it the route passes.
- Never mention the drawing: no lines, line colours, drawn routes, numbers, highlights or
  coordinates. The driver will not see any of them.
- Never name an object that is not in this image.

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
        return c if c in (1, 2) else None
    found = set(re.findall(r"\b([12])\b", answer or ""))
    return int(found.pop()) if len(found) == 1 else None


# References to the overlay itself: the driver at inference sees a bare frame, so an
# instruction that leans on the drawing teaches Qwen to look for something that is not there.
_DRAWING = re.compile(
    r"\b(?:lines?|stroke|overlay|drawn|drawing|highlight\w*|marked|arrows?|badges?|magenta|"
    r"numbered|coordinates?|options?|(?:route|path) (?:1|2|one|two)|"
    r"(?:bold|thin|grey|gray) (?:route|path))\b",
    re.I,
)
_WORD_LIMITS = (6, 25)


def instruction_rejects(text):
    """Why an instruction is unusable; [] when it is fine."""
    if not text:
        return ["empty"]
    out = [f"drawing:{m.group(0).lower()}" for m in _DRAWING.finditer(text)]
    if re.search(r"\d", text):
        out.append("digits")
    n = len(text.split())
    if not (_WORD_LIMITS[0] <= n <= _WORD_LIMITS[1]):
        out.append(f"length:{n}")
    return out


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

    A pair counts towards the total even when one of its requests errored (error rows
    carry no pair_id, so it is recovered from the record id): an unlabelled pair is a
    pair not kept, and hiding it would flatter the survival rate.
    """
    ok = [r for r in rows if not r.get("error")]
    checked = [r for r in ok if r.get("check_correct") is not None]
    pair_ids = {r.get("pair_id") or r["id"].rsplit("_", 1)[0] for r in rows}
    kept = {pid for pid, rs in _by_pair(ok, lambda r: r["pair_id"]).items()
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
        "pairs_total": len(pair_ids),
        "pairs_kept": len(kept),
        "pair_survival": round(len(kept) / len(pair_ids), 4) if pair_ids else 0.0,
    }


def gate_failures(summary, *, min_pair_survival, max_drawing_refs):
    """[] when the labelling round is good enough to scale up."""
    out = []
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
