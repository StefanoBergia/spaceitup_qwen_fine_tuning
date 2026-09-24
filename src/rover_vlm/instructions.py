"""Instruction-conditioned paths: the route instruction is an INPUT, the path the output.

Pure and CPU-only. See docs/superpowers/specs/2026-09-24-instruction-conditioned-paths-design.md.

The experiment only means something where one image and one pair of endpoints admit two
correct routes, so the unit of data is a *pair*: a frame with a feasible route round each
side of the obstacle, both entering the frame at (almost) the same point. Each pair yields
two records that share image and endpoints and differ only in the instruction and the path;
a model that ignores the instruction can get at most one of them right.
"""

import json
import math
from pathlib import Path

from .habitat_data import build_record, format_answer_goal, instruction_endpoints_prompt

SIDES = ("left", "right")
#: Max distance between the two routes' entry points. Beyond it the entry point handed in
#: the prompt already reveals the side (57% of two-sided frames enter 0.5-1.0 apart).
SHARED_ENTRY_TOL = 0.05
TEMPLATE_INSTRUCTIONS = {"left": "Go around to the left.", "right": "Go around to the right."}


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
    if gap > 0:
        paths["right"][0] = list(paths["left"][0])
        _set_path(recs["right"], paths["right"], texts["right"])
        recs["right"]["habitat_meta"]["start_uv_norm"] = paths["right"][0][:2]

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
