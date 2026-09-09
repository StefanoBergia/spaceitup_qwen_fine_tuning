"""Cheap, CPU-only metrics for the *reasoning trace itself* (not the waypoint answer).

Two kinds, because "reasoning accuracy" is not one thing:

1. Distillation fidelity — how close the student's generated reasoning is to the Cosmos3
   teacher trace for the SAME eval frame (ROUGE-L / token-F1). This is what the field calls
   a reference-based similarity metric. Caveats worth stating wherever it is shown: the
   teacher trace is a model generation (not verified truth) AND it was written with the
   ground-truth answer in view, while the student reasons without it — so a high score means
   "reasons like the teacher", never "is correct", and perfect similarity is not attainable.

2. Occlusion grounding — a correctness signal that needs NO reference: does the trace's
   hidden/visible language agree with the ground-truth goal-visibility flag? This tests the
   one spatial claim the rover task is really about. It is a LEXICAL heuristic (negation is
   hard in text), so treat it as approximate — the VLM judge (scripts/judge_traces.py) is the
   trustworthy version.

Lexical/embedding overlap is a weak proxy for reasoning quality in general; these numbers are
the cheap floor, not the verdict. Everything here is pure Python + regex, no model, no network.
"""

import re

_WORD = re.compile(r"[a-z0-9]+")

# Occlusion vs clear-view vocabulary. Net (occ - clear) decides the trace's stance, so a
# frame that says "the floor is unobstructed but the goal is hidden behind the cabinet"
# still reads as claiming occlusion. Deliberately conservative; see module docstring.
_OCCLUDED = re.compile(
    r"\b(?:hidden|out of sight|not (?:yet )?visible|no longer visible|occlud\w+|obscured|"
    r"concealed|blocked from view|behind (?:the|a|an|it|that|another|some)\b|"
    r"around the corner|can(?:not|'t) (?:see|be seen))",
    re.I)
_CLEAR = re.compile(
    r"\b(?:unobstructed|not obstructed|no obstacle\w*|clear (?:line of sight|view|path|of)|"
    r"fully visible|in (?:plain |full )?view|directly visible|visible and unobstructed)",
    re.I)


def tokens(text):
    return _WORD.findall((text or "").lower())


def _lcs_len(a, b):
    """Length of the longest common subsequence, O(len(a)*len(b)) with a rolling row."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l(hyp, ref):
    """ROUGE-L precision/recall/F1 on the LCS of whitespace tokens. F1 is the headline."""
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    lcs = _lcs_len(h, r)
    p, rec = lcs / len(h), lcs / len(r)
    f1 = 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)
    return {"p": p, "r": rec, "f1": f1}


def claims_occlusion(trace):
    """Trace's net stance: True if it asserts something is occluded more than it asserts a
    clear view. Lexical heuristic (see module docstring) — approximate by design."""
    t = trace or ""
    return len(_OCCLUDED.findall(t)) > len(_CLEAR.findall(t))


def goal_occluded(gt):
    """Ground-truth: is the goal obstructed? Visibility flag 0 = obstructed, 1 = visible."""
    return bool(gt and gt.get("goal") and gt["goal"][2] == 0)


def aggregate_trace_quality(records):
    """records: list of dicts with keys `rouge_f1` (float) and, when a GT is present,
    `occ_claim` (bool) and `gt_occ` (bool). Returns aggregate distillation + grounding stats.

    Grounding is reported as balanced accuracy plus per-class recall, because goal-occlusion
    is class-imbalanced (as with the answer-side goal-visibility metric) — raw accuracy would
    reward always saying "occluded"."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    fs = sorted(r["rouge_f1"] for r in records)
    grounded = [r for r in records if "gt_occ" in r and "occ_claim" in r]

    out = {
        "n": n,
        "rouge_l_mean": sum(fs) / n,
        "rouge_l_median": fs[n // 2],
    }
    if grounded:
        occ = [r for r in grounded if r["gt_occ"]]
        vis = [r for r in grounded if not r["gt_occ"]]
        correct = sum(1 for r in grounded if r["occ_claim"] == r["gt_occ"])
        rec_occ = (sum(1 for r in occ if r["occ_claim"]) / len(occ)) if occ else None
        rec_vis = (sum(1 for r in vis if not r["occ_claim"]) / len(vis)) if vis else None
        recalls = [x for x in (rec_occ, rec_vis) if x is not None]
        out.update({
            "occ_n": len(grounded),
            "occ_accuracy": correct / len(grounded),
            "occ_balanced": sum(recalls) / len(recalls) if recalls else None,
            "occ_recall_occluded": rec_occ, "occ_n_occluded": len(occ),
            "occ_recall_visible": rec_vis, "occ_n_visible": len(vis),
        })
    return out


def reasoning_and_answer(generated):
    """Split a `generated` string into (reasoning, answer) on the first </think>.

    The --enable-thinking prompt supplies the opening <think>, so the decoded text begins
    INSIDE the reasoning; everything up to the first </think> is the trace and the rest is
    the emitted answer. A string with no </think> (typical base-model ramble) is treated as
    all-reasoning with an empty answer. Only the FIRST </think> splits, so a stray tag in
    the answer body cannot re-split it.
    """
    marker = "</think>"
    i = generated.find(marker)
    if i == -1:
        return generated.strip(), ""
    return generated[:i].strip(), generated[i + len(marker):].strip()
