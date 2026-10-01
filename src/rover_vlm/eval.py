"""Evaluation utilities: waypoint parsing and trajectory metrics.

All coordinates are normalized [0,1]. Predicted and ground-truth trajectories may
have different point counts, so distance metrics resample both polylines to a fixed
number of points before comparing.
"""

import ast
import re
from collections import Counter

import numpy as np

from .landmarks import CAM_HEIGHT_M, Camera, densify

# Matches the outermost [[x, y], ...] list in generated text; inner pairs may use
# [] or () (the prompt says "list of tuples", so models sometimes emit parens)
_LIST_RE = re.compile(r"\[\s*[\[(].*?[\])]\s*\]", re.DOTALL)
_PAIR_RE = re.compile(r"[\[(]\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*[\])]")


def parse_waypoints(text: str) -> list[list[float]] | None:
    """Extract a waypoint list from generated text; None if unparseable.

    Tries strict literal parsing of the first [[...]] block, then falls back to
    regex-scraping [x, y] pairs. Returns None unless >= 2 valid pairs are found.
    """
    match = _LIST_RE.search(text)
    if match:
        try:
            value = ast.literal_eval(match.group(0))
            points = [
                [float(p[0]), float(p[1])]
                for p in value
                if isinstance(p, (list, tuple)) and len(p) == 2
            ]
            if len(points) >= 2:
                return points
        except (ValueError, SyntaxError, TypeError):
            pass
    pairs = _PAIR_RE.findall(text)
    if len(pairs) >= 2:
        try:
            return [[float(x), float(y)] for x, y in pairs]
        except ValueError:
            return None
    return None


def in_unit_range(points: list[list[float]], tol: float = 0.001) -> bool:
    return all(-tol <= v <= 1 + tol for p in points for v in p)


def normalize_prediction(points: list[list[float]]) -> tuple[list[list[float]], bool]:
    """Best-effort rescale of a prediction to [0,1] coordinates.

    The base (non-fine-tuned) model tends to answer in Qwen's native grounding
    convention — integers on a 0-1000 scale — despite the prompt asking for [0,1]
    floats. To keep the zero-shot baseline fair, predictions that look per-mille
    (out of unit range but within [0, 1000]) are divided by 1000 before distance
    metrics. Returns (points, rescaled_flag); in-range predictions pass through.
    """
    if in_unit_range(points):
        return points, False
    if all(0 <= v <= 1000 for p in points for v in p):
        return [[x / 1000, y / 1000] for x, y in points], True
    return points, False


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to n points, uniformly spaced by arc length."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    resampled = np.empty((n, 2))
    for axis in range(2):
        resampled[:, axis] = np.interp(targets, cum, points[:, axis])
    return resampled


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Fréchet distance between two polylines (DP over the coupling)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = dists.shape
    dp = np.full((n, m), np.inf)
    dp[0, 0] = dists[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best_prev = min(
                dp[i - 1, j] if i > 0 else np.inf,
                dp[i, j - 1] if j > 0 else np.inf,
                dp[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            )
            dp[i, j] = max(best_prev, dists[i, j])
    return float(dp[-1, -1])


def trajectory_metrics(pred: list[list[float]], gt: list[list[float]], n_resample: int = 10) -> dict:
    """Per-sample distance metrics between predicted and ground-truth trajectories."""
    pred_rs = resample_polyline(np.asarray(pred), n_resample)
    gt_rs = resample_polyline(np.asarray(gt), n_resample)
    pointwise = np.linalg.norm(pred_rs - gt_rs, axis=1)
    return {
        "mean_point_error": float(pointwise.mean()),
        "endpoint_error": float(np.linalg.norm(np.asarray(pred[-1]) - np.asarray(gt[-1]))),
        "start_error": float(np.linalg.norm(np.asarray(pred[0]) - np.asarray(gt[0]))),
        "frechet": frechet_distance(np.asarray(pred), np.asarray(gt)),
    }


def aggregate_metrics(records: list[dict]) -> dict:
    """Aggregate per-sample eval records (as produced by scripts/evaluate.py)."""
    n = len(records)
    parsed = [r for r in records if r["parsed"] is not None]
    in_range = [r for r in parsed if in_unit_range(r["parsed"])]
    summary = {
        "num_samples": n,
        "parse_rate": len(parsed) / n if n else 0.0,
        "in_range_rate": len(in_range) / n if n else 0.0,
        "rescaled_rate": sum(1 for r in parsed if r.get("rescaled")) / n if n else 0.0,
    }
    for key in ("mean_point_error", "endpoint_error", "start_error", "frechet"):
        values = [r["metrics"][key] for r in parsed if r.get("metrics")]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
    return summary


import json as _json

# non-nested brace group: our answer object has no nested {} so this isolates it
# even amid surrounding prose or distractor brace groups
_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_path_answer(text):
    """Parse a path answer; None if unusable.

    Accepts both answer shapes: rounds 1-2 emit
    {"path":[[x,y,v],...],"goal":[x,y,v]}, round 3 (goal handed in the prompt) emits
    {"path":[[x,y,v],...]} whose final waypoint *is* the goal. When "goal" is absent
    it is taken to be path[-1], so every downstream metric keeps working unchanged and
    old prediction files re-aggregate identically.

    Scans brace groups and accepts the first that JSON-parses and carries a usable
    "path" (so triples in surrounding reasoning text are ignored). Only if no such
    object exists does it fall back to scraping [x, y, v] triples from the whole text
    (path = all but the last, goal = the last) — a best-effort net for genuinely
    malformed output."""
    for match in _OBJ_RE.finditer(text):
        try:
            obj = _json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict) or "path" not in obj:
            continue
        try:
            path = [[float(a), float(b), int(round(float(c)))] for a, b, c in obj["path"]]
            g = obj["goal"] if "goal" in obj else (path[-1] if path else None)
            goal = [float(g[0]), float(g[1]), int(round(float(g[2])))]
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        if len(path) >= 1:
            return {"path": path, "goal": goal}
    triples = re.findall(r"[\[(]\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*[\])]", text)
    if len(triples) >= 2:
        pts = [[float(a), float(b), int(round(float(c)))] for a, b, c in triples]
        return {"path": pts[:-1], "goal": pts[-1]}
    return None


def _resample_flags(wps, n):
    """Resample [x,y,v] waypoints to n points by arc length; each resampled point's
    visibility is that of the nearest original waypoint. Returns (xy[n,2], v[n])."""
    xy = np.array([[x, y] for x, y, _ in wps], dtype=float)
    vis = np.array([v for _, _, v in wps], dtype=int)
    rs = resample_polyline(xy, n)
    # nearest original waypoint per resampled point
    d = np.linalg.norm(rs[:, None, :] - xy[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return rs, vis[nearest]


#: |pred goal - gt goal| under which the goal counts as reproduced (round 3 copy check).
GOAL_COPY_TOL = 0.02


def habitat_metrics(pred, gt, n_resample=10):
    """Per-sample position + visibility metrics for the path and goal.

    From round 3 the goal is handed to the model in the prompt, so `goal_point_error`,
    `goal_visibility_correct` and `goal_copy_correct` stop measuring inference and
    become a copy-fidelity check: did the model reproduce the endpoint it was given.
    They are kept (and read that way) rather than dropped, so round-1/2 numbers stay
    comparable field-for-field. `mean_point_error` and `frechet` -- the route between
    two known endpoints -- are what the round is actually scored on.
    """
    pred_pts = pred["path"] if pred["path"] else [pred["goal"]]
    gt_pts = gt["path"] if gt["path"] else [gt["goal"]]
    pred_xy, pred_v = _resample_flags(pred_pts, n_resample)
    gt_xy, gt_v = _resample_flags(gt_pts, n_resample)
    pointwise = np.linalg.norm(pred_xy - gt_xy, axis=1)
    pg, gg = np.array(pred["goal"][:2], dtype=float), np.array(gt["goal"][:2], dtype=float)
    ps, gs = np.array(pred_pts[0][:2], dtype=float), np.array(gt_pts[0][:2], dtype=float)
    return {
        "mean_point_error": float(pointwise.mean()),
        "frechet": frechet_distance(
            np.array([p[:2] for p in pred_pts], dtype=float),
            np.array([p[:2] for p in gt_pts], dtype=float),
        ),
        "path_visibility_acc": float((pred_v == gt_v).mean()),
        "goal_point_error": float(np.linalg.norm(pg - gg)),
        "goal_visibility_correct": int(pred["goal"][2] == gt["goal"][2]),
        "goal_copy_correct": int(float(np.linalg.norm(pg - gg)) <= GOAL_COPY_TOL),
        "start_point_error": float(np.linalg.norm(ps - gs)),
        "start_copy_correct": int(float(np.linalg.norm(ps - gs)) <= GOAL_COPY_TOL),
    }


def aggregate_habitat_metrics(records):
    """Aggregate per-sample habitat eval records (as written by scripts/evaluate.py).

    Records carrying habitat_meta.desc_check (route-description sets) also get a
    "description" block from aggregate_description_metrics: follow rate (every event and
    the stop), per-event-type rates, pair success, endpoint error next to the
    constant-goal baseline."""
    n = len(records)
    parsed = [r for r in records if r.get("parsed") is not None]
    summary = {
        "num_samples": n,
        "parse_rate": len(parsed) / n if n else 0.0,
    }
    for key in ("mean_point_error", "frechet", "path_visibility_acc", "goal_point_error"):
        vals = [r["metrics"][key] for r in parsed if r.get("metrics")]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_median"] = float(np.median(vals))
    goal_v = [r["metrics"]["goal_visibility_correct"] for r in parsed if r.get("metrics")]
    if goal_v:
        summary["goal_visibility_accuracy"] = float(np.mean(goal_v))
    for key, name in (("goal_copy_correct", "goal_copy_rate"),
                      ("start_copy_correct", "start_copy_rate")):
        vals = [r["metrics"][key] for r in parsed
                if r.get("metrics") and key in r["metrics"]]
        if vals:
            summary[name] = float(np.mean(vals))
    win = baseline_win_rate(records)
    if win is not None:
        summary["win_vs_to_goal"] = win
    desc = aggregate_description_metrics(records)
    if desc is not None:
        summary["description"] = desc
    return summary


def goal_visibility_confusion(records):
    """Per-class recall for the binary goal-visibility call, plus balanced accuracy.

    Raw accuracy is misleading here: the eval split is roughly 75/25 obstructed/visible,
    so a model that answers "obstructed" every time already scores ~0.75 while having
    learned nothing about the rare class. Balanced accuracy — the mean of the two
    per-class recalls — is 0.5 for that degenerate model, which is what makes the
    collapse visible.

    Returns {recall_visible, recall_obstructed, balanced_accuracy, n_visible,
    n_obstructed} over parseable predictions only.
    """
    vv = vo = oo = ov = 0
    for r in records:
        pred = r.get("parsed")
        if not pred:
            continue
        gt_v, pred_v = r["gt"]["goal"][2], pred["goal"][2]
        if gt_v == 1:
            vv += pred_v == 1
            vo += pred_v != 1
        else:
            oo += pred_v == 0
            ov += pred_v != 0
    rv = vv / (vv + vo) if vv + vo else 0.0
    ro = oo / (oo + ov) if oo + ov else 0.0
    return {
        "recall_visible": rv,
        "recall_obstructed": ro,
        "balanced_accuracy": (rv + ro) / 2,
        "n_visible": vv + vo,
        "n_obstructed": oo + ov,
    }


# --- habitat path classification ("which candidate is traversable?") -----------------

_CHOICE_INT_RE = re.compile(r"-?\d+")


def parse_choice_answer(text):
    """Parse a {"choice": N} answer to an int; None if no number is recoverable.

    Prefers a JSON object carrying a "choice" key (so a number mentioned in reasoning
    text can't hijack the answer) and only falls back to the first bare integer when no
    such object exists."""
    for match in _OBJ_RE.finditer(text):
        try:
            obj = _json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and "choice" in obj:
            try:
                return int(round(float(obj["choice"])))
            except (TypeError, ValueError):
                continue
    m = _CHOICE_INT_RE.search(text)
    return int(m.group(0)) if m else None


def choice_metrics(pred_idx, meta):
    """Per-sample classification metrics against the label + the accepted set.

    `strict_correct` scores against the single canonical `label`; `accepted_correct`
    scores against every candidate the dataset marks feasible — the honest number when
    ~46% of samples have more than one acceptable answer.
    """
    n = meta["n_candidates"]
    kinds = meta.get("kinds") or []
    valid = pred_idx is not None and 0 <= pred_idx < n
    picked_direct = bool(valid and pred_idx < len(kinds) and kinds[pred_idx] == "direct")
    return {
        "valid": int(valid),
        "strict_correct": int(valid and pred_idx == meta["label"]),
        "accepted_correct": int(valid and pred_idx in meta["accepted"]),
        "picked_direct": int(picked_direct),
    }


def _chance(meta, exclude_direct=False):
    """Random-guess accuracy for one sample: (strict, accepted).

    With exclude_direct, guessing is restricted to non-`direct` candidates — the
    baseline a model gets for free by learning only "never pick the straight line".
    """
    kinds = meta.get("kinds") or []
    pool = [i for i in range(meta["n_candidates"])
            if not (exclude_direct and i < len(kinds) and kinds[i] == "direct")]
    if not pool:
        return None
    hits = len([i for i in pool if i in meta["accepted"]])
    return (1.0 / len(pool), hits / len(pool))


def aggregate_choice_metrics(records):
    """Aggregate per-sample choice eval records (as written by scripts/evaluate.py).

    Accuracies are over ALL samples (an unparseable answer counts as wrong), so the
    headline number is never inflated by dropping failures — parse_rate reports those
    separately.
    """
    n = len(records)
    summary = {"num_samples": n}
    if not n:
        return summary

    parsed = [r for r in records if r.get("parsed") is not None]
    scored = [r for r in records if r.get("metrics")]
    summary["parse_rate"] = len(parsed) / n
    summary["valid_choice_rate"] = sum(r["metrics"]["valid"] for r in scored) / n
    summary["strict_accuracy"] = sum(r["metrics"]["strict_correct"] for r in scored) / n
    summary["accepted_accuracy"] = sum(r["metrics"]["accepted_correct"] for r in scored) / n
    summary["picked_direct_rate"] = sum(r["metrics"]["picked_direct"] for r in scored) / n

    metas = [r["gt"] for r in records if r.get("gt")]
    for suffix, excl in (("", False), ("_excluding_direct", True)):
        vals = [c for c in (_chance(m, excl) for m in metas) if c]
        if vals:
            summary[f"chance_strict{suffix}"] = float(np.mean([v[0] for v in vals]))
            summary[f"chance_accepted{suffix}"] = float(np.mean([v[1] for v in vals]))

    by_n, by_sym = {}, {}
    for r in scored:
        by_n.setdefault(r["gt"]["n_candidates"], []).append(r["metrics"]["accepted_correct"])
        by_sym.setdefault(bool(r["gt"].get("near_symmetric")), []).append(
            r["metrics"]["accepted_correct"]
        )
    summary["accepted_accuracy_by_n_candidates"] = {
        str(k): float(np.mean(v)) for k, v in sorted(by_n.items())
    }
    summary["accepted_accuracy_by_near_symmetric"] = {
        str(k): float(np.mean(v)) for k, v in sorted(by_sym.items())
    }
    return summary


# --- real-image eval: is a number good, and is the model reading the image? -----------

def _signed_lateral(wps, n=10):
    """Mean sideways offset of a path from its own first waypoint (+ = drifts right).

    A one-number summary of which way a path bends, used to correlate a prediction's
    shape against the ground truth's. Sign matters: an unsigned magnitude would score a
    left turn and a right turn as agreeing.
    """
    xy, _ = _resample_flags(wps, n)
    return float((xy[:, 0] - xy[0, 0]).mean())


def spearman(a, b):
    """Rank correlation; NaN when either side is constant (no ranks to correlate)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def shape_correlation(records):
    """Spearman(predicted path bend, ground-truth path bend) over an eval run.

    The decisive test for an out-of-domain eval: a model that reads the image bends its
    path the same way the true route bends, so rho is positive. Around zero means the
    answer is uninformed by the picture, however well-formatted it is.
    """
    pairs = [(_signed_lateral(r["parsed"]["path"]), _signed_lateral(r["gt"]["path"]))
             for r in records
             if r.get("parsed") and r["parsed"].get("path") and r["gt"].get("path")]
    if len(pairs) < 3:
        return float("nan")
    return spearman([p for p, _ in pairs], [g for _, g in pairs])


def endpoint_spread(records, decimals=3):
    """How much a model's path *terminus* varies across an eval run.

    The Habitat generator yaws the camera onto the goal, so every training label ends at
    x = 0.5 exactly. A model fine-tuned on it learns to funnel every path into the frame
    centre and can no longer express a route that ends off-axis -- `pred_end_x_sd` near
    zero while `gt_end_x_sd` is not is that collapse, and it makes the prediction
    degenerate whatever its point error says. `mode_frac`, the share of samples sitting on
    the single most common x, reads more directly than a standard deviation once the
    collapse is total (sd 0.0, mode_frac 1.0).
    """
    pairs = [(r["parsed"]["path"][-1][0], r["gt"]["path"][-1][0])
             for r in records
             if r.get("parsed") and r["parsed"].get("path") and r["gt"].get("path")]
    if not pairs:
        nan = float("nan")
        return {"n": 0, "pred_end_x_sd": nan, "pred_end_x_mode_frac": nan,
                "gt_end_x_sd": nan, "gt_end_x_mode_frac": nan}
    out = {"n": len(pairs)}
    for key, values in (("pred", [p for p, _ in pairs]), ("gt", [g for _, g in pairs])):
        counts = Counter(round(v, decimals) for v in values)
        out[f"{key}_end_x_sd"] = float(np.std(values))
        out[f"{key}_end_x_mode_frac"] = counts.most_common(1)[0][1] / len(values)
    return out


def trivial_baselines(gts, n=10):
    """Scores an image-blind predictor would get on this eval set.

    Normalized-coordinate errors have no absolute meaning, so a model's number is only
    interpretable against what costs nothing to achieve:
      * `straight`: a vertical line up the middle of the frame, ending at the set's
        median goal height -- no perception at all.
      * `constant`: the set's own mean path, i.e. the best possible image-blind answer.
      * `to_goal`: a straight line from each sample's own start to its own goal.

    `straight` and `constant` are set-level, and stop being honest floors once the goal
    is handed to the model in the prompt (round 3 onward): a blind predictor gets the
    endpoint for free, so the bar it has to clear is the straight line to that endpoint.
    `to_goal` is that bar -- per-sample, and by far the hardest of the three. A model
    that does not beat it has not used the image at all.
    """
    paths = [g["path"] or [g["goal"]] for g in gts]
    resampled = np.stack([_resample_flags(p, n)[0] for p in paths])
    mean_path = resampled.mean(axis=0)
    median_goal_y = float(np.median([g["goal"][1] for g in gts]))
    straight = np.stack([np.full(n, 0.5), np.linspace(1.0, median_goal_y, n)], axis=1)
    out = {}
    for name, ref in (("straight", straight), ("constant", mean_path)):
        errs = [float(np.linalg.norm(ref - r, axis=1).mean()) for r in resampled]
        out[name] = {"mean_point_error_median": float(np.median(errs)),
                     "mean_point_error_mean": float(np.mean(errs))}
    out["constant"]["path"] = mean_path.tolist()

    errs = [to_goal_error(gt, n) for gt in gts]
    out["to_goal"] = {"mean_point_error_median": float(np.median(errs)),
                      "mean_point_error_mean": float(np.mean(errs))}
    return out


def to_goal_error(gt, n=10):
    """What the straight line from this sample's own start to its own goal costs.

    The single definition of the `to_goal` bar, shared by trivial_baselines (which
    averages it over a set) and baseline_win_rate (which compares it per sample).
    """
    gt_xy, _ = _resample_flags(gt["path"] or [gt["goal"]], n)
    start, goal = np.array(gt_xy[0], dtype=float), np.array(gt["goal"][:2], dtype=float)
    ref = np.stack([np.linspace(start[0], goal[0], n),
                    np.linspace(start[1], goal[1], n)], axis=1)
    return float(np.linalg.norm(ref - gt_xy, axis=1).mean())


def baseline_win_rate(records, n=10):
    """Fraction of samples where the model beats *that sample's own* to_goal line.

    Read this instead of the mean once the goal is handed over. In round 3 the set mean
    (0.072) lost to the to_goal mean (0.044) while the model actually beat the line on
    53% of samples and on 63% of the routes that start where the rover is: a handful of
    catastrophic wrong-edge starts carried 59% of the total error and swamped the
    average. The mean answers "how bad is the worst case", the win rate answers "is this
    model better than drawing a straight line", and only the second is the question.

    None when no sample has both a score and a ground-truth path.
    """
    wins = [r["metrics"]["mean_point_error"] < to_goal_error(r["gt"], n)
            for r in records if r.get("metrics") and r.get("gt")]
    return float(np.mean(wins)) if wins else None


def instruction_follow(pred_path, gt_path, alt_path):
    """Is the prediction closer (Fréchet) to the instructed route than to the other side?"""
    def xy(p):
        return np.array([q[:2] for q in p], dtype=float)
    f_t = frechet_distance(xy(pred_path), xy(gt_path))
    f_a = frechet_distance(xy(pred_path), xy(alt_path))
    return {"frechet_target": f_t, "frechet_alt": f_a, "follows": int(f_t < f_a)}


def aggregate_instruction_metrics(records):
    """Instruction-following summary for records carrying habitat_meta.alt_path, else None.

    Pair success is the headline: both records of a frame share image and endpoints and
    differ only in the instruction, so a model that ignores it returns the same path twice
    and wins at most one of the two -- 0% pair success by construction. An unparseable
    prediction counts as not following; it is never dropped from the denominator.

    Pairs come only from records with a pair_id. Records with an alt_path but no pair_id
    (v2 single-route instructed records, the wrong-side probe) count towards follow_rate
    only; one whose alt_path is None (nothing to compare against) is left out.
    """
    rows = [r for r in records if (r.get("habitat_meta") or {}).get("alt_path")]
    if not rows:
        return None
    follows, by_pair, by_side = {}, {}, {}
    for r in rows:
        hm = r["habitat_meta"]
        ok = 0
        if r.get("parsed") and r["parsed"].get("path"):
            ok = instruction_follow(r["parsed"]["path"], r["gt"]["path"], hm["alt_path"])["follows"]
        follows[r["id"]] = ok
        if hm.get("pair_id"):
            by_pair.setdefault(hm["pair_id"], []).append(r)
        by_side.setdefault(hm["side"], []).append(ok)
    pairs = [p for p in by_pair.values() if len(p) == 2]
    success = [all(follows[r["id"]] for r in p) for p in pairs]
    by_source = {}
    for p, ok in zip(pairs, success):
        by_source.setdefault(p[0]["habitat_meta"].get("source") or "unknown", []).append(ok)
    return {
        "n_records": len(rows),
        "n_pairs": len(pairs),
        "follow_rate": float(np.mean(list(follows.values()))),
        "pair_success": float(np.mean(success)) if pairs else None,
        "pair_success_by_source": {k: float(np.mean(v)) for k, v in sorted(by_source.items())},
        "identical_output_rate": (float(np.mean(
            [p[0].get("generated", "").strip() == p[1].get("generated", "").strip() for p in pairs]))
            if pairs else None),
        "follow_rate_by_side": {s: float(np.mean(v)) for s, v in sorted(by_side.items())},
    }


def clustered_bootstrap_ci(values, groups, n=4000, seed=0):
    """95% bootstrap interval of mean(values), resampling whole groups (e.g. viewpoints).

    v2's eval pairs include yaw-rotated copies of the same viewpoint, which are not
    independent; resampling pairs one by one would make the interval too narrow.
    """
    rng = np.random.default_rng(seed)
    by = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(float(v))
    keys = list(by)
    sums = np.array([sum(by[k]) for k in keys])
    counts = np.array([len(by[k]) for k in keys])
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        means.append(sums[idx].sum() / counts[idx].sum())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bend_side(path, tol=0.01, n=10):
    """Which way a path bows from the straight line entry -> goal: "left", "right", "straight".

    Measured as the mean perpendicular offset of the resampled path from that line, in the
    direction of travel (image y points down, so "left" of travel is the -90° normal). On
    round-4 labels this agrees with the candidate's side label 93% of the time; the offset
    from the first waypoint alone (_signed_lateral) does not, because the goal's own x
    offset dominates it.
    """
    xy, _ = _resample_flags(path, n)
    a, b = xy[0], xy[-1]
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return "straight"
    normal = np.array([-d[1], d[0]]) / length          # points to the left of travel
    offset = float(((xy - a) @ normal).mean())
    if abs(offset) < tol:
        return "straight"
    return "left" if offset < 0 else "right"


def wrong_side_metrics(records):
    """Probe summary for records with habitat_meta.kind == "wrong_side", else None.

    Each record is a single-route frame told to go to the side where no route exists; its
    ground truth is the feasible route. obeyed = the prediction bends to the instructed
    side, kept_feasible = it bends to the feasible side. Unparseable predictions stay in the
    denominator.
    """
    rows = [r for r in records if (r.get("habitat_meta") or {}).get("kind") == "wrong_side"]
    if not rows:
        return None
    counts = {"obeyed": 0, "kept_feasible": 0, "straight": 0, "unparsed": 0}
    for r in rows:
        hm = r["habitat_meta"]
        if not (r.get("parsed") and r["parsed"].get("path")):
            counts["unparsed"] += 1
            continue
        side = bend_side(r["parsed"]["path"])
        if side == hm["side"]:
            counts["obeyed"] += 1
        elif side == hm["feasible_side"]:
            counts["kept_feasible"] += 1
        else:
            counts["straight"] += 1
    return {"n": len(rows), **{k: v / len(rows) for k, v in counts.items()}}


# --- route descriptions (prompt d1, 2026-09-30) ---------------------------------------
FOLLOW_NEAR_M = 2.0    # a prediction farther than this from an obstacle did not "pass" it
STOP_TOL_M = 0.75      # floor fallback only: predicted end within this of goal_xz = stopped right
# The stop is judged in image space when the record's GT terminus is known: back-projecting
# to the floor is unstable far away (at 5-8 m a 0.005 error in v is > 0.75 m) and ~6% of
# round-4 answers end at goal_uv_norm, which is not the projection of goal_xz. 0.03 (normalised
# image units, ~15 px at 512) is a chosen tolerance, not a fitted one.
STOP_TOL_UV = 0.03
# A pass this close is a graze: its side flips between a dense and a sparse path, so it counts
# as satisfied whichever side it lands on.
PASS_TIE_M = 0.15
MAX_FLOOR_M = 15.0


def image_to_floor(path, pose_xz, heading_xz):
    """Floor xz of each [x, y, v] waypoint below the horizon (level camera at CAM_HEIGHT_M,
    the inverse of landmarks.Camera.uv at height 0); points on or above it are dropped."""
    cam = Camera(pose_xz, heading_xz)
    out = []
    for p in path:
        x, y = float(p[0]), float(p[1])
        if y <= 0.5 + 1e-3:
            continue
        fwd = min(0.5 * CAM_HEIGHT_M / (y - 0.5), MAX_FLOOR_M)
        out.append(cam.pos + fwd * cam.h + (x - 0.5) * 2 * fwd * cam.right)
    return np.array(out, float).reshape(-1, 2)


def _pass_side(route, cells):
    """(side the route passes the cells on, closest distance)."""
    pts = densify(route, 0.05)
    cells = np.asarray(cells, float).reshape(-1, 2)
    d = np.linalg.norm(pts[:, None] - cells[None], axis=2)
    i, j = np.unravel_index(d.argmin(), d.shape)
    t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
    t /= np.linalg.norm(t) + 1e-9
    right = np.array([-t[1], t[0]])
    return ("left" if (cells[j] - pts[i]) @ right > 0 else "right"), float(d[i, j])


_OPPOSITE = {"left": "right", "right": "left"}


def _event_followed(floor, e):
    """1 if the route does what one event says; malformed or empty event data counts as 0.

    pass: every object is passed on `side` (the route's side). wall: the wall stays on the
    rover's side `side`, i.e. the route passes it on the opposite side. Both within
    FOLLOW_NEAR_M; a graze (< PASS_TIE_M) counts either way."""
    kind = e.get("type")
    if kind in ("pass", "wall"):
        obs = e.get("obstacles") or []
        want = e.get("side") if kind == "pass" else _OPPOSITE.get(e.get("side"))
        if not obs or any(not len(g) for g in obs) or (kind == "wall" and len(obs) != 1):
            return 0
        sides = [_pass_side(floor, g) for g in obs]
        return int(all(dist < PASS_TIE_M or (side == want and dist <= FOLLOW_NEAR_M) for side, dist in sides))
    if kind == "between":
        groups = e.get("obstacles") or []
    elif kind == "through":
        gap = e.get("gap_xz")
        groups = [[p] for p in gap] if gap is not None else []
    else:
        return 0
    if len(groups) < 2 or not len(groups[0]) or not len(groups[1]):
        return 0
    a, b = (_pass_side(floor, g) for g in groups[:2])
    return int(a[0] != b[0] and max(a[1], b[1]) <= FOLLOW_NEAR_M)


def description_follow(pred_path, check, gt_end_uv=None):
    """Did a predicted path do what its description says? Per event and the stop.

    With `gt_end_uv` (the record's GT terminus, image [x, y]) the stop is the path's last
    waypoint within STOP_TOL_UV of it; without it, the floor check against goal_xz."""
    floor = image_to_floor(pred_path or [], check["pose_xz"], check["heading_xz"])
    n = len(check["events"])
    if len(floor):  # consecutive duplicate waypoints give a zero tangent in _pass_side
        keep = np.r_[True, np.linalg.norm(np.diff(floor, axis=0), axis=1) > 1e-9]
        floor = floor[keep]
    if len(floor) < 2:
        return {"follows": 0, "events": [0] * n, "stop": 0}
    res = [_event_followed(floor, e) for e in check["events"]]
    if gt_end_uv is not None:
        end = np.asarray(pred_path[-1][:2], float)
        stop = int(np.linalg.norm(end - np.asarray(gt_end_uv[:2], float)) <= STOP_TOL_UV)
    else:
        stop = int(np.linalg.norm(floor[-1] - np.asarray(check["goal_xz"], float)) <= STOP_TOL_M)
    return {"follows": int(all(res) and stop), "events": res, "stop": stop}


def _gt_end_uv(rec):
    """The record's GT terminus in the image: the answer path's last waypoint, else the goal."""
    gt = rec.get("gt") or {}
    if gt.get("path"):
        return gt["path"][-1][:2]
    return gt["goal"][:2] if gt.get("goal") else None


def aggregate_description_metrics(records):
    """Summary for records carrying habitat_meta.desc_check, else None. Unparseable
    predictions count as not followed and stay in every denominator. gt_follow_rate runs
    the same check on each record's GT path: the ceiling a perfect model would reach."""
    rows = [r for r in records if (r.get("habitat_meta") or {}).get("desc_check")]
    if not rows:
        return None
    follows, stops, by_type, pairs, gt_follows = {}, [], {}, {}, []
    for r in rows:
        chk = r["habitat_meta"]["desc_check"]
        path = (r.get("parsed") or {}).get("path")
        end = _gt_end_uv(r)
        f = (description_follow(path, chk, end) if path
             else {"follows": 0, "events": [0] * len(chk["events"]), "stop": 0})
        if (r.get("gt") or {}).get("path"):
            gt_follows.append(description_follow(r["gt"]["path"], chk, end)["follows"])
        follows[r["id"]] = f["follows"]
        stops.append(f["stop"])
        for e, ok in zip(chk["events"], f["events"]):
            by_type.setdefault(e["type"], []).append(ok)
        if r["habitat_meta"].get("pair_id"):
            pairs.setdefault(r["habitat_meta"]["pair_id"], []).append(r["id"])
    full = [p for p in pairs.values() if len(p) == 2]
    goals = np.array([r["gt"]["goal"][:2] for r in rows if r.get("gt") and r.get("metrics")], float)
    ep = [r["metrics"]["goal_point_error"] for r in rows if r.get("metrics")]
    return {
        "n_records": len(rows),
        "follow_rate": float(np.mean(list(follows.values()))),
        "gt_follow_rate": float(np.mean(gt_follows)) if gt_follows else None,
        "stop_rate": float(np.mean(stops)),
        "event_rate_by_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
        "n_pairs": len(full),
        "pair_success": float(np.mean([all(follows[i] for i in p) for p in full])) if full else None,
        "endpoint_error_median": float(np.median(ep)) if ep else None,
        # every prediction ending at the set's mean goal: 41% of goals sit at x = 0.5 and goal y
        # spans 0.545-0.629, so a text-blind endpoint is close — read endpoint error against this
        "constant_goal_endpoint_error_median": (float(np.median(np.linalg.norm(goals - goals.mean(0), axis=1)))
                                                if len(goals) else None),
    }
