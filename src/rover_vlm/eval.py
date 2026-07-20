"""Evaluation utilities: waypoint parsing and trajectory metrics.

All coordinates are normalized [0,1]. Predicted and ground-truth trajectories may
have different point counts, so distance metrics resample both polylines to a fixed
number of points before comparing.
"""

import ast
import re

import numpy as np

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
    """Parse a {"path":[[x,y,v],...],"goal":[x,y,v]} answer; None if unusable.

    Scans brace groups and accepts the first that JSON-parses and has both "path"
    and "goal" keys (so triples in surrounding reasoning text are ignored). Only if
    no such object exists does it fall back to scraping [x, y, v] triples from the
    whole text (path = all but the last, goal = the last) — a best-effort net for
    genuinely malformed output."""
    for match in _OBJ_RE.finditer(text):
        try:
            obj = _json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict) or "path" not in obj or "goal" not in obj:
            continue
        try:
            path = [[float(a), float(b), int(round(float(c)))] for a, b, c in obj["path"]]
            g = obj["goal"]
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


def habitat_metrics(pred, gt, n_resample=10):
    """Per-sample position + visibility metrics for the path and goal."""
    pred_pts = pred["path"] if pred["path"] else [pred["goal"]]
    gt_pts = gt["path"] if gt["path"] else [gt["goal"]]
    pred_xy, pred_v = _resample_flags(pred_pts, n_resample)
    gt_xy, gt_v = _resample_flags(gt_pts, n_resample)
    pointwise = np.linalg.norm(pred_xy - gt_xy, axis=1)
    pg, gg = np.array(pred["goal"][:2], dtype=float), np.array(gt["goal"][:2], dtype=float)
    return {
        "mean_point_error": float(pointwise.mean()),
        "frechet": frechet_distance(
            np.array([p[:2] for p in pred_pts], dtype=float),
            np.array([p[:2] for p in gt_pts], dtype=float),
        ),
        "path_visibility_acc": float((pred_v == gt_v).mean()),
        "goal_point_error": float(np.linalg.norm(pg - gg)),
        "goal_visibility_correct": int(pred["goal"][2] == gt["goal"][2]),
    }


def aggregate_habitat_metrics(records):
    """Aggregate per-sample habitat eval records (as written by scripts/evaluate.py)."""
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
    return summary


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
