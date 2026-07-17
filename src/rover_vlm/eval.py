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

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_path_answer(text):
    """Parse a {"path":[[x,y,v],...],"goal":[x,y,v]} answer; None if unusable.

    Tries strict JSON on the first {...} block, then a regex fallback that scrapes
    [x, y, v] triples (path = all but the last, goal = the last)."""
    match = _OBJ_RE.search(text)
    if match:
        try:
            obj = _json.loads(match.group(0))
            path = [[float(a), float(b), int(round(float(c)))] for a, b, c in obj["path"]]
            g = obj["goal"]
            goal = [float(g[0]), float(g[1]), int(round(float(g[2])))]
            if len(path) >= 1:
                return {"path": path, "goal": goal}
        except (ValueError, KeyError, TypeError, IndexError):
            pass
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
