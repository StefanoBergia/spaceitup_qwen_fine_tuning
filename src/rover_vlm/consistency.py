"""Sampling-consistency metrics: how much a model's answer for ONE image changes across
independent sampled draws (different seeds, temperature > 0).

Input is the set of `predictions.json` records that scripts/evaluate.py writes under
`<tag>/seeds/seed<k>/` — one dict per seed, keyed by sample id — optionally plus the greedy
run at `<tag>/`. Spread is measured on the same 10-point arc-length resampling as the
waypoint-error metric (rover_vlm.eval.habitat_metrics), so "spread" and "error" share a
scale and can be compared directly: a model whose draws spread by 0.1 is as uncertain as a
model that is 0.1 off the truth.
"""

from itertools import combinations

import numpy as np

from rover_vlm.eval import resample_polyline


def _points(pred):
    pts = pred["path"] if pred["path"] else [pred["goal"]]
    return np.array([[p[0], p[1]] for p in pts], dtype=float)


def sample_spread(draws, n_resample=10):
    """Per-image spread across sampled draws. `draws` is a list of parsed predictions
    ({"path": [[x,y,v],...], "goal": [x,y,v]}) or None for an unparseable draw.

    path_spread / goal_spread: mean pairwise distance between draws (mean resampled point
    distance / goal-point distance). goal_vis_agreement: fraction of parsed draws agreeing
    with the majority visibility flag. All None when fewer than two draws parsed."""
    parsed = [d for d in draws if d is not None]
    out = {"n_draws": len(draws), "n_parsed": len(parsed),
           "path_spread": None, "goal_spread": None, "goal_vis_agreement": None}
    if len(parsed) < 2:
        return out
    paths = [resample_polyline(_points(d), n_resample) for d in parsed]
    goals = [np.array(d["goal"][:2], dtype=float) for d in parsed]
    pairs = list(combinations(range(len(parsed)), 2))
    out["path_spread"] = float(np.mean(
        [np.linalg.norm(paths[i] - paths[j], axis=1).mean() for i, j in pairs]))
    out["goal_spread"] = float(np.mean([np.linalg.norm(goals[i] - goals[j]) for i, j in pairs]))
    flags = [int(d["goal"][2]) for d in parsed]
    out["goal_vis_agreement"] = max(flags.count(0), flags.count(1)) / len(flags)
    return out


def per_image_consistency(seed_preds, greedy_preds=None, n_resample=10):
    """Merge per-seed prediction maps ({id: record}) into {id: spread + error summary}.

    err_mean / err_std summarise mean_point_error vs ground truth over the parsed draws
    (err_std needs at least two);
    greedy_err is the greedy run's error for the same image (None if absent/unparsed)."""
    ids = sorted(set().union(*[set(p) for p in seed_preds]))
    out = {}
    for i in ids:
        recs = [p.get(i) for p in seed_preds]
        draws = [r["parsed"] if r else None for r in recs]
        row = sample_spread(draws, n_resample)
        errs = [r["metrics"]["mean_point_error"] for r in recs if r and r.get("metrics")]
        row["err_mean"] = float(np.mean(errs)) if errs else None
        row["err_std"] = float(np.std(errs)) if len(errs) >= 2 else None
        g = greedy_preds.get(i) if greedy_preds else None
        row["greedy_err"] = g["metrics"]["mean_point_error"] if g and g.get("metrics") else None
        out[i] = row
    return out


def _ranks(values):
    v = np.asarray(values, dtype=float)
    order = v.argsort()
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    # average ranks for ties
    for val in np.unique(v):
        mask = v == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x, y):
    """Spearman rank correlation; None when undefined (n < 2 or a constant input)."""
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def aggregate_consistency(per_image):
    """Dataset-level summary of per_image_consistency() output."""
    rows = list(per_image.values())
    with_spread = [r for r in rows if r["path_spread"] is not None]
    n_draws = sum(r["n_draws"] for r in rows)
    agg = {
        "num_images": len(rows),
        "num_with_spread": len(with_spread),
        "parse_rate": (sum(r["n_parsed"] for r in rows) / n_draws) if n_draws else 0.0,
    }
    for key in ("path_spread", "goal_spread"):
        vals = [r[key] for r in with_spread]
        agg[f"{key}_median"] = float(np.median(vals)) if vals else None
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else None
    agree = [r["goal_vis_agreement"] for r in with_spread]
    agg["goal_vis_agreement_mean"] = float(np.mean(agree)) if agree else None
    stds = [r["err_std"] for r in with_spread if r["err_std"] is not None]
    agg["err_std_mean"] = float(np.mean(stds)) if stds else None
    both = [(r["path_spread"], r["greedy_err"]) for r in with_spread if r["greedy_err"] is not None]
    agg["spread_vs_greedy_err_spearman"] = (
        spearman([b[0] for b in both], [b[1] for b in both]) if both else None)
    return agg
