"""Paired comparison of two base models evaluated on the same held-out split.

Both Habitat sweeps score every model on identical frames, so model-vs-model
differences are *paired* and should be tested as such: McNemar's exact test for the
binary classification outcomes, a paired bootstrap for the continuous regression
errors. Comparing two independent confidence intervals instead would badly understate
the power available here, and eyeballing a 2-point gap on 500 samples says nothing at
all without a test.
"""

import json
import math
from pathlib import Path


def load_predictions(eval_dir, tag):
    path = Path(eval_dir) / tag / "predictions.json"
    if not path.exists():
        return None
    return {r["id"]: r for r in json.loads(path.read_text())}


def load_metrics(eval_dir, tag):
    path = Path(eval_dir) / tag / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def mcnemar_exact(a, b, key):
    """Two-sided exact McNemar on paired binary outcomes keyed by sample id.

    Returns (a_only_correct, b_only_correct, p). Only the discordant pairs carry
    information — samples both models get right (or both wrong) are uninformative
    about which is better, which is exactly why the paired test is sharper.
    """
    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids
                 if _ok(a[i], key) and not _ok(b[i], key))
    b_only = sum(1 for i in ids
                 if _ok(b[i], key) and not _ok(a[i], key))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(1.0, 2 * tail / 2 ** n)


def _ok(rec, key):
    m = rec.get("metrics")
    return bool(m and m.get(key))


def paired_bootstrap(a, b, key, iters=2000, seed=0):
    """Bootstrap CI for the paired mean difference (a - b) on a continuous metric.

    Uses only samples where BOTH models produced a parseable prediction, so the
    comparison isn't contaminated by one model's parse failures being scored as if
    they were accurate. Returns (mean_a, mean_b, diff, lo, hi).
    """
    ids = [i for i in sorted(set(a) & set(b))
           if a[i].get("metrics") and b[i].get("metrics")]
    if not ids:
        return None
    va = [a[i]["metrics"][key] for i in ids]
    vb = [b[i]["metrics"][key] for i in ids]
    diffs = [x - y for x, y in zip(va, vb)]
    n = len(diffs)

    rng = _Lcg(seed)
    means = []
    for _ in range(iters):
        s = sum(diffs[rng.below(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return {
        "n": n,
        "a": sum(va) / n,
        "b": sum(vb) / n,
        "diff": sum(diffs) / n,
        "lo": means[int(0.025 * iters)],
        "hi": means[int(0.975 * iters)],
    }


class _Lcg:
    """Tiny deterministic RNG so bootstrap results are reproducible without numpy."""

    def __init__(self, seed):
        self.s = (seed * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)

    def below(self, n):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
        return (self.s >> 33) % n


def significance_label(p):
    return "significant" if p < 0.05 else "not significant"
