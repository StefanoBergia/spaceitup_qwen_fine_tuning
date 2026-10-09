"""Pick benchmark clips out of a labelled real-image sequence.

The real-image videos are 7 minutes of mostly-straight driving; the parts worth scoring a
model on are the manoeuvres. This module scores every labelled frame against a few
criteria, then groups the qualifying frames into contiguous *runs* -- a clip should be one
manoeuvre, not a scatter of isolated frames, so runs are merged across short dropouts and
must last a minimum time before they count.

Criteria (see CRITERIA):
  * `curve`       the path bends hard in image space (`eval._signed_lateral`), goal still
                  on screen and the rover driving forward rather than spinning.
  * `occluded`    the goal is hidden behind geometry. Needs a depth-based visibility test,
                  so **TUM only** -- GND carries no depth and no usable LiDAR extrinsic,
                  and its visibility flags are all-visible by construction.
  * `out_of_view` the goal projects outside the frame: drive toward a destination you
                  cannot see. Needs `allow_offscreen_goal` labelling; these frames are
                  rejected outright by the default eval-set filters.
"""

from dataclasses import dataclass, field

from rover_vlm.eval import _signed_lateral


MAX_OVERSHOOT = 1.0  # frame-widths past the edge; beyond this the goal is simply elsewhere


@dataclass(frozen=True)
class Criterion:
    name: str
    describe: str
    score: callable          # entry -> float, higher = stronger example
    eligible: callable       # entry -> bool, hard gate before scoring
    threshold: float


def _answer(entry):
    return entry["answer"]


def bend(entry):
    """Signed image-space bend of the labelled path (+ = drifts right)."""
    path = _answer(entry).get("path") or []
    return _signed_lateral(path) if len(path) >= 2 else 0.0


def _forward(entry, max_initial=30.0):
    """The rover is driving, not spinning in place."""
    return abs(entry["meta"].get("initial_angle_deg", 0.0)) <= max_initial


def _onscreen(entry):
    return not entry["meta"].get("goal_offscreen", False)


def goal_overshoot(entry):
    """How far outside the frame the goal projects, in image widths/heights (0 = on the edge).

    Ranking off-screen goals by bearing picks the useless extreme: at the 95th percentile
    the goal sits 85 deg off axis and ten frame-widths away, i.e. beside the rover, with
    almost no route left in view. Overshoot instead favours a goal *just* past the edge,
    where the visible path plainly leads off toward it -- the case worth scoring.
    """
    u, v = entry["meta"].get("goal_uv_norm", (0.5, 0.5))
    return max(0.0, -u, u - 1.0) + max(0.0, -v, v - 1.0)


CRITERIA = (
    Criterion(
        "curve", "path bends hard in image space, goal still visible on screen",
        score=lambda e: abs(bend(e)),
        eligible=lambda e: _onscreen(e) and _forward(e),
        threshold=0.06),
    Criterion(
        "occluded", "goal hidden behind geometry (depth test; TUM only)",
        score=lambda e: 1.0 + e["meta"].get("frac_hidden", 0.0),
        eligible=lambda e: (_answer(e).get("goal", [0, 0, 1])[2] == 0 and _forward(e)),
        threshold=1.0),
    Criterion(
        "out_of_view", "goal just past the frame edge, with the route still leading to it",
        score=lambda e: 1.0 / (1.0 + goal_overshoot(e)),
        eligible=lambda e: (e["meta"].get("goal_offscreen", False)
                            and goal_overshoot(e) <= MAX_OVERSHOOT and _forward(e, 60.0)),
        threshold=1.0 / (1.0 + MAX_OVERSHOOT)),
)
CRITERIA_BY_NAME = {c.name: c for c in CRITERIA}


@dataclass
class Run:
    """A contiguous stretch of qualifying frames, in *video frame* positions."""

    criterion: str
    start: int
    end: int                 # inclusive
    peak: float
    peak_index: int
    n_hits: int
    entries: list = field(default_factory=list, repr=False)

    @property
    def length(self):
        return self.end - self.start + 1


def find_runs(entries, criterion, min_frames=1, merge_gap=0, pad=0, limit=None):
    """Group frames passing `criterion` into padded, merged runs, strongest first.

    `entries` are per-frame index records in video order. A frame qualifies when the
    criterion's `eligible` gate passes and its score clears `threshold`. Runs closer than
    `merge_gap` frames are joined -- a rover that momentarily straightens mid-corner
    should still give one clip -- then each run is padded by `pad` frames on both sides so
    the clip shows the approach, and runs shorter than `min_frames` are dropped.
    """
    by_pos = {e["v"]: e for e in entries if e.get("v") is not None}
    hits = sorted(pos for pos, e in by_pos.items()
                  if criterion.eligible(e) and criterion.score(e) >= criterion.threshold)
    if not hits:
        return []
    groups, cur = [], [hits[0]]
    for pos in hits[1:]:
        if pos - cur[-1] <= merge_gap + 1:
            cur.append(pos)
        else:
            groups.append(cur)
            cur = [pos]
    groups.append(cur)

    lo_all, hi_all = min(by_pos), max(by_pos)
    runs = []
    for g in groups:
        if len(g) < min_frames:
            continue
        peak_index = max(g, key=lambda pos: criterion.score(by_pos[pos]))
        start, end = max(lo_all, g[0] - pad), min(hi_all, g[-1] + pad)
        runs.append(Run(criterion.name, start, end, criterion.score(by_pos[peak_index]),
                        peak_index, len(g),
                        [by_pos[p] for p in range(start, end + 1) if p in by_pos]))
    runs.sort(key=lambda r: (-r.peak, r.start))
    return runs[:limit] if limit else runs
