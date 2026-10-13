# Turn Events and d2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route descriptions say "turn left / right (behind / round X)" when a route ends ≥ 45° off its start→goal
line; out-and-back spur routes are dropped. Then relabel only the changed routes (d2), retrain, and rebuild the
steering stop pairs with a `turn` slice.

**Architecture:** Turn and spur geometry live in `rover_vlm.route_events`: `heading_profile`, `detect_turn` and
`has_spur` are pure functions on a route polyline. `turn_anchor` works on a loaded frame. `describe_xz` emits a
`turn` event row. `rover_vlm.descriptions` phrases and checks turns, `rover_vlm.eval` checks that a predicted path
turns, and `rover_vlm.steering` adds two turn cuts per turning route. The d2 data goes in
`data/prepared_habitat_instr_v4/`. The label files are seeded with every d1 row that still holds, so the
unchanged GPU label job (`--resume`) only names new objects and composes changed routes.

**Tech Stack:** Python 3.12 (uv), numpy, PIL, pytest; SLURM sbatch for GPU steps (Cosmos3-Nano / Gemma 3 via vLLM,
Qwen3.5-2B LoRA training/eval).

**Spec:** `docs/superpowers/specs/2026-10-13-turn-events-d2-design.md` (commits a2ad36a, 9b43b2a). Read it first.

**Refinements found while prototyping this plan:** the scratchpad prototypes were run on the synthetic scenes used
below. Both refinements are added to the spec in the commit that adds this plan.
1. A single-object **pass of the turn's anchor that comes after the turn point** is dropped. Without this, the round
   scene read "turn left round the post, then go right of the post"; the turn already says how the route goes
   round it.
2. A turn counts as **followed** when the prediction ends at least `min(TURN_FOLLOW_M, TURN_FOLLOW_SHARE ×
   gt_lateral_m)` to the stated side of the pre-turn line (0.5 m, or half the GT end's own offset when that is
   smaller). The spec says only "≥ 0.5 m"; the minimum keeps a GT route that ends shortly after its turn
   following its own text (the ceiling stays 1).

## Global Constraints

- Constants (verbatim from the spec): `TURN_MIN_DEG = 45`, `HEADING_CHORD_M = 0.5`, `TURN_END_WINDOW_M = 0.5`,
  `TURN_HOLD_DEG = 15`, `TURN_PRE_M = 1.0`, `TURN_ANCHOR_NEAR_M = 1.0`, `SPUR_REVERSAL_DEG = 150`,
  `SPUR_CHECK_M = 0.6`, `TURN_FOLLOW_M = 0.5`, steering turn cuts at `start_arc_m − 0.3` and
  `end_arc_m + 1.0`.
- `EVENTS_SCHEMA = "d2-turns"`, `PROMPT_VERSION = "d4"`, new drop reason `"spur"`.
- Gates: G1 geometry audit 100 routes ≥ 90% true; G2 label gate accept ≥ 0.90, template ≤ 0.10; G3 label audit
  100 ≥ 90%; steering Gate 0 ≥ 150 pairs and ≥ 50 turn pairs; Gate 1 50 pairs (20 turn + 30 stop) ≥ 90%;
  Gate 2 composed ≥ 0.85 (unchanged).
- Never load Qwen / Cosmos / Gemma on the login node. GPU work only via `sbatch`, and **ask the user before every
  submit**. At most 3 SLURM jobs at once.
- CPU stages: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (the stages run 4 workers).
  Tests: prefix `OMP_NUM_THREADS=4`.
- Always `uv run python` / `uv run pytest`. Bare `python3` in the repo hits pyenv 3.12 and fails.
- Never `git add -A`. Stage files by name. Never stage `scripts/evaluate.py` or the user's uncommitted README row
  (README commits are index-only, see Task 10).
- d1 files are read-only: `data/prepared_habitat_instr_v2/*`, `data/prepared_habitat_instr_v3/*`,
  `outputs/instr_labels_v3/*_d1.jsonl`, `outputs/runs_desc/d1/`.
- Synthetic test maps: 5 m × 5 m at 0.05 m; camera at (2.5, 0.2) facing +z, so the camera's right is −x and
  **+x is the rover's left**.
- Branch: `feat/turn-events` (already created off local `main`).

## Review Focus

1. **"turn right after the table".** `_RIGHT_ADVERB` blanks "right" before "after / past / …" as the adverb.
   The turn rule must claim that "right" first, or a correct turn text is rejected
   (`turn:side:missing`). Test in Task 4.
2. **Pronoun anchor.** "go right of the table, then turn left behind it" must pass: the anchor is the previous
   step's object and Cosmos will often use "it". Test in Task 4.
3. **Re-running `d2-seed` after the GPU job** would overwrite hours of labels. The stage refuses existing non-empty
   d2 label files unless `--force`. Test in Task 6.
4. **A steering cut that stops inside the turn** (between `start_arc_m` and `end_arc_m`) must get no turn event; it
   has not finished turning. Test in Task 2.
5. **A prediction with no usable floor points** (all at or above the horizon) must count as not following a turn,
   never crash. Test in Task 5.

---

### Task 1: Turn and spur geometry (pure functions)

**Files:**
- Modify: `src/rover_vlm/route_events.py` (new section after `stop_phrase`, before `_none`)
- Test: `tests/test_route_turns.py` (new)

**Interfaces:**
- Consumes: `densify` (landmarks), `ROUTE_STEP_M`.
- Produces:
  - `heading_profile(route_xz, step=ROUTE_STEP_M) -> (pts: ndarray[N,2], arc: ndarray[N], theta_deg: ndarray[N],
    line: ndarray[2])`
  - `detect_turn(route_xz) -> dict | None` with keys `side` ("left"/"right"), `angle_deg`, `arc_m`,
    `start_arc_m`, `end_arc_m`, `xz` [x, z], `pre_heading_xz` [x, z] (unit)
  - `has_spur(route_xz) -> bool`
  - constants listed in Global Constraints

- [ ] **Step 1: Write the failing tests**

Create `tests/test_route_turns.py`:

```python
"""d2 turn events and spur routes (spec 2026-10-13): geometry on plain polylines and synthetic maps.

Polylines in the first part are in free metres. Synthetic maps (second part, from Task 2) are 5 m x 5 m at
5 cm with the camera at (2.5, 0.2) facing +z: the camera's right is -x, so +x is the rover's LEFT.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm.route_events import (
    HEADING_CHORD_M, SPUR_CHECK_M, SPUR_REVERSAL_DEG, TURN_MIN_DEG, detect_turn, has_spur, heading_profile,
)

L_RIGHT = [(0.0, 0.0), (0.0, 3.0), (-1.5, 3.0)]     # travelling +z, then -x: a right turn (right of +z is -x)


def test_constants():
    assert (TURN_MIN_DEG, HEADING_CHORD_M, SPUR_REVERSAL_DEG, SPUR_CHECK_M) == (45.0, 0.5, 150.0, 0.6)


def test_heading_profile_is_relative_to_the_start_goal_line():
    pts, arc, th, line = heading_profile([(0.0, 0.0), (0.0, 2.0)])
    assert arc[-1] == pytest.approx(2.0) and np.allclose(line, [0.0, 1.0])
    assert np.allclose(th, 0.0, atol=1e-6)
    _, _, th, _ = heading_profile(L_RIGHT)
    assert th[0] == pytest.approx(-26.6, abs=0.5)        # the first leg runs left of the start->goal line
    assert th[-1] == pytest.approx(63.4, abs=0.5)        # the last leg ends right of it


def test_l_route_is_one_right_turn_at_its_corner():
    t = detect_turn(L_RIGHT)
    assert t["side"] == "right" and t["angle_deg"] == pytest.approx(63.4, abs=0.5)
    assert abs(t["arc_m"] - 3.0) <= 0.15
    assert 2.6 <= t["start_arc_m"] <= t["arc_m"] <= t["end_arc_m"] <= 3.4
    assert np.allclose(t["pre_heading_xz"], [0.0, 1.0], atol=1e-3)     # it was going +z before the turn
    assert t["xz"][1] == pytest.approx(3.0, abs=0.06)


def test_a_mirrored_l_route_turns_left():
    t = detect_turn([(0.0, 0.0), (0.0, 3.0), (1.5, 3.0)])
    assert t["side"] == "left" and t["angle_deg"] < 0


def test_a_swerve_that_comes_back_to_the_line_is_not_a_turn():
    assert detect_turn([(0.0, 0.0), (0.8, 1.5), (0.0, 3.0), (0.0, 4.5)]) is None


def test_swing_one_way_then_turn_the_other():
    route = [(0.0, 0.0), (1.0, 1.5), (1.0, 3.5), (-1.0, 3.5)]
    t = detect_turn(route)
    assert t["side"] == "right" and t["angle_deg"] == pytest.approx(74.1, abs=0.5)
    assert heading_profile(route)[2].min() < -15            # it swung left of the line first


@pytest.mark.parametrize("phi", [44.0, 46.0])
def test_threshold(phi):
    # the last leg makes exactly phi with the start->goal line (+z), turning left
    route = [(0.0, 0.0), (-1.0, 3.0), (0.0, 3.0 + 1.0 / np.tan(np.radians(phi)))]
    t = detect_turn(route)
    assert (t is not None) == (phi >= 45.0)
    if t is not None:
        assert t["angle_deg"] == pytest.approx(-phi, abs=0.2) and t["side"] == "left"


def test_straight_diagonal_and_short_routes_never_turn():
    # a yaw-augmented frame sees a straight route diagonally; turns are measured on the route alone
    assert detect_turn([(0.0, 0.0), (3.0, 3.0)]) is None
    assert detect_turn([(0.0, 0.0), (0.0, 0.3)]) is None


@pytest.mark.parametrize("route,spur", [
    ([(0, 0), (0, 2), (1.5, 2), (0, 2), (0, 4)], True),            # exact retrace (P[k-1] == P[k+1])
    ([(0, 0), (0, 2), (1.5, 2), (0.05, 2.1), (0, 4)], True),       # near-mirror spike
    (L_RIGHT, False),
    ([(0.0, 0.0), (0.8, 1.5), (0.0, 3.0), (0.0, 4.5)], False),     # swerve
    ([(0, 0), (0, 2), (0.866 * 1.5, 2 - 0.75)], False),            # a sharp but real 120 deg corner
])
def test_has_spur(route, spur):
    assert has_spur(route) is spur
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_route_turns.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'HEADING_CHORD_M' from 'rover_vlm.route_events'`.

- [ ] **Step 3: Implement**

In `src/rover_vlm/route_events.py`, insert this block right after the end of `stop_phrase` (before
`def _none(`):

```python
# --- turns and spur routes (d2, docs/superpowers/specs/2026-10-13-turn-events-d2-design.md) -----------------
TURN_MIN_DEG = 45.0          # a route that ENDS this far off its start->goal line has turned
HEADING_CHORD_M = 0.5        # heading at s = direction of the chord from s - 0.25 m to s + 0.25 m
TURN_END_WINDOW_M = 0.5      # the final heading = mean heading over the last this-many metres
TURN_PRE_M = 1.0             # the pre-turn heading is read this far before the turn point
TURN_HOLD_DEG = 15.0         # turn start / end: heading back within this of the pre-turn / final heading
TURN_ANCHOR_NEAR_M = 1.0     # a visible turn is anchored to a blob this close on its inner side
SPUR_REVERSAL_DEG = 150.0    # the chord heading turning this much within SPUR_CHECK_M = an out-and-back spur
SPUR_CHECK_M = 0.6           # (normal GT routes reach ~100 deg within 0.6 m; spurs 180)


def heading_profile(route_xz, step=ROUTE_STEP_M):
    """(pts, arc, theta_deg, line) of a route: its points densified at `step` (zero-length steps dropped),
    their arclength, the heading at every point in degrees relative to the start->goal line (+ = right,
    Camera.right handedness; unwrapped) taken from a HEADING_CHORD_M chord, and that line's unit vector.
    Relative to the line, not the camera: a yaw-augmented frame sees the same route turn the same way."""
    pts = densify(np.asarray(route_xz, float), step)
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    g = pts[-1] - pts[0]
    n = float(np.linalg.norm(g))
    g = g / n if n > 1e-9 else np.array([0.0, 1.0])
    right = np.array([-g[1], g[0]])

    def at(s):
        s = np.clip(s, 0.0, arc[-1])
        return np.stack([np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], -1)

    d = at(arc + HEADING_CHORD_M / 2) - at(arc - HEADING_CHORD_M / 2)
    return pts, arc, np.degrees(np.unwrap(np.arctan2(d @ right, d @ g))), g


def detect_turn(route_xz):
    """The route's final turn (spec detector A), or None. The route has turned when its mean heading over the
    last TURN_END_WINDOW_M is at least TURN_MIN_DEG off its start->goal line; a swerve round an obstacle ends
    back on that line. The turn point is where the heading last crosses half the final angle; it starts at
    the last earlier point within TURN_HOLD_DEG of the heading TURN_PRE_M before it, and ends at the first
    later point within TURN_HOLD_DEG of the final heading. Run it on a FULL route: a cut that stops
    mid-swerve would look like a turn (steering passes the full route's turn to its cuts)."""
    pts, arc, th, g = heading_profile(route_xz)
    if arc[-1] < HEADING_CHORD_M or np.linalg.norm(pts[-1] - pts[0]) < 1e-6:
        return None
    e = float(np.mean(th[arc >= arc[-1] - TURN_END_WINDOW_M]))
    if abs(e) < TURN_MIN_DEG:
        return None
    not_over = np.nonzero(np.sign(e) * (th - e / 2) <= 0)[0]
    k = int(min(not_over[-1] + 1, len(pts) - 1)) if len(not_over) else 0
    j = int(np.searchsorted(arc, arc[k] - TURN_PRE_M))
    pre = float(th[j])
    i0 = k
    while i0 > j and abs(th[i0] - pre) > TURN_HOLD_DEG:
        i0 -= 1
    i1 = k
    while i1 < len(th) - 1 and abs(th[i1] - e) > TURN_HOLD_DEG:
        i1 += 1
    r = np.radians(pre)
    h0 = np.cos(r) * g + np.sin(r) * np.array([-g[1], g[0]])
    return {"side": "right" if e > 0 else "left", "angle_deg": round(e, 1), "arc_m": float(arc[k]),
            "start_arc_m": float(arc[i0]), "end_arc_m": float(arc[i1]),
            "xz": [float(v) for v in pts[k]], "pre_heading_xz": [float(v) for v in h0]}


def has_spur(route_xz):
    """True when the route doubles back on itself: its chord heading turns by >= SPUR_REVERSAL_DEG within
    SPUR_CHECK_M of arclength. These are the generator's via-point routes that drive out and back along one
    line (87% of d1's "opposite" routes); d2 drops them (spec "Spur routes")."""
    _, arc, th, _ = heading_profile(route_xz)
    j = np.minimum(np.searchsorted(arc, arc + SPUR_CHECK_M), len(arc) - 1)
    return bool(np.any(np.abs(th[j] - th) >= SPUR_REVERSAL_DEG))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_route_turns.py tests/test_route_events.py -q`
Expected: all pass. The prototype measured L-route arc 3.10, start 2.85, end 3.15; swing 74.1°; φ = 44 → None and
φ = 46 → −46.0.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py tests/test_route_turns.py
git commit -m "feat: route heading profile, final-turn and spur detection (d2)"
```

---

### Task 2: Turn event rows and spur drops in `describe_xz`

**Files:**
- Modify: `src/rover_vlm/route_events.py`: `first_blocker` (via a new `_first_hit`), new `turn_anchor`,
  `_anchor_outline`, `_straight_before`, `describe_route`, `describe_xz`, `REASONS`, `EVENTS_SCHEMA`,
  `check_events_schema` message, module docstring (one line on turns).
- Test: `tests/test_route_turns.py` (append), `tests/test_route_events.py` (two existing tests updated)

**Interfaces:**
- Consumes: Task 1 (`detect_turn`, `has_spur`, `TURN_ANCHOR_NEAR_M`).
- Produces:
  - `turn_anchor(frame, turn) -> {"blob": int, "relation": "behind"|"round", "vis_xz": ndarray, "outline": dict} | None`
  - `describe_xz(meta, frame, route, turn=None)`. The turn event row is `{"type": "turn", "side", "relation":
    "behind"|"round"|None, "objects": [oid] or [], "arc_m", "turn_xz": [x, z], "pre_heading_xz": [x, z],
    "gt_lateral_m": float}`, placed in arc order among the obstacle rows. Reason `"spur"` for a spur route.
  - `describe_route(sample_dir, candidate_index)` passes `turn=detect_turn(route)`.
  - `EVENTS_SCHEMA == "d2-turns"`, `"spur" in REASONS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_route_turns.py`:

```python
from rover_vlm.landmarks import FREE, OCC                                           # noqa: E402
from rover_vlm.route_events import (                                                # noqa: E402
    EVENTS_SCHEMA, REASONS, describe_route, describe_xz, first_blocker, load_frame, turn_anchor,
)


def write_sample(d, state, routes, *, pose=(2.5, 0.2), heading=(0.0, 1.0)):
    """A synthetic sample dir (as tests/test_route_events.py): state.npz + meta.json, one candidate per route."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    (d / "meta.json").write_text(json.dumps({"pose_xz": list(pose), "heading_xz": list(heading), "flavor": "detour",
                                             "chord": {"crossing_xz": None}, "candidates": cands}))
    return d


def _free():
    return np.full((100, 100), FREE, np.uint8)


def _cabinet(s):
    s[52:68, 96:99] = OCC        # x 4.8..4.95, z 2.6..3.4: the stop, 0.8 m past the routes' ends (no event)
    return s


def _round_state():              # a post on the inner side of the L_LEFT corner, in plain view
    s = _cabinet(_free())
    s[48:54, 57:61] = OCC        # x 2.85..3.05, z 2.4..2.7
    return s


def _behind_state():             # a box straight ahead; the route goes right of it and turns left behind it
    s = _cabinet(_free())
    s[32:52, 43:55] = OCC        # x 2.15..2.75, z 1.6..2.6
    return s


L_LEFT = [(2.5, 0.2), (2.5, 3.0), (4.0, 3.0)]                      # ahead, then +x: a left turn
BEHIND = [(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (3.9, 3.0)]
SWING = [(2.5, 0.2), (1.7, 1.5), (1.7, 3.0), (4.0, 3.0)]           # swings right, then turns left


def _turns(out):
    return [e for e in out["events"] if e["type"] == "turn"]


def test_hidden_turn_is_anchored_behind_the_box_the_route_went_round(tmp_path):
    sd = write_sample(tmp_path / "s", _behind_state(), [BEHIND])
    meta, frame = load_frame(sd)
    t = detect_turn(BEHIND)
    box = first_blocker(frame, t["xz"])
    assert box is not None                                            # scene check: the box hides the corner
    assert turn_anchor(frame, t)["blob"] == box and turn_anchor(frame, t)["relation"] == "behind"
    out = describe_route(sd, 0)
    assert out["reason"] is None
    kinds = [(e["type"], e["side"]) for e in out["events"]]
    assert kinds == [("pass", "right"), ("turn", "left")]
    turn = out["events"][1]
    assert turn["relation"] == "behind" and turn["objects"] == out["events"][0]["objects"]
    assert turn["gt_lateral_m"] == pytest.approx(1.9, abs=0.15)       # the end is 2 m left of the pre-turn line
    assert np.allclose(turn["pre_heading_xz"], [0.0, 1.0], atol=1e-3)
    assert out["stop"]["type"] == "near"


def test_visible_turn_is_anchored_round_the_inner_post_and_absorbs_its_pass(tmp_path):
    out = describe_route(write_sample(tmp_path / "s", _round_state(), [L_LEFT]), 0)
    assert out["reason"] is None
    assert [(e["type"], e["side"], e.get("relation")) for e in out["events"]] == [("turn", "left", "round")]
    assert out["objects"][out["events"][0]["objects"][0]]["kind_hint"] == "object"


def test_turn_with_nothing_nameable_near_it_is_bare(tmp_path):
    out = describe_route(write_sample(tmp_path / "s", _cabinet(_free()), [L_LEFT]), 0)
    assert out["reason"] is None and len(out["events"]) == 1
    assert (out["events"][0]["type"], out["events"][0]["relation"], out["events"][0]["objects"]) == ("turn", None, [])


def test_turn_alone_after_a_swing_is_still_no_events_bent(tmp_path):
    assert describe_route(write_sample(tmp_path / "s", _cabinet(_free()), [SWING]), 0)["reason"] == "no_events_bent"


@pytest.mark.parametrize("cut,has_turn", [
    ([(2.5, 0.2), (1.9, 1.2), (1.9, 2.4)], False),                    # stops before the turn starts
    ([(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (2.0, 3.0)], False),        # stops inside the turn (Review Focus 4)
    ([(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (3.2, 3.0)], True),         # has finished turning
])
def test_a_cut_keeps_the_full_routes_turn_only_once_finished(tmp_path, cut, has_turn):
    meta, frame = load_frame(write_sample(tmp_path / "s", _behind_state(), [BEHIND]))
    out = describe_xz(meta, frame, np.array(cut), turn=detect_turn(BEHIND))
    assert out["reason"] is None
    assert bool(_turns(out)) is has_turn


def test_spur_routes_are_dropped(tmp_path):
    spur = [(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (1.0, 3.0), (1.9, 3.0), (3.9, 3.0)]
    assert describe_route(write_sample(tmp_path / "s", _behind_state(), [spur]), 0)["reason"] == "spur"
    assert "spur" in REASONS and EVENTS_SCHEMA == "d2-turns"


def test_describe_route_passes_the_routes_own_turn(tmp_path):
    sd = write_sample(tmp_path / "s", _behind_state(), [BEHIND])
    meta, frame = load_frame(sd)
    assert describe_xz(meta, frame, np.array(BEHIND), turn=detect_turn(BEHIND)) == describe_route(sd, 0)
    assert not _turns(describe_xz(meta, frame, np.array(BEHIND)))       # no turn handed in -> none emitted
```

In `tests/test_route_events.py`, update the two existing tests that pin d1 behaviour:

```python
def test_events_schema_check_accepts_stamped_meta(tmp_path):
    (tmp_path / "events_meta.json").write_text(json.dumps({"schema": EVENTS_SCHEMA, "train": {}}))
    assert EVENTS_SCHEMA == "d2-turns"
```
(keep the rest of that test's body unchanged), and

```python
def test_describe_xz_matches_describe_route(tmp_path):
    sd = write_sample(tmp_path / "s", _stop_state(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    assert describe_xz(meta, frame, np.array(STOP_ROUTE), turn=detect_turn(STOP_ROUTE)) == describe_route(sd, 0)
```
and add `detect_turn` to that file's `from rover_vlm.route_events import (... )` line that brings in `describe_xz`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_route_turns.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'turn_anchor'`.

- [ ] **Step 3: Implement**

In `src/rover_vlm/route_events.py`:

(a) Replace `first_blocker` with a thin wrapper over a new `_first_hit` (the anchor needs the hit point):

```python
def _first_hit(frame, xz):
    """(blob label, hit xz) of the first occupied cell between the camera and xz, or (None, None).

    Cells within NEAR_CAMERA_M of the camera are skipped (see NEAR_CAMERA_M)."""
    for p in densify(np.array([frame.cam.pos, xz], float), frame.grid.mpp / 2)[1:]:
        if np.linalg.norm(p - frame.cam.pos) <= NEAR_CAMERA_M:
            continue
        r, c = (int(round(v)) for v in frame.grid.rc(p)[0])
        if frame.grid.cell((r, c)) == OCC:
            return int(frame.lab[r, c]), p
    return None, None


def first_blocker(frame, xz):
    """Blob label of the first occupied cell between the camera and xz, or None (see _first_hit)."""
    return _first_hit(frame, xz)[0]
```

(b) `REASONS` gains `"spur"`:

```python
REASONS = ("no_state", "bad_candidate", "spur", "no_stop_anchor", "no_events_bent", "error")
```

(c) After `has_spur` (Task 1 block) add:

```python
def _anchor_outline(frame, blob, near_xz):
    """(visible cells, outline) of the part of a blob within KEEP_NEAR_ROUTE_M of near_xz, or None when it is
    not nameable (fewer than MIN_VISIBLE_CELLS visible cells, or an outline under MIN_AREA)."""
    cells, xz = _blob_cells(frame)[blob]
    keep = np.linalg.norm(xz - np.asarray(near_xz, float), axis=1) <= KEEP_NEAR_ROUTE_M
    vis = visible_cells(frame, cells[keep], blob)
    if len(vis) < MIN_VISIBLE_CELLS:
        return None
    o = object_outline(frame, vis)
    return (vis, o) if o["area"] >= MIN_AREA else None


def turn_anchor(frame, turn):
    """The object a turn is said relative to, or None (a bare "turn left"). A hidden turn point -> the
    nameable blob hiding it, relation "behind" (its outline: the part near where the camera ray hits it).
    Else the nearest nameable blob within TURN_ANCHOR_NEAR_M of the turn point on the turn's inner side
    (the rover's left for a left turn, in the pre-turn heading's frame) -> "round"."""
    p = np.asarray(turn["xz"], float)
    occ, hit = _first_hit(frame, p)
    if occ is not None:
        got = _anchor_outline(frame, occ, hit)
        if got:
            return {"blob": occ, "relation": "behind", "vis_xz": got[0], "outline": got[1]}
    h0 = np.asarray(turn["pre_heading_xz"], float)
    inner = np.array([-h0[1], h0[0]]) * (1.0 if turn["side"] == "right" else -1.0)
    near = []
    for b, (_, xz) in _blob_cells(frame).items():
        d = np.linalg.norm(xz - p, axis=1)
        k = int(d.argmin())
        if d[k] <= TURN_ANCHOR_NEAR_M and float((xz[k] - p) @ inner) > 0:
            near.append((float(d[k]), b))
    for _, b in sorted(near):
        got = _anchor_outline(frame, b, p)
        if got:
            return {"blob": b, "relation": "round", "vis_xz": got[0], "outline": got[1]}
    return None


def _straight_before(route, arc_m):
    """True when the route up to `arc_m` stays within STRAIGHT_MAX_M of its own chord ("go ahead, then turn")."""
    rg = route_geometry(route)
    pts = rg["pts"][rg["arc"] <= max(arc_m, ROUTE_STEP_M)]
    return len(pts) < 2 or route_geometry(pts)["offset"].max() <= STRAIGHT_MAX_M
```

(d) `describe_route`: change its last line to

```python
    route = cm[candidate_index]["polyline_xz"]
    return describe_xz(meta, frame, route, turn=detect_turn(route))
```

(e) Replace `describe_xz` with:

```python
def describe_xz(meta, frame, route, turn=None):
    """The events + stop phrase for any route polyline (world xz) on a loaded frame (steering pairs describe
    routes that are not candidates of the frame). `turn` is the FULL route's turn (detect_turn): a route
    shorter than its end_arc_m (a steering cut that stops before or inside the turn) gets no turn event, and
    without `turn` none is emitted. A route that doubles back on itself is dropped as "spur"."""
    route = np.asarray(route, float)
    if has_spur(route):
        return _none("spur")
    cands = candidates(frame, route)
    events = build_events(frame, cands)
    if turn is not None and route_geometry(route)["arc"][-1] + 1e-6 < turn["end_arc_m"]:
        turn = None
    anchor = turn_anchor(frame, turn) if turn is not None else None
    if anchor is not None:          # a pass of the anchor alone, after the turn point, is what the turn says
        events = [e for e in events if not (e["type"] == "pass" and len(e["members"]) == 1
                                            and cands[e["members"][0]]["blob"] == anchor["blob"]
                                            and e["arc_m"] >= turn["arc_m"])]
    if not events:
        if turn is None and route_geometry(route)["offset"].max() > STRAIGHT_MAX_M:
            return _none("no_events_bent")                   # "go straight ahead" would be false
        if turn is not None and not _straight_before(route, turn["start_arc_m"]):
            return _none("no_events_bent")                   # an unexplained swing before the turn
    stop = stop_phrase(frame, route, events, cands)          # obstacle events only: "just after" / "past" name objects
    if stop is None:
        return _none("no_stop_anchor", n_events=len(events) + (turn is not None))
    # Object rows in explicit passes. Events first: an event member's kind_hint is its event role
    # (pass member -> "object", wall member -> "wall", opening -> "opening"). Then the turn anchor, then the stop
    # anchor: each reuses an existing object row unchanged and sets kind_hint / outline only for a new object.
    objects = {}

    def add(key, kind, outline, cells):
        oid = f"{frame.name}:{key}"
        if oid not in objects:
            objects[oid] = {"object_id": oid, "kind_hint": kind,
                            **{k: outline[k] for k in ("polygon", "bbox", "crop_box", "area", "depth_fallback")},
                            "cells_xz": _cells(cells)}
        return oid

    rows = []
    for e in events:
        blobs = [cands[m]["blob"] for m in e["members"]]
        if e["type"] == "through":
            ids = [add(_gap_key(blobs, e["opening"]["gap_xz"]), "opening", e["opening"], e["opening"]["gap_xz"])]
        else:
            ids = [add(f"b{cands[m]['blob']}", "wall" if cands[m]["wall"] else "object",
                       cands[m]["outline"], cands[m]["vis_xz"]) for m in e["members"]]
        row = {"type": e["type"], "side": e["side"], "objects": ids, "arc_m": round(e["arc_m"], 2)}
        if e["type"] == "through":
            row["gap_xz"] = e["opening"]["gap_xz"]
        rows.append(row)
    if turn is not None:
        ids = []
        if anchor is not None:
            kind = "wall" if _wall_like(_blob_cells(frame)[anchor["blob"]][1]) else "object"
            ids = [add(f"b{anchor['blob']}", kind, anchor["outline"], anchor["vis_xz"])]
        p, h0 = np.asarray(turn["xz"], float), np.asarray(turn["pre_heading_xz"], float)
        lateral = float((route[-1] - p) @ np.array([-h0[1], h0[0]]))     # how far the end is off the pre-turn line
        rows.append({"type": "turn", "side": turn["side"], "relation": anchor["relation"] if anchor else None,
                     "objects": ids, "arc_m": round(turn["arc_m"], 2),
                     "turn_xz": [round(float(v), 3) for v in p], "pre_heading_xz": [round(float(v), 4) for v in h0],
                     "gt_lateral_m": round(abs(lateral), 2)})
        rows.sort(key=lambda r: r["arc_m"])                  # stable: an obstacle event at the same arc stays first
    st = {"type": stop["type"], "relation": stop["relation"], "amount": stop.get("amount"),
          "distance_m": stop.get("distance_m"), "object": None}
    if stop["anchor"] is not None:                       # after every event row (see above)
        a = stop["anchor"]
        st["object"] = add(a.get("key", f"b{a['blob']}"), a.get("kind", "object"), a["outline"], a["vis_xz"])
    return {"reason": None, "events": rows, "stop": st, "objects": objects,
            "goal_xz": [round(float(v), 3) for v in route[-1]],
            "pose_xz": [float(v) for v in meta["pose_xz"]], "heading_xz": [float(v) for v in meta["heading_xz"]],
            "route_len_m": round(float(route_geometry(route)["arc"][-1]), 2)}
```

(f) Schema:

```python
EVENTS_SCHEMA = "d2-turns"  # stamped into events_meta.json by prepare_habitat_instr.py events
```
and in `check_events_schema` change the SystemExit text to
`f"events files are not {EVENTS_SCHEMA!r} — rerun prepare_habitat_instr.py events ({mp}: schema {schema!r})"`.

(g) Module docstring: add one line after the A2 paragraph:
`d2 (2026-10-13): a route that ends >= 45 deg off its start->goal line gets a "turn" event (behind / round an object, or bare); routes that double back on themselves are dropped as "spur".`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_route_turns.py tests/test_route_events.py -q`
Expected: all pass. If a real-frame regression test in `test_route_events.py` fails, read its assertion. A frame
whose GT route now ends ≥ 45° off its line gains a turn row; the test only looks at obstacle events, so filter
`e["type"] != "turn"` there rather than weakening the rule.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py tests/test_route_turns.py tests/test_route_events.py
git commit -m "feat: turn event rows (behind / round / bare) and spur drops in describe_xz; schema d2-turns"
```

---

### Task 3: Turn phrasing, compose prompt d4, desc_check turn fields

**Files:**
- Modify: `src/rover_vlm/descriptions.py`: `event_phrase`, `template_description`, `compose_prompt`,
  `PROMPT_VERSION`, `desc_check`, module docstring first line.
- Test: `tests/test_descriptions.py` (append)

**Interfaces:**
- Consumes: the turn event row from Task 2.
- Produces: `event_phrase(turn_event, names)` gives "turn left behind the X" / "turn right round the X" /
  "turn left". `desc_check(...)["events"][i]` carries `turn_xz`, `pre_heading_xz`, `gt_lateral_m` for a turn.
  `PROMPT_VERSION == "d4"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_descriptions.py`:

```python
from rover_vlm.descriptions import PROMPT_VERSION, compose_prompt, desc_check, event_phrase  # noqa: E402

PASS_R6 = {"type": "pass", "side": "right", "objects": ["f:b6"]}
TURN_BEHIND6 = {"type": "turn", "side": "left", "relation": "behind", "objects": ["f:b6"], "arc_m": 3.0,
                "turn_xz": [1.9, 3.0], "pre_heading_xz": [0.0, 1.0], "gt_lateral_m": 1.9}
TURN_ROUND3 = {"type": "turn", "side": "right", "relation": "round", "objects": ["f:b3"], "arc_m": 2.0,
               "turn_xz": [1.0, 2.0], "pre_heading_xz": [0.0, 1.0], "gt_lateral_m": 1.0}
TURN_BARE_L = {"type": "turn", "side": "left", "relation": None, "objects": [], "arc_m": 2.5,
               "turn_xz": [0.0, 2.5], "pre_heading_xz": [0.0, 1.0], "gt_lateral_m": 1.2}


def test_turn_phrases():
    assert event_phrase(TURN_BEHIND6, ["table"]) == "turn left behind the table"
    assert event_phrase(TURN_ROUND3, ["grey sofa"]) == "turn right round the grey sofa"
    assert event_phrase(TURN_BARE_L, []) == "turn left"


def test_turn_templates():
    assert template_description([PASS_R6, TURN_BEHIND6], NEAR, NAMES) == (
        "Go right of the table, then turn left behind the table, and stop in front of the fridge.")
    assert template_description([TURN_BARE_L], NEAR, NAMES) == "Go ahead, then turn left, and stop in front of the fridge."


def test_compose_prompt_d4_carries_the_turn_step_and_rule():
    assert PROMPT_VERSION == "d4"
    p = compose_prompt([PASS_R6, TURN_BEHIND6], NEAR, NAMES, "plain", "one sentence")
    assert "2. turn left behind the table" in p
    assert "'turn left' / 'turn right'" in p and "never add a turn" in p


def test_desc_check_keeps_the_turn_geometry():
    objects = {"f:b6": {"cells_xz": [[2.2, 2.0]]}}
    ev = {"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0], "goal_xz": [3.9, 3.0], "events": [PASS_R6, TURN_BEHIND6]}
    c = desc_check(ev, objects)
    assert c["events"][1]["turn_xz"] == [1.9, 3.0] and c["events"][1]["pre_heading_xz"] == [0.0, 1.0]
    assert c["events"][1]["gt_lateral_m"] == 1.9 and "turn_xz" not in c["events"][0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_descriptions.py -k "turn or d4" -v`
Expected: FAIL: `test_turn_phrases` (`event_phrase` returns "go through the table" for an unknown type),
`assert PROMPT_VERSION == "d4"` fails, and the template / desc_check asserts fail.

- [ ] **Step 3: Implement**

In `src/rover_vlm/descriptions.py`:

```python
def event_phrase(ev, names):
    """One step. Object passes name the ROUTE's side ("go right of the table"); walls name the
    ROVER's side the wall is on ("keep the white wall on your left"); a turn names the rover's own turn and,
    when anchored, the object it turns behind / round (d2)."""
    if ev["type"] == "pass":
        return f"go {ev['side']} of the " + " and the ".join(names)
    if ev["type"] == "wall":
        return f"keep the {names[0]} on your {ev['side']}"
    if ev["type"] == "between":
        return f"go between the {names[0]} and the {names[1]}"
    if ev["type"] == "turn":
        rel = ev.get("relation")
        return f"turn {ev['side']}" + (f" {rel} the {names[0]}" if rel and names else "")
    return f"go through the {names[0]}"
```

In `template_description`, after `body = ", then ".join(parts) if parts else "go straight ahead"` add:

```python
    if events and events[0]["type"] == "turn":
        body = "go ahead, then " + body              # the route runs straight before it turns (route_events)
```

Set `PROMPT_VERSION = "d4"`. In `compose_prompt`, replace the sentence that ends `…so do not add one. Keep where
to stop.` with:

```python
/ 'right of' stays that way; a step with 'between' or 'through' has no left or right, so do not add
one. In a 'turn left' / 'turn right' step, keep 'turn left' / 'turn right' exactly (it is the rover's own
turn) and keep its 'behind' / 'round' and its object; never add a turn the steps do not have. Keep where
to stop. You may add small words
```
(the surrounding f-string lines stay as they are; only this passage gains the turn sentence).

Replace `desc_check` with:

```python
def desc_check(ev, objects):
    """What eval.description_follow needs, carried in the record (no file access at eval time). A turn
    event also carries its turn point, pre-turn heading and the GT end's lateral offset (d2)."""
    return {"pose_xz": ev["pose_xz"], "heading_xz": ev["heading_xz"], "goal_xz": ev["goal_xz"],
            "events": [_check_event(e, objects) for e in ev["events"]]}


def _check_event(e, objects):
    out = {"type": e["type"], "side": e["side"], "obstacles": [objects[o]["cells_xz"] for o in e["objects"]],
           "gap_xz": e.get("gap_xz")}
    if e["type"] == "turn":
        out.update({k: e[k] for k in ("turn_xz", "pre_heading_xz", "gt_lateral_m")})
    return out
```

Module docstring first line becomes:
`"""Route descriptions (prompt d4 since d2; d2 / d1 before): turning geometry's events into checked sentences.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_descriptions.py -q`
Expected: all pass, except tests that assert the old `PROMPT_VERSION` value. Update any such assert to `"d4"`.
Nothing else changes.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py tests/test_descriptions.py
git commit -m "feat: turn phrases, compose prompt d4 and turn fields in desc_check"
```

---

### Task 4: Turn reject rules

**Files:**
- Modify: `src/rover_vlm/descriptions.py`: new `_TURN`, `TURN_RELATION_WORDS`, `turn_rejects`; `description_rejects`.
- Test: `tests/test_descriptions.py` (append)

**Interfaces:**
- Consumes: Task 3 phrases.
- Produces: `turn_rejects(text, events) -> list[str]`, which needs no names (finalize re-checks reused d1 texts with
  it). Reasons: `turn:stray`, `turn:side:missing:<i>`, `turn:side:opposite:<i>`, `turn:relation:<i>`.
  `description_rejects` includes them, accepts a pronoun for a turn anchor that is the previous step's object,
  and gives the turn's side word to the turn.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_descriptions.py`:

```python
from rover_vlm.descriptions import turn_rejects  # noqa: E402

EV_T = [PASS_R6, TURN_BEHIND6]


@pytest.mark.parametrize("text", [
    "Go right of the table, then turn left behind it, and stop in front of the fridge.",          # Review Focus 2
    "Pass the table on the right, turn left behind the table and stop in front of the fridge.",
    "Go right of the table, then turn to your left behind it and stop in front of the fridge.",
    "Go right of the table, then turn left behind the table, and stop in front of the fridge.",   # the turn owns 'left'
])
def test_turn_texts_that_pass(text):
    assert description_rejects(text, EV_T, NEAR, NAMES) == [], text


@pytest.mark.parametrize("text,reason", [
    ("Go right of the table, then turn behind it and stop in front of the fridge.", "turn:side:missing:1"),
    ("Go right of the table, then turn right behind it and stop in front of the fridge.", "turn:side:opposite:1"),
    ("Go right of the table, then turn left past it, and stop in front of the fridge.", "turn:relation:1"),
    ("Go right of the table, then turn left behind the sofa, and stop in front of the fridge.", "name:missing:1"),
])
def test_turn_rejections(text, reason):
    assert reason in description_rejects(text, EV_T, NEAR, NAMES), description_rejects(text, EV_T, NEAR, NAMES)


def test_a_turn_word_without_a_turn_event_is_stray():
    t = "Go left of the coffee table, then turn right and stop in front of the fridge."
    assert "turn:stray" in description_rejects(t, [PASS_L], NEAR, NAMES)
    assert turn_rejects(t, [PASS_L]) == ["turn:stray"]
    assert turn_rejects("Go left of the coffee table and stop in front of the fridge.", [PASS_L]) == []


def test_turn_right_after_is_a_turn_not_the_adverb():          # Review Focus 1
    turn_r = {**TURN_BARE_L, "side": "right"}
    t = "Go left of the coffee table, turn right after it, and stop in front of the fridge."
    assert description_rejects(t, [PASS_L, turn_r], NEAR, NAMES) == []


@pytest.mark.parametrize("events", [[PASS_R6, TURN_BEHIND6], [TURN_BARE_L], [PASS_L, TURN_ROUND3]])
def test_turn_templates_pass_their_own_rules(events):
    text = template_description(events, NEAR, NAMES)
    assert description_rejects(text, events, NEAR, NAMES) == [], text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_descriptions.py -k "turn" -v`
Expected: `ImportError: cannot import name 'turn_rejects'`.

- [ ] **Step 3: Implement**

In `src/rover_vlm/descriptions.py`, next to the other regexes:

```python
# d2: "turn left", "turns right", "turning to the left", "turn to your right". Its side word belongs to the turn
# event, never to the next name, and is claimed BEFORE _RIGHT_ADVERB blanks "right after" (Review Focus 1).
_TURN = re.compile(r"\bturn(?:s|ed|ing)?\s+(?:to\s+(?:the|your)\s+)?(left|right)\b", re.I)
TURN_RELATION_WORDS = {"behind": r"\bbehind\b", "round": r"\b(?:a)?round\b"}
```

After `_side_rejects`, add:

```python
def turn_rejects(text, events):
    """The turn rules, which need no names (finalize re-checks reused d1 texts with them): a turn phrase only
    where the events have a turn, saying the turn's side and no other, and an anchored turn's relation word
    ("behind" / "round") in the clause after the turn phrase."""
    found = list(_TURN.finditer(text))
    turns = [i for i, e in enumerate(events) if e["type"] == "turn"]
    if not turns:
        return ["turn:stray"] if found else []
    i, e = turns[0], events[turns[0]]
    sides = {m.group(1).lower() for m in found}
    out = []
    if e["side"] not in sides:
        out.append(f"turn:side:missing:{i}")
    if sides - {e["side"]}:
        out.append(f"turn:side:opposite:{i}")
    rel = e.get("relation")
    if rel and found:
        m = next((m for m in found if m.group(1).lower() == e["side"]), found[0])
        b = _BREAK.search(text, m.end())
        if not re.search(TURN_RELATION_WORDS[rel], text[m.end(): b.start() if b else len(text)], re.I):
            out.append(f"turn:relation:{i}")
    return out
```

In `description_rejects`, change the name loop to accept a pronoun for a turn anchor that is the previous step's
object:

```python
    for i, e in enumerate(events):
        for oid in e["objects"]:
            m = _find_name(text, names[oid], pos)
            if m is None and e["type"] == "turn" and i > 0 and oid in events[i - 1]["objects"]:
                m = _PRONOUN.search(text, pos)       # "...right of the table, then turn left behind it"
            if m is None:
                out.append(f"name:missing:{i}")
                continue
            spans.append((m.start(), m.end(), i))
            pos = m.end()
```

and replace its final line with:

```python
    # the turn phrase's side word is the turn's: blank it (same length, spans stay valid) before side ownership
    side_text = _TURN.sub(lambda m: " " * len(m.group(0)), text)
    return out + turn_rejects(text, events) + _side_rejects(side_text, events, spans)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_descriptions.py tests/test_steering.py -q`
Expected: all pass (no d1 test uses a turn word, so d1 behaviour is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py tests/test_descriptions.py
git commit -m "feat: turn reject rules (stray / side / relation), turn-owned side words, pronoun anchors"
```

---

### Task 5: Eval: turn followed, by-turn split, steering per slice

**Files:**
- Modify: `src/rover_vlm/eval.py`: `TURN_FOLLOW_M`, `TURN_FOLLOW_SHARE`, `_event_followed`,
  `aggregate_description_metrics`, `aggregate_steering_metrics`.
- Test: `tests/test_description_eval.py` (append)

**Interfaces:**
- Consumes: desc_check turn fields (Task 3).
- Produces: `_event_followed(floor, turn_check)` gives 1 / 0. `aggregate_description_metrics(...)["by_turn"] ==
  {"turn"|"no_turn": {"n", "follow_rate", "endpoint_error_median", "point_error_median"}}`.
  `aggregate_steering_metrics(...)["by_slice"] == {slice: {"n_pairs", "pair_success", "pair_success_ci"}}`;
  `"slice"` is the single slice, or `"all"` when there are several.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_description_eval.py`:

```python
from rover_vlm.eval import TURN_FOLLOW_M, TURN_FOLLOW_SHARE  # noqa: E402

TURN_L = {"type": "turn", "side": "left", "obstacles": [], "gap_xz": None,
          "turn_xz": [0.0, 3.0], "pre_heading_xz": [0.0, 1.0], "gt_lateral_m": 1.5}


@pytest.mark.parametrize("xz,ok", [
    ([(0.0, 0.8), (0.0, 3.0), (1.5, 3.0)], 1),       # turns left (+x is the rover's left)
    ([(0.0, 0.8), (0.0, 4.5)], 0),                    # carries straight on
    ([(0.0, 0.8), (0.0, 2.5)], 0),                    # stops short of the turn
    ([(0.0, 0.8), (0.0, 3.0), (-1.5, 3.0)], 0),       # turns the other way
])
def test_turn_followed(xz, ok):
    assert description_follow(to_image(xz), check([TURN_L], goal=xz[-1]))["events"] == [ok]


def test_turn_threshold_shrinks_with_a_short_gt_turn():
    assert (TURN_FOLLOW_M, TURN_FOLLOW_SHARE) == (0.5, 0.5)
    path = to_image([(0.0, 0.8), (0.0, 3.0), (0.3, 3.0)])                   # 0.3 m left of the pre-turn line
    assert description_follow(path, check([TURN_L], goal=(0.3, 3.0)))["events"] == [0]
    short = {**TURN_L, "gt_lateral_m": 0.4}                                  # the GT itself ends 0.4 m off
    assert description_follow(path, check([short], goal=(0.3, 3.0)))["events"] == [1]


def test_turn_with_no_floor_points_is_not_followed():           # Review Focus 5
    assert description_follow([[0.5, 0.4, 1], [0.6, 0.45, 1]], check([TURN_L]))["events"] == [0]


def test_by_turn_split():
    def rec(rid, events, path_xz):
        p = to_image(path_xz)
        return {"id": rid, "parsed": {"path": p}, "gt": {"path": p, "goal": p[-1]},
                "metrics": {"goal_point_error": 0.1 if events else 0.02, "mean_point_error": 0.05 if events else 0.01},
                "habitat_meta": {"desc_check": check(events, goal=path_xz[-1]), "pair_id": None}}
    recs = [rec("t", [TURN_L], [(0.0, 0.8), (0.0, 3.0), (1.5, 3.0)]), rec("s", [], [(0.0, 0.8), (0.0, 4.0)])]
    m = aggregate_description_metrics(recs)
    assert m["by_turn"]["turn"] == {"n": 1, "follow_rate": 1.0, "endpoint_error_median": 0.1, "point_error_median": 0.05}
    assert m["by_turn"]["no_turn"]["n"] == 1 and m["event_rate_by_type"]["turn"] == 1.0


def test_steering_by_slice():
    recs = _pair(A_GT, B_GT, pid="p") + _pair(B_GT, B_GT, pid="q")
    for r in recs[2:]:
        r["habitat_meta"]["steer"]["slice"] = "turn"
    m = aggregate_steering_metrics(recs)
    assert m["slice"] == "all" and m["by_slice"]["stop"]["pair_success"] == 1.0
    assert m["by_slice"]["turn"]["n_pairs"] == 1 and m["by_slice"]["turn"]["pair_success"] == 0.0
    assert aggregate_steering_metrics(_pair(A_GT, B_GT))["slice"] == "stop"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_description_eval.py -v`
Expected: `ImportError: cannot import name 'TURN_FOLLOW_M'`.

- [ ] **Step 3: Implement**

In `src/rover_vlm/eval.py`, after `PASS_TIE_M`:

```python
# d2 turn check: the prediction ends at least this far to the turn's side of the GT pre-turn line (the line
# through the turn point along the heading before it) -- or, when the GT route itself ends closer than twice
# that, half the GT end's own offset, so the GT always follows its own text (the ceiling stays 1).
TURN_FOLLOW_M = 0.5
TURN_FOLLOW_SHARE = 0.5
```

At the top of `_event_followed`, before `if kind in ("pass", "wall"):`, insert:

```python
    if kind == "turn":
        p, h = e.get("turn_xz"), e.get("pre_heading_xz")
        if p is None or h is None or e.get("side") not in _OPPOSITE or not len(floor):
            return 0
        h = np.asarray(h, float)
        h = h / (np.linalg.norm(h) + 1e-9)
        lateral = float((floor[-1] - np.asarray(p, float)) @ np.array([-h[1], h[0]]))   # + = right of the line
        gt = float(e.get("gt_lateral_m") or TURN_FOLLOW_M / TURN_FOLLOW_SHARE)
        want = min(TURN_FOLLOW_M, TURN_FOLLOW_SHARE * gt)
        return int((lateral if e["side"] == "right" else -lateral) >= want)
```
and add `turn:` to its docstring: `turn: the path's end lies TURN_FOLLOW_M (or half the GT's offset) to the turn's
side of the pre-turn line.` `description_follow` already returns 0 events when the floor path has fewer than 2
points, which covers Review Focus 5.

In `aggregate_description_metrics`, collect per-record turn membership and add `by_turn` to the returned dict:

```python
    split = {"turn": [], "no_turn": []}
    for r in rows:
        chk = r["habitat_meta"]["desc_check"]
        split["turn" if any(e["type"] == "turn" for e in chk["events"]) else "no_turn"].append(r)

    def _group(rs):
        met = [x["metrics"] for x in rs if x.get("metrics")]
        return {"n": len(rs), "follow_rate": float(np.mean([follows[x["id"]] for x in rs])),
                "endpoint_error_median": float(np.median([m["goal_point_error"] for m in met])) if met else None,
                "point_error_median": float(np.median([m["mean_point_error"] for m in met])) if met else None}
```
(placed after the main loop), and in the return dict: `"by_turn": {k: _group(v) for k, v in split.items() if v},`.

In `aggregate_steering_metrics`, after the pairs loop compute:

```python
    slices = [a["habitat_meta"]["steer"]["slice"] for a, _ in pairs]
    by_slice = {}
    for s in sorted(set(slices)):
        idx = [i for i, x in enumerate(slices) if x == s]
        lo_s, hi_s = clustered_bootstrap_ci([succ[i] for i in idx], [groups[i] for i in idx])
        by_slice[s] = {"n_pairs": len(idx), "pair_success": float(np.mean([succ[i] for i in idx])),
                       "pair_success_ci": [lo_s, hi_s]}
```
and in the returned dict replace the `"slice"` entry with `"slice": slices[0] if len(by_slice) == 1 else "all",`
and add `"by_slice": by_slice,`. Update the docstring: `by_slice: pair success per steer.slice ("stop" / "turn").`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_description_eval.py tests/test_steering.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/eval.py tests/test_description_eval.py
git commit -m "feat: eval turn check (end past the pre-turn line), by_turn split, steering by_slice"
```

---

### Task 6: d2 stages: events from the v2 labelsets, `d2-seed`, `finalize-descriptions --round d2`

**Files:**
- Modify: `src/rover_vlm/descriptions.py`: new `same_description_geometry`, `seed_d2_labels`;
  `finalize_descriptions` gains `recheck=None`.
- Modify: `scripts/prepare_habitat_instr.py`: `V4_OUT`; `stage_events` (`--labelset-dir`); new `stage_d2_seed`;
  `desc_path(..., round_="d1")`; `stage_finalize_descriptions` (`--round`, `--labelset-dir`, defaults per round);
  docstring section "d2".
- Test: `tests/test_d2_stages.py` (new)

**Interfaces:**
- Consumes: Tasks 2 and 4 (`turn_rejects`), `route_objects`.
- Produces:
  - `same_description_geometry(old_row, new_row) -> bool`
  - `seed_d2_labels(old_events, new_events, new_objects, old_names: {"gemma": rows, "cosmos": rows}, old_desc)
    -> {"names": {"gemma": rows, "cosmos": rows}, "desc": rows, "changed": rows}`
  - `finalize_descriptions(label_recs, events_by_id, desc_by_id, objects, recheck=None)`. `recheck(text, events)
    -> list` (non-empty = drop, counted in `stats["recheck_failed"]`).
  - CLI:
    - `prepare_habitat_instr.py events --labelset-dir DIR --out-dir DIR`
    - `prepare_habitat_instr.py d2-seed [--force]`
    - `prepare_habitat_instr.py finalize-descriptions --round d2 --desc-suffix _mix`
  - Files:
    - `outputs/instr_labels_v3/{names_<s>_gemma_d2,names_<s>_d2,desc_<s>_mix_d2}.jsonl`
    - `data/prepared_habitat_instr_v4/{events_eval_changed,objects_eval_changed}.jsonl`
    - `data/prepared_habitat_instr_v4/seed_meta.json`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_d2_stages.py`:

```python
"""d2 stages (spec 2026-10-13 "Data flow"): events from the v2 labelsets, seeding the d2 label files from d1,
finalize --round d2 re-checking every text."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from rover_vlm.descriptions import finalize_descriptions, same_description_geometry, seed_d2_labels, turn_rejects

_SPEC = importlib.util.spec_from_file_location(
    "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

STOP = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}


def _row(rid, events, reason=None, stop=STOP):
    return {"id": rid, "reason": reason, "events": events, "stop": stop if reason is None else None,
            "pair_id": None, "kind": "instructed"}


PASS = {"type": "pass", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
TURN = {"type": "turn", "side": "left", "relation": "behind", "objects": ["f:b1"], "arc_m": 3.0,
        "turn_xz": [0, 3], "pre_heading_xz": [0, 1], "gt_lateral_m": 1.0}


def test_same_description_geometry():
    assert same_description_geometry(_row("a", [PASS]), _row("a", [dict(PASS, arc_m=1.2)]))   # arc does not matter
    assert not same_description_geometry(_row("a", [PASS]), _row("a", [PASS, TURN]))
    assert not same_description_geometry(_row("a", [PASS]), _row("a", [], reason="spur"))
    assert not same_description_geometry(_row("a", [PASS]), _row("a", [PASS], stop=dict(STOP, object="f:b9")))


def test_seed_reuses_unchanged_routes_and_lists_changed_ones():
    old = [_row("a", [PASS]), _row("b", [PASS]), _row("c", [], reason="no_events_bent")]
    new = [_row("a", [PASS]), _row("b", [PASS, TURN]), _row("c", [TURN]), _row("d", [], reason="spur")]
    objects = [{"object_id": "f:b1"}, {"object_id": "f:b2"}]
    names = {"gemma": [{"id": "f:b1", "name": "sofa", "passed": True}, {"id": "f:b7", "name": "x", "passed": True},
                       {"id": "f:b2", "error": "timeout"}], "cosmos": []}
    desc = [{"id": "a", "description": "Go left of the sofa.", "passed": True},
            {"id": "b", "description": "Go left of the sofa.", "passed": True}]
    s = seed_d2_labels(old, new, objects, names, desc)
    assert [r["id"] for r in s["desc"]] == ["a"] and s["desc"][0]["reused_from"] == "d1"
    assert [r["id"] for r in s["changed"]] == ["b", "c"]
    assert [r["id"] for r in s["names"]["gemma"]] == ["f:b1"]          # gone object and error row are not seeded


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _seed_args(tmp_path, force=False):
    d1, v4, labels = tmp_path / "v2", tmp_path / "v4", tmp_path / "labels"
    for s in ("train", "eval"):
        _write(d1 / f"events_{s}.jsonl", [_row(f"{s}a", [PASS])])
        _write(v4 / f"events_{s}.jsonl", [_row(f"{s}a", [PASS]), _row(f"{s}b", [PASS, TURN])])
        _write(v4 / f"objects_{s}.jsonl", [{"object_id": "f:b1"}, {"object_id": "f:b2"}])
        _write(labels / f"names_{s}_gemma_d1.jsonl", [{"id": "f:b1", "name": "sofa", "passed": True}])
        _write(labels / f"names_{s}_d1.jsonl", [])
        _write(labels / f"desc_{s}_mix_d1.jsonl", [{"id": f"{s}a", "description": "Go left of the sofa.", "passed": True}])
    return Namespace(d1_events_dir=d1, events_dir=v4, labels=labels, force=force)


def test_d2_seed_stage_writes_seeded_files_and_the_gate_subset(tmp_path):
    args = _seed_args(tmp_path)
    prep.stage_d2_seed(args)
    desc = [json.loads(x) for x in (args.labels / "desc_eval_mix_d2.jsonl").read_text().splitlines()]
    assert [r["id"] for r in desc] == ["evala"]
    changed = [json.loads(x) for x in (args.events_dir / "events_eval_changed.jsonl").read_text().splitlines()]
    assert [r["id"] for r in changed] == ["evalb"]
    objs = [json.loads(x)["object_id"] for x in (args.events_dir / "objects_eval_changed.jsonl").read_text().splitlines()]
    assert set(objs) == {"f:b1", "f:b2"}
    meta = json.loads((args.events_dir / "seed_meta.json").read_text())
    assert meta["eval"]["reused"] == 1 and meta["eval"]["to_compose"] == 1 and meta["eval"]["changed_with_turn"] == 1


def test_d2_seed_refuses_to_overwrite_label_work(tmp_path):   # Review Focus 3
    args = _seed_args(tmp_path)
    prep.stage_d2_seed(args)
    with pytest.raises(SystemExit, match="refusing"):
        prep.stage_d2_seed(args)
    prep.stage_d2_seed(_seed_args(tmp_path, force=True))


def test_finalize_recheck_drops_a_reused_text_that_breaks_a_d2_rule():
    rec = {"id": "a", "image": ["/x.png"], "conversations": [
        {"from": "human", "value": "<image>\n…"},
        {"from": "gpt", "value": json.dumps({"path": [[0.5, 1.0, 1], [0.5, 0.6, 1]]})}],
        "habitat_meta": {"sample_id": "s", "pair_id": None}}
    ev = {"a": {**_row("a", [PASS]), "pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 3]}}
    objects = {"f:b1": {"cells_xz": [[0.5, 1.0]]}}
    bad = {"a": {"id": "a", "passed": True, "description": "Go left of the sofa, then turn right.", "instruction_source": "cosmos"}}
    out = finalize_descriptions([rec], ev, bad, objects, recheck=turn_rejects)
    assert out["stats"]["recheck_failed"] == 1 and out["kept"] == []
    assert finalize_descriptions([rec], ev, bad, objects)["stats"]["kept"] == 1          # d1 behaviour unchanged


def test_desc_path_round():
    assert prep.desc_path(Path("L"), "eval", "_mix").name == "desc_eval_mix_d1.jsonl"
    assert prep.desc_path(Path("L"), "eval", "_mix", "d2").name == "desc_eval_mix_d2.jsonl"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_d2_stages.py -v`
Expected: `ImportError: cannot import name 'same_description_geometry'`.

- [ ] **Step 3: Implement**

In `src/rover_vlm/descriptions.py` (after `finalize_descriptions`'s helpers, before `as_form`):

```python
def _geometry_key(row):
    st = row["stop"]
    return ([(e["type"], e["side"], tuple(e["objects"])) for e in row["events"]],
            (st["type"], st["relation"], st.get("amount"), st["object"]))


def same_description_geometry(old, new):
    """True when a d2 events row describes its route as the d1 row did: both described, same events (type,
    side, objects; arcs may move) and the same stop. Its d1 text then stays true: no d1 text says "turn"."""
    if old.get("reason") is not None or new.get("reason") is not None:
        return False
    return _geometry_key(old) == _geometry_key(new)


def seed_d2_labels(old_events, new_events, new_objects, old_names, old_desc):
    """What the d2 label job can start from (spec "Seeding"): the d1 name rows of objects that still exist
    ({"gemma": rows, "cosmos": rows}, error rows left out so they are retried), the d1 desc row of every
    unchanged described route (tagged "reused_from": "d1"), and the described routes that need composing."""
    ids = {o["object_id"] for o in new_objects}
    names = {tag: [r for r in rows if r["id"] in ids and not r.get("error")] for tag, rows in old_names.items()}
    old = {r["id"]: r for r in old_events}
    desc = {r["id"]: r for r in old_desc if not r.get("error")}
    reused, changed = [], []
    for r in new_events:
        if r.get("reason") is not None:
            continue
        o = old.get(r["id"])
        if o is not None and r["id"] in desc and same_description_geometry(o, r):
            reused.append({**desc[r["id"]], "reused_from": "d1"})
        else:
            changed.append(r)
    return {"names": names, "desc": reused, "changed": changed}
```

Change `finalize_descriptions`' signature and its passed-row branch:

```python
def finalize_descriptions(label_recs, events_by_id, desc_by_id, objects, recheck=None):
    kept, geo, unlabelled, rejected, rechecked = [], 0, 0, 0, 0
    for rec in label_recs:
        ...                                   # unchanged up to `if not row.get("passed"):`
        if not row.get("passed"):
            rejected += 1
            continue
        if recheck is not None and recheck(row["description"], ev["events"]):
            rechecked += 1                    # d2: a reused d1 text that breaks a d2 rule (spec: finalize re-checks)
            continue
        ...
```
and add `"recheck_failed": rechecked,` to the stats dict.

In `scripts/prepare_habitat_instr.py`:

```python
V4_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v4"          # d2 (spec 2026-10-13)
```
(next to `V3_OUT`).

`stage_events`: read the labelset from `args.labelset_dir or d`, and create `d`:

```python
def stage_events(args):
    import time
    d = args.out_dir
    d.mkdir(parents=True, exist_ok=True)
    src = args.labelset_dir or d
    meta = {"schema": EVENTS_SCHEMA, "labelset_dir": str(src)}
    for name in SPLITS:
        recs = json.loads((src / f"labelset_{name}.json").read_text())
        ...                                   # rest unchanged
```
Its parser gets `h.add_argument("--labelset-dir", type=Path, default=None, help="where labelset_<split>.json are
(default: --out-dir); d2: data/prepared_habitat_instr_v2")`.

`desc_path`:

```python
def desc_path(labels: Path, split: str, suffix: str = "", round_: str = "d1") -> Path:
    """Composed-description file of one split: desc_<split><suffix>_<round>.jsonl (suffix "_mix" = NAMER=mix)."""
    return labels / f"desc_{split}{suffix}_{round_}.jsonl"
```

New stage:

```python
def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


COMPOSE_PER_MIN = 14     # Cosmos compose throughput on 3g.40gb (job 92229)


def stage_d2_seed(args):
    """d2: seed the d2 label files with the d1 rows that still hold, so the label job (--resume) names only new
    objects and composes only changed routes; write the changed eval routes for the gate (spec "Seeding")."""
    files = {s: {"gemma": args.labels / f"names_{s}_gemma_d2.jsonl", "cosmos": args.labels / f"names_{s}_d2.jsonl",
                 "desc": desc_path(args.labels, s, "_mix", "d2")} for s in SPLITS}
    existing = [str(p) for f in files.values() for p in f.values() if p.exists() and p.stat().st_size]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite d2 label files (they may hold GPU work): {existing[:3]} — "
                         "pass --force only if they are seeds you mean to redo")
    meta = {}
    for s in SPLITS:
        new_ev = _jsonl(args.events_dir / f"events_{s}.jsonl")
        new_obj = _jsonl(args.events_dir / f"objects_{s}.jsonl")
        seed = seed_d2_labels(_jsonl(args.d1_events_dir / f"events_{s}.jsonl"), new_ev, new_obj,
                              {"gemma": _jsonl(args.labels / f"names_{s}_gemma_d1.jsonl"),
                               "cosmos": _jsonl(args.labels / f"names_{s}_d1.jsonl")},
                              _jsonl(desc_path(args.labels, s, "_mix")))
        _write_jsonl(files[s]["gemma"], seed["names"]["gemma"])
        _write_jsonl(files[s]["cosmos"], seed["names"]["cosmos"])
        _write_jsonl(files[s]["desc"], seed["desc"])
        wanted = {o for r in seed["changed"] for o in route_objects(r)}
        if s == "eval":
            _write_jsonl(args.events_dir / "events_eval_changed.jsonl", seed["changed"])
            _write_jsonl(args.events_dir / "objects_eval_changed.jsonl",
                         [o for o in new_obj if o["object_id"] in wanted])
        named = {r["id"] for r in seed["names"]["gemma"] if r.get("passed")}
        meta[s] = {"described": sum(r.get("reason") is None for r in new_ev), "reused": len(seed["desc"]),
                   "to_compose": len(seed["changed"]),
                   "changed_with_turn": sum(any(e["type"] == "turn" for e in r["events"]) for r in seed["changed"]),
                   "objects": len(new_obj), "objects_named_d1_gemma": len(named),
                   "objects_to_name": len(wanted - named),
                   "est_compose_hours": round(len(seed["changed"]) / COMPOSE_PER_MIN / 60, 1)}
    (args.events_dir / "seed_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
```
(import `route_objects`, `seed_d2_labels`, `turn_rejects` from `rover_vlm.descriptions` at the top).

Parser:

```python
    sd2 = sub.add_parser("d2-seed", help="d2: seed the d2 label files from d1 + the changed eval routes (CPU)")
    sd2.add_argument("--d1-events-dir", type=Path, default=V2_OUT)
    sd2.add_argument("--events-dir", type=Path, default=V4_OUT)
    sd2.add_argument("--labels", type=Path, default=D1_LABELS)
    sd2.add_argument("--force", action="store_true", help="overwrite existing d2 seed files")
```
and `"d2-seed": stage_d2_seed` in the dispatch dict.

`stage_finalize_descriptions`, defaults by round:

```python
def stage_finalize_descriptions(args):
    import random
    d2 = args.round != "d1"
    events_dir = args.events_dir or (V4_OUT if d2 else V2_OUT)
    labelset_dir = args.labelset_dir or V2_OUT
    out_dir = args.out_dir or (V4_OUT if d2 else V3_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, splits = {}, {}
    for name in SPLITS:
        recs = json.loads((labelset_dir / f"labelset_{name}.json").read_text())
        events = {r["id"]: r for r in map(json.loads, (events_dir / f"events_{name}.jsonl").open())}
        objects = {o["object_id"]: o for o in map(json.loads, (events_dir / f"objects_{name}.jsonl").open())}
        desc = _label_rows([desc_path(args.labels, name, args.desc_suffix, args.round)])
        out = finalize_descriptions(recs, events, desc, objects, recheck=turn_rejects if d2 else None)
        ...                                   # unchanged checks
    ...
    d = out_dir
    ...                                       # unchanged file writing
    (d / ("meta_d2.json" if d2 else "meta_v3.json")).write_text(json.dumps(meta, indent=2))
```
Parser: `--events-dir` / `--out-dir` default `None`; add `k.add_argument("--labelset-dir", type=Path, default=None)`
and `k.add_argument("--round", default="d1", choices=["d1", "d2"])`. The existing test
`test_finalize_descriptions_desc_suffix` calls `desc_path(seen["labels"], "eval", seen["desc_suffix"])`; it still
passes with the default round.

Docstring: add a "d2" section to the script's module docstring:

```
Turn events and d2 (2026-10-13, docs/superpowers/specs/2026-10-13-turn-events-d2-design.md): events into v4 from
the v2 labelsets, seed the d2 label files from d1, label (GPU), finalize. CPU, 4 workers:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py events --labelset-dir data/prepared_habitat_instr_v2 \
        --out-dir data/prepared_habitat_instr_v4
    uv run scripts/prepare_habitat_instr.py d2-seed
    ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix sbatch slurm/label_descriptions.sbatch   # gate
    uv run scripts/prepare_habitat_instr.py finalize-descriptions --round d2 --desc-suffix _mix
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_d2_stages.py tests/test_route_events.py tests/test_descriptions.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py scripts/prepare_habitat_instr.py tests/test_d2_stages.py
git commit -m "feat: d2 stages — events from v2 labelsets, d2-seed (reuse d1 labels), finalize --round d2 re-check"
```

---

### Task 7: sbatch: `ROUND`, the gate on changed routes, `STEER_DIR`, eval `DATA` / `OUT_DIR`

**Files:**
- Modify: `slurm/label_descriptions.sbatch`, `slurm/eval_descriptions.sbatch`
- Test: `tests/test_slurm_thread_caps.py` (update two tests, add two)

**Interfaces:**
- Consumes: Task 6 files (`events_eval_changed.jsonl`, `objects_eval_changed.jsonl`, seeded `*_d2.jsonl`).
- Produces:
  - `ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix [STAGE=full] sbatch slurm/label_descriptions.sbatch`
  - `ROUND=d2 STEER_DIR=data/prepared_habitat_instr_v4/steer STAGE=steer_names|steer sbatch slurm/label_descriptions.sbatch`
  - `DATA=data/prepared_habitat_instr_v4 OUT_DIR=outputs/eval_desc_d2 [SETS=steer] ADAPTER=… TAG=… sbatch slurm/eval_descriptions.sbatch`

- [ ] **Step 1: Write the failing tests**

In `tests/test_slurm_thread_caps.py`, replace the two steer tests and add two:

```python
def test_label_descriptions_has_a_steer_names_stage():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert '"$STAGE" = "steer_names"' in text and "--all-objects" in text
    assert 'names_steer_gemma_${ROUND}.jsonl' in text and 'serve "$GEMMA_MODEL"' in text


def test_label_descriptions_has_a_steer_stage():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert '"$STAGE" = "steer"' in text and "events_steer_eval.jsonl" in text
    assert 'desc_steer_eval_${ROUND}.jsonl' in text and 'missing $OUT/names_eval_${ROUND}.jsonl' in text
    assert "names_primary_eval.jsonl" in text and "--fallback-names" in text
    assert "steer-propose" not in text


def test_label_descriptions_round_and_changed_gate():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert 'ROUND="${ROUND:-d1}"' in text and "_d1.jsonl" not in text.split("set -euo pipefail", 1)[1]
    assert 'GATE_SPLIT=eval_changed' in text and 'name_stage "$GATE_SPLIT" gate --limit 300' in text
    assert 'SD="${STEER_DIR:-data/prepared_habitat_instr_v3/steer}"' in text


def test_eval_descriptions_data_and_out_dir():
    text = (SLURM / "eval_descriptions.sbatch").read_text()
    assert 'D="${DATA:-data/prepared_habitat_instr_v3}"' in text and 'OUT="${OUT_DIR:-outputs/eval_desc}"' in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_slurm_thread_caps.py -v`
Expected: the four tests FAIL (literals not found).

- [ ] **Step 3: Implement**

`slurm/label_descriptions.sbatch`:
- After `STAGE="${STAGE:-gate}"` add:
  ```bash
  ROUND="${ROUND:-d1}"     # label-file tag: d1, or d2 (turn events; files seeded by prepare_habitat_instr.py d2-seed)
  GATE_SPLIT=eval          # d2 gates only the changed eval routes (events_eval_changed.jsonl, written by d2-seed)
  if [ "$ROUND" != d1 ]; then
      GATE_SPLIT=eval_changed
      for f in events objects; do
          [ -f "$D/${f}_eval_changed.jsonl" ] || { echo "missing $D/${f}_eval_changed.jsonl — run prepare_habitat_instr.py d2-seed"; exit 1; }
      done
  fi
  ```
- Replace every `_d1.jsonl` after `set -euo pipefail` with `_${ROUND}.jsonl`. That covers `name_stage`'s `--out`,
  `compose_stage`'s `--names` / `--out`, `mix_cosmos_phase`'s `--only-unusable-under`, `mix_compose`'s
  `--fallback-names`, the steer stages' `names_steer_gemma_…`, `names_eval_…` (both the existence check and
  `--fallback-names`) and `desc_steer_eval_…`, plus the final `wc -l "$OUT"/*_${ROUND}.jsonl`. Header comment
  lines before `set -euo pipefail` keep their d1 examples.
- In the `STAGE=gate` block, replace each `name_stage eval gate --limit 300` with
  `name_stage "$GATE_SPLIT" gate --limit 300`, each `compose_stage eval gate …` with
  `compose_stage "$GATE_SPLIT" gate …`, and likewise `mix_cosmos_phase` / `mix_compose`.
- Both steer blocks: replace `SD=data/prepared_habitat_instr_v3/steer` with
  `SD="${STEER_DIR:-data/prepared_habitat_instr_v3/steer}"`.
- Header comment: add a d2 paragraph:
  ```bash
  # d2 (2026-10-13, turn events): after prepare_habitat_instr.py events --out-dir data/prepared_habitat_instr_v4
  # and d2-seed, the same job names only new objects and composes only changed routes (--resume on seeded files):
  #     ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix sbatch slurm/label_descriptions.sbatch            # gate
  #     ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix STAGE=full sbatch slurm/label_descriptions.sbatch
  #     ROUND=d2 STEER_DIR=data/prepared_habitat_instr_v4/steer STAGE=steer_names sbatch slurm/label_descriptions.sbatch
  ```

`slurm/eval_descriptions.sbatch`: replace `D=data/prepared_habitat_instr_v3` with
`D="${DATA:-data/prepared_habitat_instr_v3}"` and `OUT=outputs/eval_desc` with `OUT="${OUT_DIR:-outputs/eval_desc}"`.
Add the header line
`#     DATA=data/prepared_habitat_instr_v4 OUT_DIR=outputs/eval_desc_d2 ADAPTER=outputs/runs_desc/d2/adapter TAG=d2 sbatch slurm/eval_descriptions.sbatch`.

- [ ] **Step 4: Run the tests and a syntax check**

Run: `uv run pytest tests/test_slurm_thread_caps.py -q && bash -n slurm/label_descriptions.sbatch && bash -n slurm/eval_descriptions.sbatch`
Expected: all pass, no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add slurm/label_descriptions.sbatch slurm/eval_descriptions.sbatch tests/test_slurm_thread_caps.py
git commit -m "feat: label sbatch ROUND tag + d2 gate on changed routes + STEER_DIR; eval sbatch DATA/OUT_DIR"
```

---

### Task 8: Steering: turn cuts, a `turn` slice, texts-differ rule

**Files:**
- Modify: `src/rover_vlm/steering.py`: `TURN_CUT_BEFORE_M`, `TURN_CUT_AFTER_M`, `GATE0_MIN_TURN_PAIRS`,
  `turn_cuts`, `stop_pairs`, `frame_pairs`, `select_pairs`, `finalize_steer`, module docstring.
- Modify: `scripts/prepare_habitat_instr.py`: `stage_steer` (count viewpoints per slice), `stage_steer_select`
  (Gate 0 overall + turn), `stage_finalize_steer` (Gate 2 over all slices).
- Test: `tests/test_steering.py` (append + update one assertion)

**Interfaces:**
- Consumes: `detect_turn`, `describe_xz(..., turn=)` (Task 2); the eval turn check (Task 5).
- Produces:
  - `turn_cuts(route_xz, turn, min_m=MIN_ROUTE_M) -> [(fraction, prefix_pts), (fraction, prefix_pts)] | []`
  - `stop_pairs(frame, meta, route_xz, turn=None)`: pairs carry `slice` "turn" (the A_turn/B_turn pair) or
    "stop"; reason `same_text` replaces `same_stop`.
  - `pair_id = f"steer_{slice}_{frame}_{fractions}"`.
  - `select_pairs` keeps one pair per viewpoint and prefers the turn slice.
  - `finalize_steer` writes both slices into the four `eval_steer_stop_*` files.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering.py`:

```python
from rover_vlm.route_events import detect_turn  # noqa: E402


def _behind_scene():
    s = _state()
    s[32:52, 43:55] = OCC          # box ahead: x 2.15..2.75, z 1.6..2.6
    s[52:68, 96:99] = OCC          # cabinet: x 4.8..4.95, z 2.6..3.4
    return s


BEHIND = [(2.5, 0.2), (1.9, 1.2), (1.9, 3.0), (3.9, 3.0)]     # right of the box, then left behind it


def test_turn_cuts_straddle_the_turn():
    t = detect_turn(BEHIND)
    (fa, a), (fb, b) = st.turn_cuts(BEHIND, t)
    assert st.route_length(a) == pytest.approx(t["start_arc_m"] - st.TURN_CUT_BEFORE_M, abs=0.06)
    assert st.route_length(b) == pytest.approx(t["end_arc_m"] + st.TURN_CUT_AFTER_M, abs=0.06)
    assert fa < fb <= 1.0
    short = [(0.0, 0.0), (0.0, 1.2), (-0.8, 1.2)]                   # turns (56 deg right) before 1.5 m: no A cut
    assert detect_turn(short) is not None and st.turn_cuts(short, detect_turn(short)) == []


def test_a_turning_route_gives_a_turn_pair(tmp_path):
    sd = _sample(tmp_path / "f", _behind_scene(), [BEHIND])
    meta, frame = load_frame(sd)
    pairs, reasons = st.stop_pairs(frame, meta, BEHIND, turn=detect_turn(BEHIND))
    turn = [p for p in pairs if p["slice"] == "turn"]
    assert len(turn) == 1, dict(reasons)
    a, b = turn[0]["routes"]
    assert not [e for e in a["desc"]["events"] if e["type"] == "turn"]
    assert [e["side"] for e in b["desc"]["events"] if e["type"] == "turn"] == ["left"]
    assert st.exclusive(a, b) and turn[0]["separation"] >= st.MIN_SEP_UV


def test_select_pairs_prefers_the_turn_pair_of_a_viewpoint():
    cands = [_cand("s1", "v1", False, 0.5), {**_cand("t1", "v1", False, 0.2), "slice": "turn"},
             _cand("s2", "v2", False, 0.3)]
    assert [p["pair_id"] for p in st.select_pairs(cands)] == ["s2", "t1"]


def test_finalize_steer_puts_turn_pairs_in_the_stop_files():
    rows = [{**_srow("p", "A", "B"), "slice": "turn"}, {**_srow("p", "B", "A"), "slice": "turn"}]
    out = st.finalize_steer({"f": _base("f")}, rows, {})
    recs = out["files"]["eval_steer_stop_template_entry"]
    assert len(recs) == 2 and {r["habitat_meta"]["steer"]["slice"] for r in recs} == {"turn"}
    assert out["stats"]["turn:pairs"] == 1
```

In `test_stop_pairs_on_a_route_past_a_box_to_a_cabinet`, replace
`assert st.stop_key(a["desc"]["stop"]) != st.stop_key(b["desc"]["stop"])` with

```python
        assert (st.stop_key(a["desc"]["stop"]), st._event_keys(a["desc"]["events"])) != \
               (st.stop_key(b["desc"]["stop"]), st._event_keys(b["desc"]["events"]))     # the texts differ
```

(`_srow(pid, tag, sib)` and `_base(sid)` are the file's existing helpers, near line 381: one route row, one base
record.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_steering.py -v`
Expected: FAIL with `AttributeError: module 'rover_vlm.steering' has no attribute 'turn_cuts'`, the select test picks
`s1`, and finalize raises `KeyError: 'eval_steer_turn_template_entry'`.

- [ ] **Step 3: Implement**

In `src/rover_vlm/steering.py`:

```python
from .route_events import detect_turn   # add to the existing route_events import

TURN_CUT_BEFORE_M = 0.3    # A_turn stops this far before the turn starts ("stop before the turn")
TURN_CUT_AFTER_M = 1.0     # B_turn stops this far after it ends ("turn ... and stop past X")
GATE0_MIN_TURN_PAIRS = 50


def turn_cuts(route_xz, turn, min_m=MIN_ROUTE_M):
    """(fraction, prefix) of the two turn cuts of a turning route (spec "Steering"): A TURN_CUT_BEFORE_M before
    the turn starts, B TURN_CUT_AFTER_M after it ends (or at the route's end). Both or none: a cut shorter
    than min_m drops the pair."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    out = []
    for s in (turn["start_arc_m"] - TURN_CUT_BEFORE_M, min(turn["end_arc_m"] + TURN_CUT_AFTER_M, arc[-1])):
        if s < min_m:
            return []
        out.append((round(float(s / arc[-1]), 3), pts[: max(2, int(np.searchsorted(arc, s - 1e-9)) + 1)]))
    return out
```

`stop_pairs(frame, meta, route_xz, turn=None)`: build `made = [(f, cut, False) for f, cut in cuts(route_xz)] +
[(f, cut, True) for f, cut in (turn_cuts(route_xz, turn) if turn else [])]`. Describe each with
`describe_xz(meta, frame, cut, turn=turn)`, and store `"turn_cut": is_turn` in the route dict. In the pair loop
replace the same-stop test with

```python
        if (stop_key(a["desc"]["stop"]), _event_keys(a["desc"]["events"])) == \
                (stop_key(b["desc"]["stop"]), _event_keys(b["desc"]["events"])):
            reasons["same_text"] += 1
            continue
```
and set `"slice": "turn" if a["turn_cut"] and b["turn_cut"] else "stop"` in the appended pair. `_event_keys`
already keys on (type, side, objects), so a turn event counts. `itertools.combinations` keeps the routes in
`made` order, so A is the earlier cut for regular pairs and for the turn pair. Sort `routes` by route length
before the combinations, so A is always the shorter cut.

`frame_pairs`: compute `route = cm[candidate_index]["polyline_xz"]`, call
`stop_pairs(frame, meta, route, turn=detect_turn(route))`, and build
`p["pair_id"] = f"steer_{p['slice']}_{frame.name}_" + "_".join(...)`.

`select_pairs`:

```python
def select_pairs(candidates):
    """One pair per viewpoint: a turn pair when the viewpoint has one (spec "Steering"), then the original frame
    before its yaw variants, then the largest end separation (pair_id breaks ties). Sorted by pair_id."""
    best = {}
    for p in candidates:
        rank = (p["slice"] != "turn", bool(p["is_aug"]), -p["separation"], p["pair_id"])
        if p["viewpoint"] not in best or rank < best[p["viewpoint"]][0]:
            best[p["viewpoint"]] = (rank, p)
    return sorted((v[1] for v in best.values()), key=lambda p: p["pair_id"])
```

`finalize_steer`: write to fixed file names (both slices share the four sets; records carry `steer.slice`):

```python
        for form in ("entry", "goal"):
            files[f"eval_steer_stop_template_{form}"] += [steer_record(base, r, r["template"], "template", form)
                                                          for r in pair]
        ...
            files[f"eval_steer_stop_composed_{form}"] += [...]
```
Keep the per-slice stats keys (`f"{sl}:pairs"` etc.).

In `scripts/prepare_habitat_instr.py`:
- `stage_steer`: `n_frames = {}`, and `n_frames.setdefault(p["slice"], set()).add(p["viewpoint"])`.
- `stage_steer_select`: replace the Gate 0 print with
  ```python
      n, t = len(kept), per.get("turn", 0)
      ok = n >= steering.GATE0_MIN_PAIRS and t >= steering.GATE0_MIN_TURN_PAIRS
      print(f"GATE 0: {n} pairs ({t} turn, {per.get('stop', 0)} stop) -> {'PASS' if ok else 'FAIL'} "
            f"(need >= {steering.GATE0_MIN_PAIRS} and >= {steering.GATE0_MIN_TURN_PAIRS} turn)")
  ```
- `stage_finalize_steer`: Gate 2 over all slices:
  ```python
      n = sum(v for k, v in stats.items() if k.endswith(":pairs"))
      c = sum(v for k, v in stats.items() if k.endswith(":composed"))
      share = c / n if n else 0.0
      print(f"GATE 2: composed {c}/{n} = {share:.3f} -> {'PASS' if share >= GATE2_MIN_COMPOSED else 'FAIL'} "
            f"(need >= {GATE2_MIN_COMPOSED})")
  ```

Module docstring of `steering.py`: add `d2 (2026-10-13): a turning GT route also gets two turn cuts (just before
the turn, 1 m after it); that pair is the viewpoint's pair, slice "turn".`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_steering.py tests/test_description_eval.py -q`
Expected: all pass. The prototype on this scene gave A "pass right, stop just after" and B "pass right + turn left,
stop a little past". The pair is exclusive, with its ends 0.23 apart.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py scripts/prepare_habitat_instr.py tests/test_steering.py
git commit -m "feat: steering turn cuts and turn slice; texts-differ pair rule; Gate 0 needs 50 turn pairs"
```

---

### Task 9: Audits: G1 turns (`--mode turns`), G3 (`--mode d1 --round d2`), stratified steer Gate 1

**Files:**
- Modify: `scripts/audit_instruction_labels.py`: `VERDICTS["turns"]`, `sample_turns`, `turns_item`,
  `select_d2_rows`, `sample_steer(n_turn=)`, `steer_item` (draw turn anchors), `main` (modes and args).
- Test: `tests/test_d2_stages.py` (append)

**Interfaces:**
- Consumes: v4 events rows (Task 2), d2 desc rows (`prompt_version` "d4"), steering rows with `slice`.
- Produces:
  - `sample_turns(rows_d2, rows_d1, seed, n_visible=20, n_hidden=30, n_bent=20, n_spur=30) -> (picked [(group, row)], population)`
  - `select_d2_rows(ok, events_by_id, n, seed) -> rows` (half with a turn)
  - `sample_steer(rows, n, seed, n_turn=0)`
  - CLI:
    - `--mode turns --events v4/events_train.jsonl --objects v4/objects_train.jsonl --d1-events v2/events_train.jsonl`
    - `--mode d1 --round d2 --events … --objects … --desc …_d2.jsonl`
    - `--mode steer --n 50 --n-turn 20 --events v4/steer/events_steer_eval.jsonl`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_d2_stages.py`:

```python
_ASPEC = importlib.util.spec_from_file_location(
    "audit_instruction_labels", Path(__file__).resolve().parent.parent / "scripts" / "audit_instruction_labels.py")
audit = importlib.util.module_from_spec(_ASPEC)
_ASPEC.loader.exec_module(audit)


def _ev(rid, reason=None, rel=None):
    events = [] if rel is False else [dict(TURN, relation=rel)]
    return {"id": rid, "reason": reason, "events": events if reason is None else [], "pair_id": None}


def test_sample_turns_stratifies():
    d2 = ([_ev(f"h{i}", rel="behind") for i in range(40)] + [_ev(f"v{i}", rel=None) for i in range(10)]
          + [_ev(f"r{i}", rel="round") for i in range(5)] + [_ev(f"b{i}", rel=False) for i in range(30)]
          + [_ev(f"s{i}", reason="spur") for i in range(50)])
    d1 = [{"id": f"b{i}", "reason": "no_events_bent"} for i in range(25)]
    picked, pop = audit.sample_turns(d2, d1, seed=0)
    groups = [g for g, _ in picked]
    assert groups.count("turn_visible") == 15 and groups.count("turn_hidden") == 35   # 20 visible wanted, 15 exist
    assert groups.count("bent") == 20 and groups.count("spur") == 30
    assert pop == {"turn_visible": 15, "turn_hidden": 40, "rescued_bent": 25, "spur": 50}


def test_select_d2_rows_half_with_a_turn():
    ok = [{"id": f"t{i}", "prompt_version": "d4"} for i in range(80)] + \
         [{"id": f"n{i}", "prompt_version": "d4"} for i in range(80)] + [{"id": "old", "prompt_version": "d3"}]
    ev = {**{f"t{i}": {"events": [TURN]} for i in range(80)}, **{f"n{i}": {"events": [PASS]} for i in range(80)},
          "old": {"events": [PASS]}}
    rows = audit.select_d2_rows(ok, ev, 100, seed=0)
    assert len(rows) == 100 and sum(r["id"].startswith("t") for r in rows) == 50 and "old" not in {r["id"] for r in rows}


def test_sample_steer_takes_turn_pairs_first():
    rows = [{"pair_id": f"t{i}", "id": f"t{i}_{k}", "slice": "turn"} for i in range(30) for k in "AB"] + \
           [{"pair_id": f"s{i}", "id": f"s{i}_{k}", "slice": "stop"} for i in range(60) for k in "AB"]
    picked, pop = audit.sample_steer(rows, 50, 0, n_turn=20)
    assert len(picked) == 50 and sum(p[0]["slice"] == "turn" for p in picked) == 20 and pop == 90
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_d2_stages.py -k "sample or select" -v`
Expected: `AttributeError: module 'audit_instruction_labels' has no attribute 'sample_turns'`.

- [ ] **Step 3: Implement**

In `scripts/audit_instruction_labels.py`:

```python
VERDICTS["turns"] = [
    {"v": "ok", "label": "True", "help": "Every step (the turn included) and the stop are true; for a dropped route, dropping it is right"},
    {"v": "turn", "label": "Wrong turn", "help": "There is no such turn, or it goes the other way"},
    {"v": "anchor", "label": "Wrong turn object", "help": "'behind / round' the outlined object is not true"},
    {"v": "step", "label": "Other step or stop wrong", "help": "A pass / wall / between / through step or the stop is wrong"},
    {"v": "keep", "label": "Should not be dropped", "help": "A spur or bent drop of a route that is fine"},
    {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
]
TURN_MARK = (255, 60, 60)


def _turn(row):
    return next(e for e in row["events"] if e["type"] == "turn")


def sample_turns(rows_d2, rows_d1, seed, n_visible=20, n_hidden=30, n_bent=20, n_spur=30):
    """G1 (spec "Success criteria"): ~20 visible + ~30 hidden turn routes (hidden fills what visible lacks),
    20 routes d1 dropped as no_events_bent that d2 describes, 30 routes d2 drops as spur. Shuffled."""
    rng = random.Random(seed)
    d1 = {r["id"]: r for r in rows_d1}
    turns = [r for r in rows_d2 if r["reason"] is None and any(e["type"] == "turn" for e in r["events"])]
    hidden = [r for r in turns if _turn(r)["relation"] == "behind"]
    visible = [r for r in turns if _turn(r)["relation"] != "behind"]
    bent = [r for r in rows_d2 if r["reason"] is None and (d1.get(r["id"]) or {}).get("reason") == "no_events_bent"]
    spur = [r for r in rows_d2 if r["reason"] == "spur"]
    k_vis = min(n_visible, len(visible))
    k_hid = min(n_hidden + n_visible - k_vis, len(hidden))
    picked = ([("turn_visible", r) for r in rng.sample(visible, k_vis)]
              + [("turn_hidden", r) for r in rng.sample(hidden, k_hid)]
              + [("bent", r) for r in rng.sample(bent, min(n_bent, len(bent)))]
              + [("spur", r) for r in rng.sample(spur, min(n_spur, len(spur)))])
    rng.shuffle(picked)
    return picked, {"turn_visible": len(visible), "turn_hidden": len(hidden), "rescued_bent": len(bent), "spur": len(spur)}


def turns_item(rec, row, objects, group):
    """One G1 item: the GT route, numbered objects and the turn point; a spur drop shows the route alone."""
    if group == "spur":
        img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
        draw_reference_line(img, _answer_path(rec))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        m = rec["habitat_meta"]
        return {"id": rec["id"], "instruction": "DROPPED as an out-and-back spur. Is dropping this route right?",
                "side": m["side"], "kind": m["kind"], "has_alt": False, "scene": m["scene_id"], "target": "spur",
                "reason": "spur", "group": group,
                "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
    item = events_item(rec, row, objects)
    t = next((e for e in row["events"] if e["type"] == "turn"), None)
    if t is not None:
        img = Image.open(io.BytesIO(base64.b64decode(item["img"].split(",", 1)[1])))
        uv, fwd = Camera(row["pose_xz"], row["heading_xz"]).uv(np.array([t["turn_xz"]], float))
        if fwd[0] > 0:
            x, y = float(uv[0, 0]) * IMG_PX, float(uv[0, 1]) * IMG_PX
            ImageDraw.Draw(img).ellipse([x - 10, y - 10, x + 10, y + 10], outline=TURN_MARK, width=4)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        item["img"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    item["group"] = group
    return item


def select_d2_rows(ok, events_by_id, n, seed):
    """G3: n passed d2 rows composed with the current prompt (reused d1 rows left out), half with a turn."""
    rng = random.Random(seed)
    new = [r for r in ok if r.get("prompt_version") == "d4" and r["id"] in events_by_id]
    has = [r for r in new if any(e["type"] == "turn" for e in events_by_id[r["id"]]["events"])]
    no = [r for r in new if r not in has]
    k = min(n // 2, len(has))
    rows = rng.sample(has, k) + rng.sample(no, min(n - k, len(no)))
    rng.shuffle(rows)
    return rows
```

`sample_steer(rows, n, seed, n_turn=0)`: after building `pairs`, pick `min(n_turn, …)` turn pairs (`p[0]["slice"]
== "turn"`) with the rng, then fill to `n` from the other pairs, then shuffle. Return `(picked, len(pairs))`.

`steer_item`: draw the turn anchors too. Change the outline loop source to
`dict.fromkeys([a["anchor"], b["anchor"]] + [o for e in b["events"] if e["type"] == "turn" for o in e["objects"]])`.

`main`:
- Add `"turns"` to `--mode` choices.
- Add the arguments:
  - `--d1-events` (Path; for `--mode turns`)
  - `--labelset-dir` (Path, default `V2`)
  - `--round` (default "d1")
  - `--n-turn` (int, default 0)
- The `turns` branch:
  - `check_events_schema(args.events)`
  - records from `args.labelset_dir / f"labelset_{split}.json"`, where split comes from the events file name
  - `picked, pop = sample_turns(jsonl(args.events), jsonl(args.d1_events), args.seed)`
  - `items = [turns_item(by_id[r["id"]], r, objects, g) for g, r in picked]`
  - payload `mode "turns"`, `VERDICTS["turns"]`, title "Turn Events Audit", output default
    `outputs/turn_events_audit.html`. Write it the same way as the `events` branch.
- `--mode d1`: when `args.round == "d2"`, use `select_d2_rows(ok, ev, args.n, args.seed)` instead of
  `select_d1_rows`, and the title "d2 Label Audit". Its labelset path uses `args.labelset_dir`.
- `--mode steer`: pass `n_turn=args.n_turn` to `sample_steer`.
- Import `Camera` from `rover_vlm.landmarks`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_d2_stages.py tests/test_steering.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_instruction_labels.py tests/test_d2_stages.py
git commit -m "feat: audits — G1 turn events page, G3 d2 label sample, steer Gate 1 stratified by slice"
```

---

### Task 10: Steering report per slice, README, full test run

**Files:**
- Modify: `scripts/visualize_steering.py`: per-slice rows, gallery tag from `--tags[0]`, docstring d2 command.
- Modify: `README.md` (**index-only commit**: the user's uncommitted row must not be staged).
- Test: `tests/test_steering.py` (append one report test)

**Interfaces:**
- Consumes: `aggregate_steering_metrics(...)["by_slice"]` (Task 5).
- Produces: `metric_cells` returns one row per (text, form, tag) with slice "all", then one per slice present.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_steering.py`:

```python
def test_report_rows_per_slice(tmp_path):
    vs = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        "visualize_steering", Path(__file__).resolve().parent.parent / "scripts" / "visualize_steering.py"))
    vs.__spec__.loader.exec_module(vs)
    from test_description_eval import _pair, A_GT, B_GT      # noqa: E402  (tests dir is on sys.path under pytest)
    recs = _pair(A_GT, B_GT, pid="p") + _pair(B_GT, B_GT, pid="q")
    for r in recs[2:]:
        r["habitat_meta"]["steer"]["slice"] = "turn"
    d = tmp_path / "eval_steer_stop_template_entry" / "d2"
    d.mkdir(parents=True)
    (d / "predictions.json").write_text(json.dumps(recs))
    cells = vs.metric_cells(tmp_path, ["d2"])
    assert [c["slice"] for c in cells] == ["all", "stop", "turn"]
    assert cells[2]["steering"]["pair_success"] == 0.0
```
(add `import importlib.util` at the top of the file if it is not there yet).

- [ ] **Step 2: Run the test to verify it fails**

Run: `OMP_NUM_THREADS=4 uv run pytest tests/test_steering.py -k report -v`
Expected: FAIL: `['stop']` != `['all', 'stop', 'turn']`.

- [ ] **Step 3: Implement**

`scripts/visualize_steering.py`, `metric_cells`:

```python
            recs = json.loads(p.read_text())
            base = {"text": t, "form": f, "tag": tag, "n": len(recs),
                    "parse_rate": sum(bool((r.get("parsed") or {}).get("path")) for r in recs) / max(1, len(recs))}
            s = aggregate_steering_metrics(recs) or {}
            out.append({**base, "slice": "all", "steering": s})
            for sl, v in sorted((s.get("by_slice") or {}).items()):
                out.append({**base, "slice": sl, "steering": v})     # pair success + CI only; the rest shows "—"
```
In `main`, the gallery title and tag become `f"stop pairs — {args.tags[0]}, template text, entry form"` and
`gallery(..., args.tags[0], args.n)`. Add the d2 command to the docstring:
`uv run scripts/visualize_steering.py --tags d1 d2 --eval-dir outputs/eval_desc_d2 --data-dir data/prepared_habitat_instr_v4 --out outputs/steering_report_d2.html`.

README: add a section **"Turn events and d2 (2026-10-13)"** right after the "Steering eval" section, holding the Run
Book commands below (CPU stages, the four sbatch lines, finalize, train, eval, steering rebuild, report) and the
gates. Commit it **index-only**, so the user's uncommitted README row stays out of the commit:

```bash
SCR=/tmp/claude-1013/-nfs-home-bergia-Projects-SPACEITUP-spaceitup-qwen-fine-tuning/32551702-d421-48f4-b33f-abe7e1fd33e7/scratchpad
git show :README.md > "$SCR/README.index.md"          # the staged version = HEAD (the user's row is unstaged)
# insert the same new section into BOTH $SCR/README.index.md and the working README.md (Edit tool), then:
blob=$(git hash-object -w "$SCR/README.index.md")
git update-index --cacheinfo 100644,"$blob",README.md
git diff --cached --stat                              # README.md only, + the new section's lines only
```

- [ ] **Step 4: Run the whole suite**

Run: `OMP_NUM_THREADS=4 uv run pytest -q`
Expected: all pass (d1's 1056 plus the new tests). Fix any failure before committing; never skip a test.

- [ ] **Step 5: Commit**

```bash
git add scripts/visualize_steering.py tests/test_steering.py     # README.md is already staged index-only
git commit -m "feat: steering report rows per slice; README d2 run book"
git status --short README.md scripts/evaluate.py                  # both still ' M' (the user's changes, unstaged)
```

---

## Run Book (operator steps after the code; every sbatch needs the user's OK first)

CPU steps on the login node: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

1. **Events** (~40 min, 4 workers):
   `uv run scripts/prepare_habitat_instr.py events --labelset-dir data/prepared_habitat_instr_v2 --out-dir data/prepared_habitat_instr_v4`
   Check `events_meta.json`. Expect `spur` ≈ 1.8k train / 212 eval, `no_events_bent` lower than d1's 2,861 / 458,
   and event_types including `turn` ≈ 1.7–1.9k train.
2. **Seed**: `uv run scripts/prepare_habitat_instr.py d2-seed`. Read `seed_meta.json`: `to_compose`,
   `objects_to_name`, `est_compose_hours`.
3. **G1 geometry audit** (human): `uv run scripts/audit_instruction_labels.py --mode turns --events
   data/prepared_habitat_instr_v4/events_train.jsonl --objects data/prepared_habitat_instr_v4/objects_train.jsonl
   --d1-events data/prepared_habitat_instr_v2/events_train.jsonl --n 100`. Publish the page as an artifact with
   the `db` capability (marks) and record the URL in the memory file `visualization-artifacts.md`.
   **Stop until the user's marks are in: ≥ 90% true.**
4. **G2 label gate** (GPU, ask first):
   `ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix sbatch slurm/label_descriptions.sbatch`.
   Pass: accept ≥ 0.90 and template ≤ 0.10, printed by compose pass 2.
5. **G3 label audit** (human, 100 gate rows):
   `--mode d1 --round d2 --events data/prepared_habitat_instr_v4/events_eval_changed.jsonl --objects
   data/prepared_habitat_instr_v4/objects_eval_changed.jsonl --desc outputs/instr_labels_v3/desc_gate_mix_d2.jsonl`.
   Pass at ≥ 90% correct.
6. **Full labels** (GPU, ask first):
   `ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix STAGE=full sbatch --time=12:00:00 slurm/label_descriptions.sbatch`.
7. **Finalize**: `uv run scripts/prepare_habitat_instr.py finalize-descriptions --round d2 --desc-suffix _mix`.
   Check `meta_d2.json` (`recheck_failed` should be ~0).
8. **Train d2** (GPU, ask first): same command and settings as d1's job 92477 (`slurm/train_habitat.sbatch`), with
   data `data/prepared_habitat_instr_v4/train_desc.json` and output `outputs/runs_desc/d2`.
9. **Eval** (GPU, ask first): `DATA=data/prepared_habitat_instr_v4 OUT_DIR=outputs/eval_desc_d2
   ADAPTER=outputs/runs_desc/d2/adapter TAG=d2 sbatch slurm/eval_descriptions.sbatch`, and the same with
   `ADAPTER=outputs/runs_desc/d1/adapter TAG=d1`. Then `uv run scripts/render_desc_videos.py` per set (path videos).
10. **Steering rebuild**:
    - `steer-objects --frames data/prepared_habitat_instr_v4/eval_desc_entry.json --names
      outputs/instr_labels_v3/names_eval_gemma_d2.jsonl --fallback-names outputs/instr_labels_v3/names_eval_d2.jsonl
      --out-dir data/prepared_habitat_instr_v4/steer`
    - `ROUND=d2 STEER_DIR=data/prepared_habitat_instr_v4/steer STAGE=steer_names sbatch …` (GPU, ask)
    - `steer` with the same `--frames`/`--names`/`--fallback-names`/`--out-dir` plus `--steer-names
      outputs/instr_labels_v3/names_steer_gemma_d2.jsonl`
    - `steer-select --out-dir data/prepared_habitat_instr_v4/steer`, which prints Gate 0
    - Gate 1: `--mode steer --n 50 --n-turn 20` on `v4/steer/events_steer_eval.jsonl` (human, ≥ 90%)
    - `ROUND=d2 STEER_DIR=… STAGE=steer sbatch …` (GPU, ask)
    - `finalize-steer --frames … --steer-dir data/prepared_habitat_instr_v4/steer --desc
      outputs/instr_labels_v3/desc_steer_eval_d2.jsonl --out-dir data/prepared_habitat_instr_v4` (Gate 2)
    - `SETS=steer` eval for d1 and d2 (GPU, ask)
11. **Report**: `uv run scripts/visualize_steering.py --tags d1 d2 --eval-dir outputs/eval_desc_d2 --data-dir
    data/prepared_habitat_instr_v4 --out outputs/steering_report_d2.html`, plus the by_turn table from the d1/d2
    eval metrics. Publish it, and add the URL to the memory file `visualization-artifacts.md`.
