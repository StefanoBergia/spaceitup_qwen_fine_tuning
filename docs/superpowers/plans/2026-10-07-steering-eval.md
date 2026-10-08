# Steering Eval Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eval-only set of steering pairs (side + destination slices) from d1's eval frames, with template and Cosmos-composed text, plus the metrics, audit page, report and run scripts to test whether d1 steers.

**Architecture (Revision R1, 2026-10-08, after Addendum A1):** A new pure-numpy module `src/rover_vlm/steering.py` builds side pairs (pass one named object on opposite sides) and destination pairs (two "stop in front of" targets), describes them with the existing `route_events` / `descriptions` code, and keeps only pairs where each route fails the other's description. The builders take a planner: `steer-propose` records what to plan, `scripts/navmesh_plan.py` (the `habitat` env) plans it on the scene navmesh baked at 0.30 m, and `steer` builds the pairs from those routes (unit tests inject Task 1's frame-map A*). Every visible eval object gets a Gemma name first (`steer-objects` + `STAGE=steer_names`). New stages in `scripts/prepare_habitat_instr.py` run objects → propose → steer → select → finalize; the existing compose job adds the natural-language text; `eval.py` gains steering pair metrics.

**Tech Stack:** Python 3.12 (uv `.venv`: numpy, PIL — no scipy), pytest; `habitat` conda env (Python 3.9, habitat_sim 0.3.3) for navmesh planning only (`scripts/navmesh_plan.py`), importing `roverbench` from `../spaceitup-habitat/src`; SLURM sbatch for the Cosmos compose and the Qwen eval.

**Spec:** `docs/superpowers/specs/2026-10-07-steering-eval-design.md` (commit 2d578e3) **and its Addenda A1 (0ea0364) and A2 (7ed516a), which override it (A2 last)**. Read both first.

**Revision R2 (2026-10-08, after Addendum A2):** the set is one slice of **stop pairs** (one ground-truth route cut at two stop points); side/destination builders, the planners and `navmesh_plan.py` are removed in Task 9. Tasks 9–15 below replace R1's Tasks 9–14. Tasks 1–8 stay as built history.

**Revision R1 (2026-10-08):** Tasks 1–5 were built as first planned (frame-map planning) and kept 0 pairs on real frames; Addendum A1 moves planning to the scene navmesh at 0.30 m and names every visible object. Tasks 6–9 are new (they replace the first plan's navmesh-check and selection tasks); the first plan's Tasks 8–12 are now Tasks 10–14.

## Global Constraints

- Thresholds, verbatim from the spec: `CLEAR_M = 0.30` (Addendum A1; was 0.45), `MIN_ROUTE_M = 1.5`, `MIN_SEP_UV = 0.1`, `MIN_DEST_DU = 0.15`, side stop 0.5–1.5 m beyond O's far edge, via at `CLEAR_M + 0.1` m beside O. Eval tolerances unchanged: `STOP_TOL_UV = 0.03` (also report 0.05), `PASS_TIE_M = 0.15`, `FOLLOW_NEAR_M = 2.0`.
- Additions decided while planning (consistent with the spec): `DEST_GAP_M = 0.8` — a destination stop sits 0.8 m before its object, beyond `route_events.PASS_M` (0.7) so the target is not also a "pass" event, inside `STOP_NEAR_M` (1.0) so `stop_phrase` says "in front of"; `SNAP_M = 0.3`; `MIN_AHEAD_M = 1.0`; `INSIDE_FRAC = 0.9`. Extra drop reasons beyond the spec's list: `no_state`, `no_start`, `too_near`, `too_short`, `no_stop`, `anchor_missing`, `unnamed_object`, `target_passed`, `behind_camera`, `error` (`no_free_side` is gone after Task 6: the planner decides). After A1 routes may leave the view and are clipped like d1's; a route passing < 0.3 m in front of the camera is dropped (`behind_camera`).
- Three refinements of the spec's wording, same intent: (1) superseded by Addendum A1: routes are planned on a navmesh baked at agent radius `CLEAR_M` with the generator's own `recompute_navmesh` (CPU, ~1 s per scene), so there is no separate navmesh check; (2) the spec's `steer_anchor` field is the dict `habitat_meta.steer = {slice, anchor, anchor_event, side, viewpoint}`; (3) the side headline ignores a graze (< `PASS_TIE_M`): the d1 checker counts a graze as either side, which would let a text-blind route through the object score both texts.
- **After Addendum A2 (from Task 9 on):** the clearance / via / navmesh constraints above no longer apply (that code is removed). Stop pairs use `MIN_ROUTE_M = 1.5` (a cut keeps at least this much route), `MIN_SEP_UV = 0.1` (answer ends at least this far apart), `CUT_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 1.0)`; one slice, `"stop"`; 4 eval files `eval_steer_stop_{template,composed}_{entry,goal}.json`; Gate 0 ≥ 150 pairs.
- Shared box: before any CPU-heavy run, `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` (pool workers: 1 thread each, 4 workers). Never load Qwen / Cosmos / Gemma on the login node.
- `.venv` has no scipy: numpy + PIL only in `src/`.
- `scripts/navmesh_plan.py` must run under Python 3.9 (`from __future__ import annotations`; no `match`); it must import `habitat_sim` / `roverbench` only inside `main()` so the uv test suite can import it. Run it with `~/miniforge3/envs/habitat/bin/python`, never `conda run -n habitat python` (inside this repo `conda run` resolves the repo's `.python-version` to pyenv's missing 3.12).
- The user's uncommitted edits to `README.md` and `scripts/evaluate.py` are NOT ours: never stage them. README changes go in with an index-only commit (Task 14 shows how). Never touch `scripts/evaluate.py`.
- Every script / module gets a docstring saying what it does and how to run it (CLAUDE.md).
- Commit messages end with a `Co-Authored-By:` trailer naming the model that wrote the commit (controller ruling).
- Tests: `uv run pytest -q` (1014 pass at plan time). Run one file with `uv run pytest tests/test_steering.py -q`.
- Data paths: eval frames from `data/prepared_habitat_instr_v3/eval_desc_entry.json`; names from `outputs/instr_labels_v3/names_eval_gemma_d1.jsonl` + `names_steer_gemma_d1.jsonl` (primary, after A1) and `names_eval_d1.jsonl` (fallback); steering intermediates under `data/prepared_habitat_instr_v3/steer/`; final files `data/prepared_habitat_instr_v3/eval_steer_{side,dest}_{template,composed}_{entry,goal}.json`.

## Review Focus

1. **An entry point that back-projects to no floor point** (v ≤ 0.5): the frame counts `no_start` and the stage goes on — pinned in Task 5 (`test_frame_pairs_counts_no_start`).
2. **A frame that raises inside a pool worker** (corrupt `state.npz`, odd meta): counted as `error` with the message kept, never a dead stage — pinned in Task 5 (`test_steer_frame_turns_an_exception_into_a_reason`).
3. **A frame whose ground-truth candidate is missing or degenerate**: counted `bad_candidate`, never a crash — pinned in Task 9 (`test_frame_pairs_names_and_reasons`). (R1's navmesh item is gone with the navmesh code.)
4. **An eval set cut short** (a record's sibling missing from predictions, or an unparseable prediction): unparseable counts as a miss and stays in the denominator; a pair without both records is skipped — pinned in Task 14.
5. **`finalize-steer` run before the compose job finished**: refuses when > 1% of pairs have no compose row, unless `--allow-missing` — pinned in Task 13.

---

### Task 1: Planner on the frame's map

**Files:**
- Create: `src/rover_vlm/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: `rover_vlm.landmarks.{FREE, OCC, UNKNOWN, Camera, Grid}` (Grid: `.s` uint8 states, `.mpp`, `.rc(xz)` → float (row, col), `.xz(rc)` → world xz of cell centres; Camera: `.uv(xz)` → (uv, fwd), `Camera.in_frame(uv, fwd)`).
- Produces (later tasks use these exact names):
  - `traversable(grid, cam, clear_m=CLEAR_M) -> np.ndarray[bool]` (grid-shaped)
  - `astar(mask, a, b) -> list[tuple[int, int]] | None`
  - `line_clear(mask, a, b) -> bool`
  - `string_pull(mask, cells) -> list[tuple[int, int]]`
  - `snap(grid, mask, xz, max_m=SNAP_M) -> tuple[int, int] | None`
  - `plan_route(grid, mask, waypoints_xz) -> np.ndarray (N, 2) | None`
  - `route_length(xz) -> float`
  - constants `CLEAR_M, SNAP_M`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_steering.py`:

```python
"""Steering eval pairs: planner, pair builders, naming, selection, finalize (synthetic maps).

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (as in test_route_events.py). The image's bottom
edge is 0.8 m ahead of the camera (z = 1.0).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm import steering as st
from rover_vlm.landmarks import FREE, OCC, UNKNOWN, Camera, Grid

CAM = Camera([2.5, 0.2], [0.0, 1.0])


def _state():
    return np.full((100, 100), FREE, np.uint8)


def _box(s, rows, cols):
    s[rows[0]:rows[1], cols[0]:cols[1]] = OCC
    return s


def _grid(s):
    return Grid(s, 0.05, 0.0, 0.0)


def cell(x, z):
    return int(z / 0.05), int(x / 0.05)


def test_traversable_keeps_clearance_unknown_and_frame():
    s = _box(_state(), (50, 58), (46, 54))      # box: cell centres x 2.325..2.675, z 2.525..2.875
    s[:, 90:] = UNKNOWN                          # x >= 4.5 never seen
    m = st.traversable(_grid(s), CAM)
    assert not m[cell(2.5, 2.2)]                 # centre 0.30 m from the box
    assert m[cell(2.5, 2.0)]                     # centre 0.50 m from the box
    assert not m[cell(4.6, 4.0)]                 # unknown
    assert not m[cell(2.5, 0.6)]                 # below the image (0.43 m ahead < 0.8 m)
    assert not m[cell(0.2, 1.5)]                 # outside the 90 degree wedge
    assert m[cell(3.0, 1.5)]


def test_astar_goes_round_an_obstacle_and_fails_when_cut_off():
    s = _box(_state(), (50, 58), (46, 54))
    m = st.traversable(_grid(s), CAM)
    a, b = cell(2.5, 1.2), cell(2.5, 3.6)
    path = st.astar(m, a, b)
    assert path[0] == a and path[-1] == b and all(m[p] for p in path)
    wall = _box(_state(), (50, 52), (0, 100))    # wall across the room at z 2.5
    assert st.astar(st.traversable(_grid(wall), CAM), a, b) is None


def test_string_pull_keeps_ends_and_line_of_sight():
    m = st.traversable(_grid(_box(_state(), (50, 58), (46, 54))), CAM)
    path = st.astar(m, cell(2.5, 1.2), cell(2.5, 3.6))
    pulled = st.string_pull(m, path)
    assert pulled[0] == path[0] and pulled[-1] == path[-1] and len(pulled) < len(path)
    assert all(st.line_clear(m, p, q) for p, q in zip(pulled[:-1], pulled[1:]))


def test_plan_route_keeps_its_via_point_and_needs_a_snap():
    g = _grid(_box(_state(), (50, 58), (46, 54)))
    m = st.traversable(g, CAM)
    route = st.plan_route(g, m, [(2.5, 1.0), (3.25, 2.7), (2.5, 3.4)])
    assert route is not None and route.shape[1] == 2
    assert np.min(np.linalg.norm(route - [3.25, 2.7], axis=1)) < 0.1     # via kept as a corner
    assert route[:, 0].max() > 2.9                                        # went round on +x
    assert st.plan_route(g, m, [(2.5, 1.0), (2.5, 2.7)]) is None         # inside the box: no cell within 0.3 m


def test_route_length():
    assert st.route_length([(0, 0), (3, 4), (3, 5)]) == pytest.approx(6.0)
    assert st.route_length([(1, 1)]) == 0.0
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_steering.py -q`
Expected: FAIL — `ImportError: cannot import name 'steering'`.

- [ ] **Step 3: Write the implementation**

Create `src/rover_vlm/steering.py`:

```python
"""Steering eval pairs: two routes from one entry whose texts differ only in what is tested.

The d1 eval pairs could not test steering (2026-10-07): both routes of every pair end at the
frame's goal and diverge only behind obstacles, so one route satisfied the other's events 94%
of the time. Here routes are planned on the frame's own map (state.npz): known free floor the
camera sees, at least CLEAR_M from every obstacle.

- side slice: same entry and stop, the route passes one named object O on opposite sides;
- destination slice: same entry, two "stop in front of" targets clearly apart in the image.

Every pair is kept only when each route fails the other's description under the eval's own
checker. Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import heapq
import math

import numpy as np

from .landmarks import FREE, OCC, Camera

CLEAR_M = 0.45      # d1 / round-4 GT routes keep 0.44-0.51 m from map obstacles (p10-p75)
SNAP_M = 0.3        # a waypoint may move this far onto the nearest traversable cell
_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_CHUNK = 1024


def traversable(grid, cam, clear_m=CLEAR_M):
    """Cells a route may use: known free, at least clear_m from every occupied cell (centre to
    centre), and in the image (the cell centre projects inside the frame). Unknown cells never
    are. The in-frame floor is convex (a 90 degree wedge beyond the bottom edge), so a straight
    move between two such cells stays in view."""
    rc = np.argwhere(grid.s == FREE)
    ok = np.zeros(grid.s.shape, bool)
    if not len(rc):
        return ok
    uv, fwd = cam.uv(grid.xz(rc))
    keep = Camera.in_frame(uv, fwd)
    occ = np.argwhere(grid.s == OCC).astype(np.int32)
    if len(occ):
        r2 = (clear_m / grid.mpp) ** 2
        rc32 = rc.astype(np.int32)
        for i in range(0, len(rc32), _CHUNK):
            d2 = ((rc32[i:i + _CHUNK, None, :] - occ[None]) ** 2).sum(-1).min(1)
            keep[i:i + _CHUNK] &= d2 >= r2
    ok[rc[keep, 0], rc[keep, 1]] = True
    return ok


def astar(mask, a, b):
    """8-connected shortest cell path a -> b over True cells of `mask`, or None."""
    if not (mask[a] and mask[b]):
        return None
    h_, w_ = mask.shape

    def h(p):
        return math.hypot(p[0] - b[0], p[1] - b[1])

    g = {a: 0.0}
    prev = {}
    heap = [(h(a), 0.0, a)]
    while heap:
        _, gc, p = heapq.heappop(heap)
        if p == b:
            out = [p]
            while p in prev:
                p = prev[p]
                out.append(p)
            return out[::-1]
        if gc > g[p]:
            continue
        for dr, dc in _NEIGH:
            q = (p[0] + dr, p[1] + dc)
            if not (0 <= q[0] < h_ and 0 <= q[1] < w_) or not mask[q]:
                continue
            ng = gc + (math.sqrt(2.0) if dr and dc else 1.0)
            if ng < g.get(q, math.inf):
                g[q] = ng
                prev[q] = p
                heapq.heappush(heap, (ng + h(q), ng, q))
    return None


def line_clear(mask, a, b):
    """True when every cell on the straight segment a -> b (half-cell steps) is in `mask`."""
    n = int(math.ceil(2 * math.hypot(b[0] - a[0], b[1] - a[1]))) + 1
    rr = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cc = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(mask[rr, cc].all())


def string_pull(mask, cells):
    """Corner cells of a cell path: from each kept cell, jump to the farthest later cell in
    straight line of sight on `mask`."""
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_clear(mask, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def snap(grid, mask, xz, max_m=SNAP_M):
    """The traversable cell nearest to world xz when within max_m, else None."""
    cells = np.argwhere(mask)
    if not len(cells):
        return None
    rc = grid.rc(xz)[0]
    d = np.hypot(cells[:, 0] - rc[0], cells[:, 1] - rc[1])
    k = int(d.argmin())
    return (int(cells[k, 0]), int(cells[k, 1])) if d[k] * grid.mpp <= max_m else None


def plan_route(grid, mask, waypoints_xz):
    """World xz corner polyline through the waypoints in order (each snapped onto `mask`):
    one A* leg per consecutive pair, each string-pulled on its own so every waypoint stays a
    corner (a pull across a via point could swap the side of the obstacle it rounds). None
    when a waypoint cannot snap or a leg has no path."""
    cells = [snap(grid, mask, p) for p in waypoints_xz]
    if any(c is None for c in cells):
        return None
    out = []
    for a, b in zip(cells[:-1], cells[1:]):
        leg = astar(mask, a, b)
        if leg is None:
            return None
        leg = string_pull(mask, leg)
        out += leg if not out else leg[1:]
    return grid.xz(np.array(out, float))


def route_length(xz):
    xz = np.asarray(xz, float)
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum()) if len(xz) > 1 else 0.0
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_steering.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py tests/test_steering.py
git commit -m "feat: steering planner — A* on the frame's known free floor, in view, 0.45 m clear

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Describe any route; route → answer path; exclusivity check

**Files:**
- Modify: `src/rover_vlm/route_events.py` (split `describe_route`, around lines 607–657)
- Modify: `src/rover_vlm/steering.py`
- Test: `tests/test_route_events.py`, `tests/test_steering.py`

**Interfaces:**
- Consumes: Task 1; `habitat_data.{clip_polyline_unit, resample_with_transitions}`; `eval.{image_to_floor, description_follow, resample_polyline}`; `descriptions.desc_check(ev, objects)`.
- Produces:
  - `route_events.describe_xz(meta, frame, route) -> dict` — same dict as `describe_route` (`reason, events, stop, objects, goal_xz, pose_xz, heading_xz, route_len_m`, or `_none(reason)`).
  - `steering.entry_floor(meta, start_uv) -> np.ndarray (2,) | None`
  - `steering.answer_path(cam, route_xz) -> list[list]` — `[[x, y, 1], ...]`
  - `steering.separation(path_a, path_b) -> float`
  - `steering.exclusive(ra, rb) -> bool` — `ra`/`rb` are route dicts with `"path"` and `"desc"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_route_events.py` (it already defines `write_sample`, `_stop_state`, `STOP_ROUTE`); add `describe_route, describe_xz` to its `from rover_vlm.route_events import (...)` line:

```python
def test_describe_xz_matches_describe_route(tmp_path):
    sd = write_sample(tmp_path / "s", _stop_state(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    assert describe_xz(meta, frame, np.array(STOP_ROUTE)) == describe_route(sd, 0)
```

Append to `tests/test_steering.py`:

```python
def _desc(side, goal=(2.5, 3.0)):
    """A hand-made describe_xz result: one pass of a post at x 2.5..2.55, z 2.0."""
    return {"reason": None, "pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0], "goal_xz": list(goal),
            "events": [{"type": "pass", "side": side, "objects": ["f:b1"], "arc_m": 1.0}],
            "stop": {"type": "after_last", "relation": "just after", "amount": None, "distance_m": None,
                     "object": None},
            "objects": {"f:b1": {"cells_xz": [[2.5, 2.0], [2.55, 2.0]]}}}


L_XZ = [(2.5, 1.0), (3.3, 2.0), (2.5, 3.0)]        # bows to +x: the rover's left
R_XZ = [(2.5, 1.0), (1.7, 2.0), (2.5, 3.0)]


def test_entry_floor_inverts_the_camera():
    meta = {"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0]}
    assert st.entry_floor(meta, [0.5, 1.0]) == pytest.approx([2.5, 1.0])
    assert st.entry_floor(meta, [0.5, 0.4]) is None             # at or above the horizon


def test_answer_path_projects_a_fully_visible_floor_route():
    path = st.answer_path(CAM, L_XZ)
    assert path[0][:2] == pytest.approx([0.5, 1.0], abs=1e-3)
    assert all(w[2] == 1 for w in path) and 3 <= len(path) <= 12
    assert path[1][0] < 0.5                                      # +x is image-left


def test_separation():
    a, b = st.answer_path(CAM, L_XZ), st.answer_path(CAM, R_XZ)
    assert st.separation(a, a) == 0.0
    assert st.separation(a, b) > 0.1


def test_exclusive_needs_each_route_to_fail_the_other_text():
    rl = {"path": st.answer_path(CAM, L_XZ), "desc": _desc("left")}
    rr = {"path": st.answer_path(CAM, R_XZ), "desc": _desc("right")}
    assert st.exclusive(rl, rr)
    assert not st.exclusive(rl, {**rr, "desc": _desc("left")})     # R fails its own text
    assert not st.exclusive(rl, {**rl, "desc": _desc("left")})     # one route satisfies both
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_route_events.py::test_describe_xz_matches_describe_route tests/test_steering.py -q`
Expected: FAIL — `ImportError: cannot import name 'describe_xz'` and `AttributeError: module 'rover_vlm.steering' has no attribute 'entry_floor'`.

- [ ] **Step 3: Write the implementation**

In `src/rover_vlm/route_events.py`, replace the body of `describe_route` after the `bad_candidate` check so it delegates, and move the rest into `describe_xz` (no behaviour change):

```python
def describe_route(sample_dir, candidate_index):
    """The events + stop phrase for one candidate route of one frame (see module docstring)."""
    sd = Path(sample_dir)
    if not (sd / "state.npz").exists():
        return _none("no_state")
    meta, frame = load_frame(sd)
    cm = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cm)) or len(cm[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    return describe_xz(meta, frame, cm[candidate_index]["polyline_xz"])


def describe_xz(meta, frame, route):
    """The events + stop phrase for any route polyline (world xz) on a loaded frame
    (steering pairs describe routes that are not candidates of the frame)."""
    route = np.asarray(route, float)
    cands = candidates(frame, route)
    events = build_events(frame, cands)
    # ... the rest of the former describe_route body, unchanged, from
    # `if not events and route_geometry(route)["offset"].max() > STRAIGHT_MAX_M:`
    # down to its final `return {"reason": None, ...}`
```

(Move the code verbatim — every line from `if not events and ...` to the final `return`. Only the two signatures and the two lines above it change.)

Append to `src/rover_vlm/steering.py` (add the imports at the top of the file):

```python
from .descriptions import desc_check
from .eval import description_follow, image_to_floor, resample_polyline
from .habitat_data import clip_polyline_unit, resample_with_transitions

SEP_POINTS = 50     # resampling for the visible-separation test


def entry_floor(meta, start_uv):
    """Floor xz of the frame's entry point (level camera), or None at/above the horizon."""
    xz = image_to_floor([[start_uv[0], start_uv[1], 1]], meta["pose_xz"], meta["heading_xz"])
    return xz[0] if len(xz) else None


def answer_path(cam, route_xz):
    """[x, y, 1] waypoints of a fully visible floor route, built as the d1 answers are (clip to
    the unit square, resample with transitions). Floor lines project to image lines, so the
    route's corners are enough."""
    uv, _ = cam.uv(np.asarray(route_xz, float))
    clipped = clip_polyline_unit([(float(u), float(v), False) for u, v in uv])
    return [list(w) for w in resample_with_transitions(clipped)]


def separation(path_a, path_b):
    """Largest image distance between the two paths, both resampled by arclength."""
    a = resample_polyline(np.array([p[:2] for p in path_a], float), SEP_POINTS)
    b = resample_polyline(np.array([p[:2] for p in path_b], float), SEP_POINTS)
    return float(np.linalg.norm(a - b, axis=1).max())


def exclusive(ra, rb):
    """Each route follows its own description and fails the other's (eval.description_follow,
    stop judged against each description's own route end)."""
    ca, cb = desc_check(ra["desc"], ra["desc"]["objects"]), desc_check(rb["desc"], rb["desc"]["objects"])
    ea, eb = ra["path"][-1], rb["path"][-1]
    return (description_follow(ra["path"], ca, ea)["follows"] == 1
            and description_follow(rb["path"], cb, eb)["follows"] == 1
            and description_follow(ra["path"], cb, eb)["follows"] == 0
            and description_follow(rb["path"], ca, ea)["follows"] == 0)
```

(`eval.py` does not import `steering`, so there is no import cycle.)

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_route_events.py tests/test_steering.py -q`
Expected: all pass (the existing route_events tests too — `describe_route` behaves as before).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py src/rover_vlm/steering.py tests/test_route_events.py tests/test_steering.py
git commit -m "feat: describe_xz for any route; steering answer paths, separation, exclusivity

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Side pairs

**Files:**
- Modify: `src/rover_vlm/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: Tasks 1–2; `route_events.{_blob_cells, describe_xz, load_frame, MIN_VISIBLE_CELLS}`; `landmarks.densify`.
- Produces:
  - `enclosed_blobs(frame, a_xz, b_xz) -> dict[int, int]` (blob → cells inside the loop a + reversed b)
  - `two_sided(frame, a_xz, b_xz, blob) -> bool`
  - `side_difference(ev_l, ev_r, oid) -> tuple[int | None, str | None]`
  - `stop_key(stop) -> tuple`
  - `side_pair(frame, meta, mask, start, blob) -> tuple[dict | None, str | None]`
  - **Route dict** (used by every later task): `{"tag": "L"|"R"|"A"|"B", "side": "left"|"right"|None, "route_xz": [[x, z], ...], "path": [[x, y, 1], ...], "desc": <describe_xz dict>, "anchor_event": int|None, "target": oid|None}`
  - **Pair dict**: `{"slice": "side"|"dest", "anchor": oid|None, "targets": [oid, oid]|None, "separation": float, "routes": [route, route]}`
  - constants `MIN_ROUTE_M, MIN_SEP_UV, VIA_EXTRA_M, STOP_BEYOND_M, MIN_AHEAD_M, INSIDE_FRAC`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering.py`:

```python
from rover_vlm.route_events import load_frame


def _sample(d, state):
    """A synthetic sample dir: state.npz (with floor_height) and meta.json, no candidates."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0, floor_height=0.1)
    (d / "meta.json").write_text(json.dumps({"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0],
                                             "flavor": "detour", "chord": {"crossing_xz": None},
                                             "candidates": []}))
    return d


START = np.array([2.5, 1.0])
BOX = ((50, 58), (46, 54))                      # 0.4 m box 1.5-1.9 m ahead of the entry


def _frame(tmp_path, state, name="f"):
    meta, frame = load_frame(_sample(tmp_path / name, state))
    return meta, frame, st.traversable(frame.grid, frame.cam)


def test_side_pair_rounds_both_sides_one_word_apart(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    pair, why = st.side_pair(frame, meta, mask, START, blob)
    assert why is None and pair["slice"] == "side" and pair["anchor"] == f"f:b{blob}"
    l, r = pair["routes"]
    assert (l["tag"], l["side"], r["tag"], r["side"]) == ("L", "left", "R", "right")
    assert l["route_xz"][0] == r["route_xz"][0] and l["path"][0] == r["path"][0]     # shared entry
    i = l["anchor_event"]
    assert i == r["anchor_event"]
    assert l["desc"]["events"][i]["side"] == "left" and r["desc"]["events"][i]["side"] == "right"
    assert st.stop_key(l["desc"]["stop"]) == st.stop_key(r["desc"]["stop"])
    assert pair["separation"] >= st.MIN_SEP_UV


def test_side_pair_rejects_a_second_obstacle_between_the_routes(tmp_path):
    s = _box(_state(), *BOX)
    s[62:64, 49:51] = OCC                        # post 0.25 m behind the box, between both routes
    meta, frame, mask = _frame(tmp_path, s)
    pair, why = st.side_pair(frame, meta, mask, START, int(frame.lab[54, 50]))
    assert pair is None and why == "not_two_sided"


def test_two_routes_round_the_same_side_are_not_two_sided(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    a = st.plan_route(frame.grid, mask, [START, (3.25, 2.7), (2.5, 3.4)])
    b = st.plan_route(frame.grid, mask, [START, (3.7, 2.7), (2.5, 3.4)])
    assert not st.two_sided(frame, a, b, blob)
    c = st.plan_route(frame.grid, mask, [START, (1.75, 2.7), (2.5, 3.4)])
    assert st.two_sided(frame, a, c, blob)


def test_anchor_too_close_to_the_entry(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), (26, 30), (48, 52)))   # 0.3-0.5 m ahead
    pair, why = st.side_pair(frame, meta, mask, START, int(frame.lab[28, 50]))
    assert pair is None and why == "too_near"


def _ev(t, side, objs):
    return {"type": t, "side": side, "objects": objs, "arc_m": 1.0}


def test_side_difference():
    o = "f:b1"
    wall_l, wall_r = _ev("wall", "left", ["f:b9"]), _ev("wall", "left", ["f:b9"])
    assert st.side_difference([wall_l, _ev("pass", "left", [o])], [wall_r, _ev("pass", "right", [o])], o) == (1, None)
    assert st.side_difference([_ev("pass", "left", [o])], [_ev("pass", "left", [o])], o) == (None, "anchor_missing")
    assert st.side_difference([_ev("pass", "left", [o, "f:b2"])], [_ev("pass", "right", [o])], o) == (None, "anchor_missing")
    assert st.side_difference([wall_l, _ev("pass", "left", [o])], [_ev("pass", "right", [o])], o) == (None, "events_differ")
    assert st.side_difference([_ev("pass", "left", [o]), _ev("pass", "left", ["f:b2"])],
                              [_ev("pass", "right", [o]), _ev("pass", "right", ["f:b2"])], o) == (None, "events_differ")
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_steering.py -q`
Expected: FAIL — `AttributeError: module 'rover_vlm.steering' has no attribute 'side_pair'` (and the other new names).

- [ ] **Step 3: Write the implementation**

Append to `src/rover_vlm/steering.py` (add to the imports: `from PIL import Image, ImageDraw`, `from .landmarks import densify`, `from .route_events import MIN_VISIBLE_CELLS, _blob_cells, describe_xz`):

```python
MIN_ROUTE_M = 1.5
MIN_SEP_UV = 0.1
VIA_EXTRA_M = 0.1                 # via point: CLEAR_M + this beyond O's side
STOP_BEYOND_M = (0.5, 1.0, 1.5)   # shared stop: this far past O's far edge, first that plans
MIN_AHEAD_M = 1.0                 # an anchor nearer the entry leaves no room to go round it
INSIDE_FRAC = 0.9                 # share of O's cells that must lie between the two routes


def _rounded(xz):
    return [[round(float(x), 3), round(float(z), 3)] for x, z in np.asarray(xz, float)]


def enclosed_blobs(frame, a_xz, b_xz):
    """{blob: cells inside the closed loop a + reversed(b)} (PIL even-odd polygon fill on the
    grid): two routes with the same ends differ in which way they go round exactly the blobs
    between them."""
    mpp = frame.grid.mpp
    pts = np.vstack([densify(np.asarray(a_xz, float), mpp / 2), densify(np.asarray(b_xz, float), mpp / 2)[::-1]])
    rc = frame.grid.rc(pts)
    img = Image.new("1", (frame.lab.shape[1], frame.lab.shape[0]), 0)
    ImageDraw.Draw(img).polygon([(float(c), float(r)) for r, c in rc], fill=1)
    inside = np.asarray(img, bool) & (frame.lab > 0)
    labs, counts = np.unique(frame.lab[inside], return_counts=True)
    return {int(b): int(n) for b, n in zip(labs, counts)}


def two_sided(frame, a_xz, b_xz, blob):
    """True when the two routes go round `blob` on opposite sides and round nothing else:
    at least INSIDE_FRAC of its cells lie between them, and no other blob with
    MIN_VISIBLE_CELLS or more cells does."""
    inside = enclosed_blobs(frame, a_xz, b_xz)
    n = len(_blob_cells(frame)[blob][0])
    return (inside.get(blob, 0) >= INSIDE_FRAC * n
            and not any(b != blob and k >= MIN_VISIBLE_CELLS for b, k in inside.items()))


def _event_key(e):
    return e["type"], e["side"], tuple(e["objects"])


def side_difference(ev_l, ev_r, oid):
    """(index of O's pass event, None) when the two event lists are identical except that the
    left route passes O alone on the left and the right route on the right; else
    (None, "anchor_missing" | "events_differ")."""
    def at(events, side):
        return [i for i, e in enumerate(events) if e["type"] == "pass" and e["objects"] == [oid] and e["side"] == side]

    il, ir = at(ev_l, "left"), at(ev_r, "right")
    if len(il) != 1 or len(ir) != 1:
        return None, "anchor_missing"
    if il != ir or len(ev_l) != len(ev_r):
        return None, "events_differ"
    if any(_event_key(a) != _event_key(b) for i, (a, b) in enumerate(zip(ev_l, ev_r)) if i != il[0]):
        return None, "events_differ"
    return il[0], None


def stop_key(stop):
    return stop["type"], stop["relation"], stop.get("amount"), stop.get("object")


def side_routes(frame, mask, start, blob):
    """({"left": xz, "right": xz}, None): one route each way round `blob` to a shared stop past
    it; else (None, reason). A via on the route's right of O means the route goes RIGHT of O
    (route_events' pass side = the route's side)."""
    xz = _blob_cells(frame)[blob][1]
    c = xz.mean(0)
    d = c - start
    dist = float(np.linalg.norm(d))
    if dist < MIN_AHEAD_M:
        return None, "too_near"
    d /= dist
    right = np.array([-d[1], d[0]])              # Camera.right's handedness
    lat = (xz - c) @ right
    via = {"right": c + right * (lat.max() + CLEAR_M + VIA_EXTRA_M),
           "left": c + right * (lat.min() - CLEAR_M - VIA_EXTRA_M)}
    if any(snap(frame.grid, mask, v) is None for v in via.values()):
        return None, "no_free_side"
    far = start + d * float(((xz - start) @ d).max())
    for k in STOP_BEYOND_M:
        stop = far + d * k
        routes = {s: plan_route(frame.grid, mask, [start, via[s], stop]) for s in ("left", "right")}
        if all(r is not None for r in routes.values()):
            return routes, None
    return None, "plan_failed"


def side_pair(frame, meta, mask, start, blob):
    """(pair dict, None) for the side slice around one blob, else (None, reason)."""
    routes, why = side_routes(frame, mask, start, blob)
    if routes is None:
        return None, why
    if min(route_length(r) for r in routes.values()) < MIN_ROUTE_M:
        return None, "too_short"
    if not two_sided(frame, routes["left"], routes["right"], blob):
        return None, "not_two_sided"
    desc = {s: describe_xz(meta, frame, r) for s, r in routes.items()}
    if any(d["reason"] for d in desc.values()):
        return None, "no_stop"
    oid = f"{frame.name}:b{blob}"
    idx, why = side_difference(desc["left"]["events"], desc["right"]["events"], oid)
    if why:
        return None, why
    if stop_key(desc["left"]["stop"]) != stop_key(desc["right"]["stop"]):
        return None, "events_differ"
    out = []
    for tag, s in (("L", "left"), ("R", "right")):
        path = answer_path(frame.cam, routes[s])
        if len(path) < 2:
            return None, "too_short"
        out.append({"tag": tag, "side": s, "route_xz": _rounded(routes[s]), "path": path,
                    "desc": desc[s], "anchor_event": idx, "target": None})
    sep = separation(out[0]["path"], out[1]["path"])
    if sep < MIN_SEP_UV:
        return None, "too_close"
    return {"slice": "side", "anchor": oid, "targets": None, "separation": round(sep, 4), "routes": out}, None
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_steering.py -q`
Expected: all pass. If `test_side_pair_rounds_both_sides_one_word_apart` fails on the stop (A* tie-breaking can make the two routes slightly asymmetric), print both `desc["stop"]` dicts and adjust only the synthetic scene (e.g. stop distance) — not the rule.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py tests/test_steering.py
git commit -m "feat: side steering pairs — round one named object both ways, texts one word apart

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Destination pairs

**Files:**
- Modify: `src/rover_vlm/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `dest_route(frame, meta, mask, start, blob) -> tuple[route | None, str | None]` (route dict with `target` set, `tag` None)
  - `dest_pairs(routes) -> tuple[list[pair], Counter]` — tags `A` (smaller end x, image-left) / `B`
  - constants `MIN_DEST_DU = 0.15`, `DEST_GAP_M = 0.8`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering.py`:

```python
def _dest_state():
    s = _state()
    s[66:74, 66:74] = OCC        # box A: x 3.3..3.7, z 3.3..3.7 (image-left)
    s[66:74, 26:34] = OCC        # box B: x 1.3..1.7 (image-right)
    return s


def test_dest_route_stops_in_front_of_its_object(tmp_path):
    meta, frame, mask = _frame(tmp_path, _dest_state())
    blob = int(frame.lab[70, 70])
    r, why = st.dest_route(frame, meta, mask, START, blob)
    assert why is None and r["target"] == f"f:b{blob}"
    stop = r["desc"]["stop"]
    assert (stop["type"], stop["relation"], stop["object"]) == ("near", "in front of", f"f:b{blob}")
    assert f"f:b{blob}" not in {o for e in r["desc"]["events"] for o in e["objects"]}   # not also a pass
    assert r["path"][-1][0] < 0.5


def test_dest_pairs_tags_by_image_x_and_drops_close_endpoints(tmp_path):
    meta, frame, mask = _frame(tmp_path, _dest_state())
    ra, _ = st.dest_route(frame, meta, mask, START, int(frame.lab[70, 70]))
    rb, _ = st.dest_route(frame, meta, mask, START, int(frame.lab[70, 30]))
    pairs, reasons = st.dest_pairs([rb, ra])
    assert len(pairs) == 1 and not reasons
    p = pairs[0]
    assert [r["tag"] for r in p["routes"]] == ["A", "B"]
    assert p["routes"][0]["path"][-1][0] < p["routes"][1]["path"][-1][0]
    assert p["targets"] == [ra["target"], rb["target"]] and p["slice"] == "dest"
    assert st.exclusive(*p["routes"])
    near_twin = {**ra, "path": [w[:] for w in ra["path"]]}
    near_twin["path"][-1][0] += 0.1                                  # endpoints 0.1 apart in x
    pairs, reasons = st.dest_pairs([ra, near_twin])
    assert pairs == [] and reasons["too_close"] == 1


def test_dest_route_too_short_and_wrong_stop(tmp_path, monkeypatch):
    s = _state()
    s[46:50, 48:52] = OCC                        # small box 1.3-1.5 m ahead: target 0.5 m from the entry
    meta, frame, mask = _frame(tmp_path, s)
    r, why = st.dest_route(frame, meta, mask, START, int(frame.lab[48, 50]))
    assert r is None and why == "too_short"
    meta, frame, mask = _frame(tmp_path, _dest_state(), name="g")
    monkeypatch.setattr(st, "describe_xz", lambda m, f, route: {
        "reason": None, "events": [], "stop": {"type": "near", "relation": "in front of", "object": "g:b999"}})
    r, why = st.dest_route(frame, meta, mask, START, int(frame.lab[70, 70]))
    assert r is None and why == "no_stop"
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_steering.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'dest_route'`.

- [ ] **Step 3: Write the implementation**

Append to `src/rover_vlm/steering.py` (add `import itertools` and `from collections import Counter` to the imports):

```python
MIN_DEST_DU = 0.15   # destination endpoints at least this far apart in image x
# A destination stop sits this far before its object: beyond route_events.PASS_M (0.7), so the
# target is not also a "pass" event of its own route, and inside STOP_NEAR_M (1.0), so the
# stop phrase is "in front of" that object.
DEST_GAP_M = 0.8


def dest_route(frame, meta, mask, start, blob):
    """(route dict, None) from the entry to DEST_GAP_M in front of `blob`, whose stop phrase is
    "in front of" that blob; else (None, reason)."""
    xz = _blob_cells(frame)[blob][1]
    dd = np.linalg.norm(xz - start, axis=1)
    if dd.min() < MIN_AHEAD_M:
        return None, "too_near"
    q = xz[dd.argmin()]
    d = (q - start) / dd.min()
    route = plan_route(frame.grid, mask, [start, q - d * DEST_GAP_M])
    if route is None:
        return None, "plan_failed"
    if route_length(route) < MIN_ROUTE_M:
        return None, "too_short"
    desc = describe_xz(meta, frame, route)
    oid = f"{frame.name}:b{blob}"
    stop = desc.get("stop") or {}
    if desc["reason"] or stop.get("type") != "near" or stop.get("relation") != "in front of" \
            or stop.get("object") != oid:
        return None, "no_stop"
    path = answer_path(frame.cam, route)
    if len(path) < 2:
        return None, "too_short"
    return {"tag": None, "side": None, "route_xz": _rounded(route), "path": path, "desc": desc,
            "anchor_event": None, "target": oid}, None


def dest_pairs(routes):
    """Every two destination routes whose endpoints are MIN_DEST_DU apart in image x, tagged
    A (image-left end) / B; (pairs, reasons)."""
    pairs, reasons = [], Counter()
    for a, b in itertools.combinations(routes, 2):
        a, b = sorted((a, b), key=lambda r: r["path"][-1][0])
        if b["path"][-1][0] - a["path"][-1][0] < MIN_DEST_DU:
            reasons["too_close"] += 1
            continue
        ra, rb = {**a, "tag": "A"}, {**b, "tag": "B"}
        pairs.append({"slice": "dest", "anchor": None, "targets": [ra["target"], rb["target"]],
                      "separation": round(separation(ra["path"], rb["path"]), 4), "routes": [ra, rb]})
    return pairs, reasons
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_steering.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py tests/test_steering.py
git commit -m "feat: destination steering pairs — two 'stop in front of' targets apart in image x

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Names, texts, per-frame candidates, and the `steer` stage

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (`choose_names`, used by `label_route`)
- Modify: `src/rover_vlm/steering.py`
- Modify: `scripts/prepare_habitat_instr.py` (docstring, imports, `stage_steer`, subparser)
- Test: `tests/test_descriptions.py`, `tests/test_steering.py`

**Interfaces:**
- Consumes: Tasks 1–4; `descriptions.{name_rows_split, route_objects, template_description, _usable}`; `instructions.head_noun`; `route_events.{WALL_EXTENT_M, load_frame}`.
- Produces:
  - `descriptions.choose_names(row, names, fallback_names=None) -> (dict | None, "primary"|"fallback"|None, None|"unnamed_object"|"name:clash")`
  - `steering.load_names(primary_rows, fallback_rows) -> (names, fallback, kinds)`
  - `steering.frame_clash(oid, names) -> bool`
  - `steering.name_pair(pair, names, fallback, kinds) -> str | None` — on success adds `namer`, `names` (route object → name), `template` to each route dict
  - `steering.anchor_blobs(frame, names, fallback, kinds) -> list[int]`
  - `steering.frame_pairs(sample_dir, start_uv, names, fallback, kinds) -> (list[pair], Counter)` — each pair has `pair_id`; reasons keyed `"side:<r>"`, `"dest:<r>"`, `"<slice>:kept"`, or unprefixed frame-level `no_state` / `no_start` / `no_named_object`
  - `steering.steer_frames(records) -> list[dict]` — items `{sample_id, sample_dir, image, start_uv, scene_id, viewpoint, is_aug}`
  - `steering.floor_height(sample_dir) -> float | None`
  - Stage `prepare_habitat_instr.py steer` → `data/prepared_habitat_instr_v3/steer/candidates_eval.jsonl` (one pair per line, plus `sample_id, viewpoint, is_aug, scene_id, image, sample_dir, floor_y`) and `steer/steer_meta.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_descriptions.py`:

```python
def test_choose_names_primary_fallback_and_rejects():
    from rover_vlm.descriptions import choose_names
    row = _route([PASS_L], NEAR)                                       # f:b1, stop f:b5
    assert choose_names(row, NAMES) == (NAMES, "primary", None)
    assert choose_names(row, {"f:b5": "fridge"}, NAMES) == (NAMES, "fallback", None)
    assert choose_names(row, {"f:b5": "fridge"}, {"f:b5": "fridge"}) == (None, None, "unnamed_object")
    clash = {**NAMES, "f:b1": "fridge"}
    assert choose_names(row, clash) == (None, None, "name:clash")
```

Append to `tests/test_steering.py`:

```python
def test_frame_clash_is_within_one_frame():
    names = {"f:b1": "grey sofa", "f:b2": "leather sofa", "g:b1": "sofa", "f:b3": "lamp"}
    assert st.frame_clash("f:b1", names)
    assert not st.frame_clash("f:b3", names)
    assert not st.frame_clash("g:b1", names)


def test_load_names_splits_files_and_keeps_kinds():
    prim = [{"id": "f:b1", "passed": True, "name": "sofa", "need": "object"},
            {"id": "f:b2", "passed": False, "name": None, "need": "wall"}]
    fb = [{"id": "f:b2", "passed": True, "name": "white wall", "need": "wall"}]
    names, fallback, kinds = st.load_names(prim, fb)
    assert names == {"f:b1": "sofa"} and fallback == {"f:b2": "white wall"}
    assert kinds == {"f:b1": "object", "f:b2": "wall"}


def test_side_pair_named_gets_one_word_apart_templates(tmp_path):
    meta, frame, mask = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    pair, _ = st.side_pair(frame, meta, mask, START, blob)
    assert st.name_pair(pair, {f"f:b{blob}": "armchair"}, None, {}) is None
    l, r = pair["routes"]
    assert l["namer"] == "primary" and "left of the armchair" in l["template"]
    assert l["template"].replace("left", "right") == r["template"]
    assert st.name_pair(pair, {}, None, {}) == "unnamed_object"


def test_anchor_blobs_and_frame_pairs_on_named_frames(tmp_path):
    # Two separate scenes: with the side box and the two destination boxes in ONE room, the
    # side routes pass within PASS_M of a destination box (an extra, one-sided event) and the
    # destination targets have the side box nearer than their own object (wrong stop anchor).
    sd = _sample(tmp_path / "f", _box(_state(), *BOX))
    _, frame = load_frame(sd)
    o = int(frame.lab[54, 50])
    names = {f"f:b{o}": "ottoman"}
    assert st.anchor_blobs(frame, names, None, {}) == [o]
    assert st.anchor_blobs(frame, names, None, {f"f:b{o}": "wall"}) == []
    assert st.anchor_blobs(frame, {}, names, {}) == [o]                     # fallback names count too
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], names, None, {})
    assert [p["pair_id"] for p in pairs] == [f"steer_side_f_b{o}"]
    assert reasons["side:kept"] == 1 and reasons["dest:too_short"] == 1  # target 0.7 m from the entry
    sd = _sample(tmp_path / "g", _dest_state())
    _, frame = load_frame(sd)
    a, b = int(frame.lab[70, 70]), int(frame.lab[70, 30])                  # a = image-left box
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], {f"g:b{a}": "armchair", f"g:b{b}": "sofa"}, None, {})
    dest = [p for p in pairs if p["slice"] == "dest"]
    assert [p["pair_id"] for p in dest] == [f"steer_dest_g_b{a}_b{b}"] and reasons["dest:kept"] == 1
    json.dumps(pairs)                                                        # JSON-ready


def test_frame_pairs_counts_no_start_and_no_names(tmp_path):
    sd = _sample(tmp_path / "f", _dest_state())
    _, reasons = st.frame_pairs(sd, [0.5, 0.4], {"f:b1": "x"}, None, {})
    assert reasons == {"no_start": 1}
    _, reasons = st.frame_pairs(sd, [0.5, 1.0], {}, None, {})
    assert reasons == {"no_named_object": 1}
    _, reasons = st.frame_pairs(tmp_path / "missing", [0.5, 1.0], {}, None, {})
    assert reasons == {"no_state": 1}


def test_steer_frames_one_item_per_frame_gt_record_first():
    def rec(rid, sid, kind, src="original", parent=None, start=(0.5, 1.0)):
        return {"id": rid, "image": [f"/d/{sid}/fpv_enhanced.png"],
                "habitat_meta": {"sample_id": sid, "kind": kind, "source": src, "parent_sample_id": parent,
                                 "start_uv_norm": list(start), "scene_id": "sc"}}
    items = st.steer_frames([rec("a_R", "a", "opposite", start=(0.4, 1.0)), rec("a_L", "a", "instructed"),
                             rec("b_y0_L", "b_y0", "instructed", "augmented", "b")])
    assert [i["sample_id"] for i in items] == ["a", "b_y0"]
    assert items[0]["start_uv"] == [0.5, 1.0] and items[0]["viewpoint"] == "a" and not items[0]["is_aug"]
    assert items[1]["viewpoint"] == "b" and items[1]["is_aug"] and items[1]["sample_dir"] == "/d/b_y0"


def test_floor_height(tmp_path):
    assert st.floor_height(_sample(tmp_path / "f", _state())) == pytest.approx(0.1)
    assert st.floor_height(tmp_path / "missing") is None
```

Also add a stage-level test that a worker exception becomes a reason (Review Focus 2). Append to `tests/test_steering.py`:

```python
def test_steer_frame_turns_an_exception_into_a_reason(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)

    def boom(*a, **k):
        raise ValueError("bad map")
    monkeypatch.setattr(prep, "frame_pairs", boom)
    prep._steer_init({}, None, {})
    pairs, reasons, err = prep.steer_frame({"sample_id": "x", "sample_dir": "/nope", "start_uv": [0.5, 1.0]})
    assert pairs == [] and reasons == {"error": 1} and "ValueError: bad map" in err
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_descriptions.py::test_choose_names_primary_fallback_and_rejects tests/test_steering.py -q`
Expected: FAIL — `ImportError: cannot import name 'choose_names'` / missing steering attributes / `steer_frame`.

- [ ] **Step 3: Write `choose_names` and use it in `label_route`**

In `src/rover_vlm/descriptions.py`, add above `label_route`:

```python
def choose_names(row, names, fallback_names=None):
    """The A5 naming rule for one route: (names, "primary", None) when `names` serves it (every
    route object named, no two sharing a head noun), else (fallback_names, "fallback", None)
    under the same test, else (None, None, "unnamed_object" | "name:clash")."""
    if _usable(row, names):
        return names, "primary", None
    if fallback_names is not None and _usable(row, fallback_names):
        return fallback_names, "fallback", None
    tried = [names] + ([fallback_names] if fallback_names is not None else [])
    if any(not any(n.get(o) for n in tried) for o in route_objects(row)):
        return None, None, "unnamed_object"
    return None, None, "name:clash"
```

and replace the naming block at the top of `label_route` (from `if _usable(row, names):` through the `name:clash` return) with:

```python
    chosen, namer, reject = choose_names(row, names, fallback_names)
    if chosen is None:
        if reject == "unnamed_object":
            tried = [names] + ([fallback_names] if fallback_names is not None else [])
            missing = [o for o in route_objects(row) if not any(n.get(o) for n in tried)]
            return {**out, "passed": False, "description": None, "instruction_source": None,
                    "rejects": ["unnamed_object"], "missing": missing}
        return {**out, "passed": False, "description": None, "instruction_source": None,
                "rejects": ["name:clash"]}
    names = chosen
```

- [ ] **Step 4: Write the steering naming + frame functions**

Append to `src/rover_vlm/steering.py` (imports: `from pathlib import Path`; `from .descriptions import choose_names, name_rows_split, route_objects, template_description`; `from .instructions import head_noun`; `from .route_events import WALL_EXTENT_M, load_frame`):

```python
def load_names(primary_rows, fallback_rows):
    """(names, fallback, kinds) from the two name-row files of the mixed namer (A5): Gemma rows
    are primary, Cosmos rows the fallback; kinds = each object's "need" (wall / object / opening)."""
    names, _ = name_rows_split(primary_rows)
    fallback, _ = name_rows_split(fallback_rows)
    kinds = {r["id"]: r.get("need") for r in primary_rows}
    for r in fallback_rows:
        kinds.setdefault(r["id"], r.get("need"))
    return names, fallback, kinds


def frame_clash(oid, names):
    """True when another object of the same frame has a name with the same head noun."""
    frame, h = oid.split(":")[0], head_noun(names.get(oid))
    return bool(h) and any(o != oid and o.split(":")[0] == frame and head_noun(n) == h
                           for o, n in names.items())


def name_pair(pair, names, fallback, kinds):
    """Pick each route's names (A5 rule), refuse anchors whose name is ambiguous in the frame and
    destination targets sharing a head noun, and write each route's template sentence. Returns
    the drop reason, or None after adding namer / names / template to both route dicts."""
    heads = []
    for r in pair["routes"]:
        row = {"events": r["desc"]["events"], "stop": r["desc"]["stop"]}
        chosen, namer, reject = choose_names(row, names, fallback)
        if chosen is None:
            return "unnamed_object" if reject == "unnamed_object" else "name_clash"
        anchor = pair["anchor"] or r["target"]
        if frame_clash(anchor, chosen):
            return "name_clash"
        heads.append(head_noun(chosen[anchor]))
        r.update(namer=namer, names={o: chosen[o] for o in route_objects(row)},
                 template=template_description(row["events"], row["stop"], chosen, kinds))
    if pair["slice"] == "dest" and heads[0] == heads[1]:
        return "name_clash"
    return None


def anchor_blobs(frame, names, fallback, kinds):
    """Blobs that may anchor a pair: named (either file), not used as a wall, shorter than
    WALL_EXTENT_M."""
    out = []
    for b, (_, xz) in sorted(_blob_cells(frame).items()):
        oid = f"{frame.name}:b{b}"
        if not (names.get(oid) or (fallback or {}).get(oid)) or kinds.get(oid) == "wall":
            continue
        if float(np.hypot(*np.ptp(xz, axis=0))) >= WALL_EXTENT_M:
            continue
        out.append(int(b))
    return out


def frame_pairs(sample_dir, start_uv, names, fallback, kinds):
    """Every candidate pair of one frame, both slices, named and mutually exclusive, with the
    drop reasons counted as "<slice>:<reason>" (frame-level reasons unprefixed)."""
    sd = Path(sample_dir)
    reasons = Counter()
    if not (sd / "state.npz").exists():
        reasons["no_state"] += 1
        return [], reasons
    meta, frame = load_frame(sd)
    mask = traversable(frame.grid, frame.cam)
    start = entry_floor(meta, start_uv)
    if start is None or snap(frame.grid, mask, start) is None:
        reasons["no_start"] += 1
        return [], reasons
    blobs = anchor_blobs(frame, names, fallback, kinds)
    if not blobs:
        reasons["no_named_object"] += 1
        return [], reasons
    pairs = []

    def keep(pair):
        why = name_pair(pair, names, fallback, kinds) or (None if exclusive(*pair["routes"]) else "not_exclusive")
        reasons[f"{pair['slice']}:{why or 'kept'}"] += 1
        if why is None:
            pairs.append(pair)

    for b in blobs:
        pair, why = side_pair(frame, meta, mask, start, b)
        if pair is None:
            reasons[f"side:{why}"] += 1
        else:
            pair["pair_id"] = f"steer_side_{frame.name}_b{b}"
            keep(pair)
    routes = []
    for b in blobs:
        r, why = dest_route(frame, meta, mask, start, b)
        if r is None:
            reasons[f"dest:{why}"] += 1
        else:
            routes.append(r)
    dpairs, dwhy = dest_pairs(routes)
    reasons.update({f"dest:{k}": v for k, v in dwhy.items()})
    for p in dpairs:
        p["pair_id"] = f"steer_dest_{frame.name}_" + "_".join(t.split(":")[-1] for t in p["targets"])
        keep(p)
    return pairs, reasons


def steer_frames(records):
    """One item per eval frame (sample_id) from the d1 entry records; the frame's GT-route
    record ("instructed") gives the entry point when present."""
    by = {}
    for r in records:
        hm = r["habitat_meta"]
        cur = by.get(hm["sample_id"])
        if cur is None or (hm.get("kind") == "instructed" and cur["habitat_meta"].get("kind") != "instructed"):
            by[hm["sample_id"]] = r
    return [{"sample_id": s, "sample_dir": str(Path(r["image"][0]).parent), "image": r["image"][0],
             "start_uv": list(r["habitat_meta"]["start_uv_norm"]), "scene_id": r["habitat_meta"]["scene_id"],
             "viewpoint": r["habitat_meta"].get("parent_sample_id") or s,
             "is_aug": r["habitat_meta"].get("source") == "augmented"} for s, r in sorted(by.items())]


def floor_height(sample_dir):
    """World Y of the frame's floor (state.npz floor_height), or None."""
    p = Path(sample_dir) / "state.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return float(z["floor_height"]) if "floor_height" in z.files else None
```

- [ ] **Step 5: Add the `steer` stage**

In `scripts/prepare_habitat_instr.py`:

1. Docstring — append after the route-descriptions block:

```
Steering eval (2026-10-07, docs/superpowers/specs/2026-10-07-steering-eval-design.md): eval-only
pairs planned on each d1 eval frame's own map (CPU, 4 single-threaded workers), checked on the
navmesh in the habitat env, then one pair per viewpoint per slice:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py steer --limit 40     # smoke run
    uv run scripts/prepare_habitat_instr.py steer
    conda run -n habitat python scripts/navmesh_check.py         # see its docstring
    uv run scripts/prepare_habitat_instr.py steer-select
    # after STAGE=steer sbatch slurm/label_descriptions.sbatch
    uv run scripts/prepare_habitat_instr.py finalize-steer
```

2. Imports: `from rover_vlm import steering` and `from collections import Counter` (already imported — keep one).

3. Stage code (next to `stage_events`):

```python
STEER_DIR = V3_OUT / "steer"
STEER_WORKERS = 4
_STEER = {}


def _steer_init(names, fallback, kinds):
    _STEER.update(names=names, fallback=fallback, kinds=kinds)


def steer_frame(item):
    """(candidate pairs, reason counts, error or None) for one eval frame (pool worker)."""
    try:
        pairs, reasons = frame_pairs(item["sample_dir"], item["start_uv"], _STEER["names"],
                                     _STEER["fallback"], _STEER["kinds"])
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted reason, not a dead stage
        return [], {"error": 1}, f"{item['sample_id']}: {type(e).__name__}: {e}"
    floor_y = steering.floor_height(item["sample_dir"])
    for p in pairs:
        p.update({k: item[k] for k in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")},
                 floor_y=floor_y)
    return pairs, dict(reasons), None


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def stage_steer(args):
    import time
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    items = steering.steer_frames(json.loads(args.frames.read_text()))
    if args.limit:
        items = items[: args.limit]
    names, fallback, kinds = steering.load_names(_jsonl(args.names), _jsonl(args.fallback_names))
    t0 = time.time()
    with Pool(STEER_WORKERS, initializer=_steer_init, initargs=(names, fallback, kinds)) as pool:
        results = pool.map(steer_frame, items, chunksize=8)
    reasons, errors, n_frames = Counter(), [], {"side": set(), "dest": set()}
    with (out / "candidates_eval.jsonl").open("w") as f:
        for pairs, rs, err in results:
            reasons.update(rs)
            if err:
                errors.append(err)
            for p in pairs:
                n_frames[p["slice"]].add(p["viewpoint"])
                f.write(json.dumps(p) + "\n")
    meta = {"frames": len(items), "seconds": round(time.time() - t0, 1),
            "viewpoints_with_candidates": {k: len(v) for k, v in n_frames.items()},
            "reasons": dict(sorted(reasons.items())), "errors": errors[:20], "n_errors": len(errors)}
    (out / "steer_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
```

and import `frame_pairs` at module top: `from rover_vlm.steering import frame_pairs` (the test monkeypatches `prep.frame_pairs`, so `steer_frame` must call the module-level name).

4. Subparser + dispatch (in `main()`):

```python
    m = sub.add_parser("steer", help="steering: candidate pairs per d1 eval frame (CPU)")
    m.add_argument("--frames", type=Path, default=V3_OUT / "eval_desc_entry.json")
    m.add_argument("--names", type=Path, default=D1_LABELS / "names_eval_gemma_d1.jsonl")
    m.add_argument("--fallback-names", type=Path, default=D1_LABELS / "names_eval_d1.jsonl")
    m.add_argument("--out-dir", type=Path, default=STEER_DIR)
    m.add_argument("--limit", type=int, default=0, help="first N frames (smoke run)")
```

and add `"steer": stage_steer` to the dispatch dict.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_descriptions.py tests/test_steering.py -q`
Expected: all pass (existing `label_route` tests too — behaviour unchanged).

- [ ] **Step 7: Smoke run on real frames (CPU, login node)**

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run scripts/prepare_habitat_instr.py steer --limit 40 --out-dir /tmp/claude-steer-smoke
```

Expected: prints `steer_meta.json` with `frames: 40`, `n_errors: 0`, some `side:kept` and `dest:kept`, and seconds (note the per-frame time — the full run is ~2,105 frames / 4 workers). If `n_errors > 0`, read `errors` and fix before the full run. Delete `/tmp/claude-steer-smoke` afterwards.

- [ ] **Step 8: Commit**

```bash
git add src/rover_vlm/descriptions.py src/rover_vlm/steering.py scripts/prepare_habitat_instr.py tests/test_descriptions.py tests/test_steering.py
git commit -m "feat: steer stage — named candidate pairs per d1 eval frame; choose_names (A5 rule)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Builders take a planner; clearance 0.30 m; hidden stretches in the answer (Addendum A1)

**Files:**
- Modify: `src/rover_vlm/steering.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: Tasks 1–5 (`traversable`, `plan_route`, `side_routes`, `side_pair`, `dest_route`, `frame_pairs`, `answer_path`, `_rounded`); `route_events.{first_blocker, ROUTE_STEP_M}`; `landmarks.MIN_FWD_M`.
- Produces (Tasks 8–9 rely on these exact names):
  - `CLEAR_M = 0.30` (was 0.45)
  - **planner callable** `plan(frame, waypoints_xz) -> np.ndarray (N, 2) | None`
  - `map_planner(clear_m=CLEAR_M)` — A* on the frame's own map (the unit tests' stand-in)
  - `request_key(frame_name, waypoints_xz) -> str`
  - `recording_planner(requests: dict) -> plan` (plans nothing, records `key -> waypoints`)
  - `lookup_planner(routes: dict) -> plan` (answers from `{key: route_xz | None}`)
  - `answer_path(cam, route_xz, frame=None) -> list | None` — `None` when a point is < `MIN_FWD_M` in front of the camera; with `frame`, `v = 0` where `first_blocker` finds an occupied cell
  - new signatures: `side_routes(frame, start, blob, plan)`, `side_pair(frame, meta, start, blob, plan)`, `dest_route(frame, meta, start, blob, plan)`, `frame_pairs(sample_dir, start_uv, names, fallback, kinds, plan)`
  - reason `no_free_side` is gone (the planner decides); new reason `behind_camera`

Why: Addendum A1 — the frame-map planner kept 0 pairs on real frames; routes now come from the scene navmesh, planned in the `habitat` env between two uv passes (Task 8). The builders therefore take the planner as an argument.

- [ ] **Step 1: Update the existing tests and write the new ones**

In `tests/test_steering.py`:
1. `test_traversable_keeps_clearance_unknown_and_frame`: call `st.traversable(_grid(s), CAM, clear_m=0.45)` (the test pins the mechanism at the old radius; the default is now 0.30).
2. Every call of `st.side_pair(frame, meta, mask, START, blob)` becomes `st.side_pair(frame, meta, START, blob, st.map_planner())`; every `st.dest_route(frame, meta, mask, START, b)` becomes `st.dest_route(frame, meta, START, b, st.map_planner())`; every `st.frame_pairs(sd, uv, names, fb, kinds)` gains a last argument `st.map_planner()`. `_frame` keeps returning `(meta, frame, mask)` (the two-sided test still plans with the mask).
3. In `test_frame_pairs_counts_no_start_and_no_names` nothing else changes (a start above the horizon is still `no_start`).

Append:

```python
def test_answer_path_marks_floor_behind_an_obstacle_hidden(tmp_path):
    _, frame, _ = _frame(tmp_path, _box(_state(), *BOX))
    path = st.answer_path(frame.cam, [(3.4, 1.2), (2.5, 3.6)], frame)    # from beside the box to straight behind it
    assert path[0][2] == 1                                   # near the entry: seen
    assert path[-1][2] == 0                                  # 0.7 m behind the box, straight behind it
    assert st.answer_path(frame.cam, [(2.5, 0.3), (2.5, 2.0)], frame) is None   # starts 0.1 m from the camera


def test_planners_record_look_up_and_agree(tmp_path):
    meta, frame, _ = _frame(tmp_path, _box(_state(), *BOX))
    blob = int(frame.lab[54, 50])
    reqs = {}
    pair, why = st.side_pair(frame, meta, START, blob, st.recording_planner(reqs))
    assert pair is None and why == "plan_failed" and len(reqs) == 2 * len(st.STOP_BEYOND_M)
    assert all(k.startswith("f|") for k in reqs)
    direct = st.map_planner()
    routes = {k: (lambda r: None if r is None else r.tolist())(direct(frame, w)) for k, w in reqs.items()}
    looked, _ = st.side_pair(frame, meta, START, blob, st.lookup_planner(routes))
    ref, _ = st.side_pair(frame, meta, START, blob, st.map_planner())
    assert looked["routes"][0]["route_xz"] == ref["routes"][0]["route_xz"]
    assert st.request_key("f", [(1.0, 2.0)]) == "f|1.000,2.000"


def test_frame_pairs_with_recorded_then_looked_up_routes(tmp_path):
    sd = _sample(tmp_path / "g", _dest_state())
    _, frame = load_frame(sd)
    a, b = int(frame.lab[70, 70]), int(frame.lab[70, 30])
    names = {f"g:b{a}": "armchair", f"g:b{b}": "sofa"}
    reqs = {}
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], names, None, {}, st.recording_planner(reqs))
    assert pairs == [] and reqs
    direct = st.map_planner()
    routes = {k: (lambda r: None if r is None else r.tolist())(direct(frame, w)) for k, w in reqs.items()}
    pairs, reasons = st.frame_pairs(sd, [0.5, 1.0], names, None, {}, st.lookup_planner(routes))
    assert [p["pair_id"] for p in pairs if p["slice"] == "dest"] == [f"steer_dest_g_b{a}_b{b}"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_steering.py -q`
Expected: FAIL — `side_pair()` takes the old arguments / `map_planner` missing.

- [ ] **Step 3: Implement**

In `src/rover_vlm/steering.py`:

```python
CLEAR_M = 0.30      # Addendum A1: navmesh bake radius and via clearance (d1 GT routes keep 0.44-0.51 m)
```

(add `MIN_FWD_M` to the `.landmarks` import and `ROUTE_STEP_M, first_blocker` to the `.route_events` import)

```python
def answer_path(cam, route_xz, frame=None):
    """[x, y, v] waypoints of a floor route, built as the d1 answers are: densified at
    ROUTE_STEP_M, projected, v = 0 where the frame map (when given) has an occupied cell between
    the camera and the point (route_events.first_blocker), clipped to the unit square, resampled
    with transitions. None when any point is less than MIN_FWD_M in front of the camera."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    uv, fwd = cam.uv(pts)
    if (fwd < MIN_FWD_M).any():
        return None
    hidden = [frame is not None and first_blocker(frame, p) is not None for p in pts]
    clipped = clip_polyline_unit([(float(u), float(v), h) for (u, v), h in zip(uv, hidden)])
    return [list(w) for w in resample_with_transitions(clipped)]


def map_planner(clear_m=CLEAR_M):
    """plan(frame, waypoints_xz) on the frame's own map (Task 1's A*): the unit tests' stand-in
    for the navmesh (synthetic scenes have none). Masks are cached per frame."""
    masks = {}

    def plan(frame, waypoints_xz):
        if frame.name not in masks:
            masks[frame.name] = traversable(frame.grid, frame.cam, clear_m)
        return plan_route(frame.grid, masks[frame.name], waypoints_xz)
    return plan


def request_key(frame_name, waypoints_xz):
    """Stable id of one planning request: the frame and its waypoints to the millimetre."""
    return frame_name + "|" + ";".join(f"{x:.3f},{z:.3f}" for x, z in np.asarray(waypoints_xz, float))


def recording_planner(requests):
    """plan() that plans nothing and records each request in `requests` (key -> waypoints):
    steer-propose runs the builders with it to learn what navmesh_plan.py must plan."""
    def plan(frame, waypoints_xz):
        requests[request_key(frame.name, waypoints_xz)] = _rounded(waypoints_xz)
        return None
    return plan


def lookup_planner(routes):
    """plan() answering from navmesh routes {key: route_xz or None} (navmesh_plan.py output)."""
    def plan(frame, waypoints_xz):
        r = routes.get(request_key(frame.name, waypoints_xz))
        return None if r is None else np.asarray(r, float)
    return plan
```

`side_routes(frame, start, blob, plan)`: drop the `mask` argument and the `snap(...) is None -> "no_free_side"` check; plan each side with `plan(frame, [start, via[s], stop])`; everything else as before (it still returns `"too_near"` / `"plan_failed"`).

`side_pair(frame, meta, start, blob, plan)`: call `side_routes(frame, start, blob, plan)`; build each path with `path = answer_path(frame.cam, routes[s], frame)`; `if path is None: return None, "behind_camera"` before the existing `len(path) < 2` check.

`dest_route(frame, meta, start, blob, plan)`: `route = plan(frame, [start, q - d * DEST_GAP_M])`; same `answer_path(..., frame)` / `behind_camera` handling.

`frame_pairs(sample_dir, start_uv, names, fallback, kinds, plan)`: delete the `mask = traversable(...)` line and the `snap(...) is None` start test (the planner snaps); `no_start` now means only `entry_floor(...) is None`; pass `plan` to `side_pair` / `dest_route`. Update its docstring: "`plan` is a planner callable (map_planner in tests, lookup_planner in the steer stage)".

`traversable`, `snap`, `plan_route` stay: `map_planner` uses them.

In `scripts/prepare_habitat_instr.py`, `steer_frame` calls `frame_pairs(..., kinds)` without a planner — leave it for Task 8, which rewrites the stage; to keep the suite green now, pass `steering.map_planner()` there with a `# replaced in Task 8` comment.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_steering.py -q && uv run pytest -q`
Expected: all pass. If a synthetic scene changes outcome at 0.30 m (e.g. the second-obstacle test), print both routes and adjust only the scene, never the rule; say so in the report.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py tests/test_steering.py scripts/prepare_habitat_instr.py
git commit -m "feat: steering builders take a planner (navmesh-ready); 0.30 m clearance; hidden answer stretches

Co-Authored-By: <your model's trailer>"
```

---

### Task 7: Name every visible object (`steer-objects`, `--all-objects`, `STAGE=steer_names`)

**Files:**
- Modify: `src/rover_vlm/steering.py`
- Modify: `scripts/prepare_habitat_instr.py`
- Modify: `scripts/label_descriptions.py`
- Modify: `slurm/label_descriptions.sbatch`
- Test: `tests/test_steering.py`, `tests/test_slurm_thread_caps.py`

**Interfaces:**
- Consumes: `route_events.{load_frame, _blob_cells, visible_cells, object_outline, WALL_EXTENT_M, MIN_VISIBLE_CELLS, MIN_AREA}`; `steering.steer_frames` (Task 5).
- Produces:
  - `steering.object_rows(sample_dir, image, skip_ids=()) -> list[dict]` — rows `{object_id, kind_hint: "object", image, polygon, bbox, crop_box, area, depth_fallback}` (the fields `descriptions.label_object` reads)
  - stage `prepare_habitat_instr.py steer-objects` → `steer/objects_eval.jsonl`, `steer/objects_meta.json`
  - `label_descriptions.py --stage name --all-objects --objects FILE --out FILE` (no `--events`)
  - `STAGE=steer_names sbatch slurm/label_descriptions.sbatch` → `outputs/instr_labels_v3/names_steer_gemma_d1.jsonl` (Gemma)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering.py`:

```python
def test_object_rows_cover_visible_non_wall_objects(tmp_path):
    s = _dest_state()
    s[90:94, 0:100] = OCC                       # a wall across the far end (longer than WALL_EXTENT_M)
    sd = _sample(tmp_path / "g", s)
    _, frame = load_frame(sd)
    a, b, w = int(frame.lab[70, 70]), int(frame.lab[70, 30]), int(frame.lab[92, 50])
    rows = st.object_rows(sd, "/img.png")
    ids = {r["object_id"] for r in rows}
    assert ids == {f"g:b{a}", f"g:b{b}"}                         # the wall is left out
    assert all(r["kind_hint"] == "object" and r["image"] == "/img.png" and len(r["polygon"]) >= 3 for r in rows)
    assert {r["object_id"] for r in st.object_rows(sd, "/img.png", skip_ids={f"g:b{a}"})} == {f"g:b{b}"}
    assert st.object_rows(tmp_path / "missing", "/img.png") == []


def _label_desc():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "label_descriptions", Path(__file__).resolve().parent.parent / "scripts" / "label_descriptions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_label_descriptions_all_objects_needs_no_events(tmp_path, monkeypatch, capsys):
    ld = _label_desc()
    objs = tmp_path / "objects.jsonl"
    objs.write_text(json.dumps({"object_id": "g:b1", "kind_hint": "object", "image": "x", "polygon": [],
                                "crop_box": None, "bbox": None}) + "\n")
    out = tmp_path / "names.jsonl"
    out.write_text(json.dumps({"id": "g:b1", "passed": True, "name": "sofa"}) + "\n")   # already done
    monkeypatch.setattr("sys.argv", ["x", "--stage", "name", "--all-objects", "--objects", str(objs),
                                     "--out", str(out), "--resume"])
    ld.main()                                                    # nothing to label: no endpoint needed
    assert "1 items, 0 to label" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["x", "--stage", "compose", "--all-objects", "--out", str(out)])
    with pytest.raises(SystemExit):
        ld.main()
```

Append to `tests/test_slurm_thread_caps.py`:

```python
def test_label_descriptions_has_a_steer_names_stage():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert '"$STAGE" = "steer_names"' in text and "--all-objects" in text
    assert "names_steer_gemma_d1.jsonl" in text and 'serve "$GEMMA_MODEL"' in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_steering.py tests/test_slurm_thread_caps.py -q`
Expected: FAIL (`object_rows` missing; `--all-objects` unknown; no `steer_names` stage).

- [ ] **Step 3: Implement**

`src/rover_vlm/steering.py` (imports: `visible_cells, object_outline, MIN_AREA` from `.route_events`):

```python
def object_rows(sample_dir, image, skip_ids=()):
    """One naming row per visible non-wall object of a frame (Addendum A1), as d1's object rows:
    blobs shorter than WALL_EXTENT_M, with >= MIN_VISIBLE_CELLS visible cells and an outline of
    at least MIN_AREA; ids in skip_ids (already named) are left out."""
    sd = Path(sample_dir)
    if not (sd / "state.npz").exists():
        return []
    _, frame = load_frame(sd)
    out = []
    for b, (cells, xz) in sorted(_blob_cells(frame).items()):
        oid = f"{frame.name}:b{b}"
        if oid in skip_ids or float(np.hypot(*np.ptp(xz, axis=0))) >= WALL_EXTENT_M:
            continue
        vis = visible_cells(frame, cells, int(b))
        if len(vis) < MIN_VISIBLE_CELLS:
            continue
        o = object_outline(frame, vis)
        if o["area"] < MIN_AREA:
            continue
        out.append({"object_id": oid, "kind_hint": "object", "image": image,
                    **{k: o[k] for k in ("polygon", "bbox", "crop_box", "area", "depth_fallback")}})
    return out
```

`scripts/prepare_habitat_instr.py` — docstring line `uv run scripts/prepare_habitat_instr.py steer-objects` in the steering block, and:

```python
def steer_objects_frame(args):
    item, skip = args
    try:
        return steering.object_rows(item["sample_dir"], item["image"], skip), None
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted error, not a dead stage
        return [], f"{item['sample_id']}: {type(e).__name__}: {e}"


def stage_steer_objects(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    items = steering.steer_frames(json.loads(args.frames.read_text()))
    if args.limit:
        items = items[: args.limit]
    named = {r["id"] for p in (args.names, args.fallback_names) for r in _jsonl(p) if r.get("passed")}
    with Pool(STEER_WORKERS) as pool:
        results = pool.map(steer_objects_frame, [(i, named) for i in items], chunksize=8)
    rows = [r for rs, _ in results for r in rs]
    errors = [e for _, e in results if e]
    (out / "objects_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"frames": len(items), "objects": len(rows), "already_named": len(named), "errors": errors[:20],
            "n_errors": len(errors)}
    (out / "objects_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
```

with subparser `steer-objects` (`--frames`, `--names`, `--fallback-names`, `--out-dir`, `--limit`, defaults as the `steer` stage's) and its dispatch entry.

`scripts/label_descriptions.py`:
- docstring: add
  ```
  # steering (Addendum A1): name every row of an objects file, no routes
  uv run scripts/label_descriptions.py --stage name --all-objects \
      --objects data/prepared_habitat_instr_v3/steer/objects_eval.jsonl \
      --out outputs/instr_labels_v3/names_steer_gemma_d1.jsonl --concurrency 8 --resume
  ```
- `--events` no longer `required=True` (help: "events_*.jsonl (not with --all-objects)"); new `p.add_argument("--all-objects", action="store_true", help="stage name: name every row of --objects; no --events (steering, Addendum A1)")`.
- after parsing:
  ```python
  if args.all_objects and args.stage != "name":
      raise SystemExit("--all-objects is for --stage name")
  if not args.all_objects:
      if not args.events:
          raise SystemExit("--events is required (unless --stage name --all-objects)")
      check_events_schema(args.events)
  routes = [] if args.all_objects else select_routes(jsonl(args.events), args.limit)
  ```
  and in the name branch, before the `--only-unusable-under` logic: `if args.all_objects: items = jsonl(args.objects)[: args.limit] if args.limit else jsonl(args.objects)` (keeping the existing branches as `elif` / `else`), skipping the `wanted`/`missing` check for this case.

`slurm/label_descriptions.sbatch`: header comment

```bash
# STAGE=steer_names (2026-10-08, steering Addendum A1): Gemma names every not-yet-named visible object of
# the eval frames (prepare_habitat_instr.py steer-objects); no compose. Then: steer-propose.
#     STAGE=steer_names sbatch slurm/label_descriptions.sbatch
```

and, just before `if [ "$NAMER" = "cosmos" ]; then`:

```bash
if [ "$STAGE" = "steer_names" ]; then
    SD=data/prepared_habitat_instr_v3/steer
    [ -f "$SD/objects_eval.jsonl" ] || { echo "missing $SD/objects_eval.jsonl — run prepare_habitat_instr.py steer-objects"; exit 1; }
    serve "$GEMMA_MODEL"
    steer_names=(--stage name --all-objects --objects "$SD/objects_eval.jsonl" --out "$OUT/names_steer_gemma_d1.jsonl"
                 --endpoint "$BASE/v1" --concurrency "$CONCURRENCY")
    echo; echo "=============== name: steer objects (pass 1) ==============="
    "$UV" run scripts/label_descriptions.py "${steer_names[@]}" --resume
    echo; echo "=============== name: steer objects (pass 2, retries errors) ==============="
    "$UV" run scripts/label_descriptions.py "${steer_names[@]}" --resume
    echo "steer names done: uv run scripts/prepare_habitat_instr.py steer-propose"
    exit 0
fi
```

- [ ] **Step 4: Run the tests, then a smoke run**

Run: `uv run pytest tests/test_steering.py tests/test_slurm_thread_caps.py -q && uv run pytest -q && bash -n slurm/label_descriptions.sbatch`. Then (CPU, OMP 1, 4 workers) `uv run scripts/prepare_habitat_instr.py steer-objects --limit 40 --out-dir <scratchpad>/steer-smoke` and paste `objects_meta.json` into the report (expect `n_errors: 0` and objects ≈ 1–2 per frame).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py scripts/prepare_habitat_instr.py scripts/label_descriptions.py slurm/label_descriptions.sbatch tests/test_steering.py tests/test_slurm_thread_caps.py
git commit -m "feat: steer-objects + --all-objects + STAGE=steer_names — name every visible eval object

Co-Authored-By: <your model's trailer>"
```

---

### Task 8: Propose → navmesh plan → steer (`steer-propose`, `scripts/navmesh_plan.py`, `steer`)

**Files:**
- Create: `scripts/navmesh_plan.py`
- Create: `tests/test_navmesh_plan.py`
- Modify: `scripts/prepare_habitat_instr.py` (rewrite `_steer_init` / `steer_frame` / `stage_steer`; add `stage_steer_propose`)
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: Task 6 planners (`recording_planner`, `lookup_planner`, `request_key`), `frame_pairs(..., plan)`; `steering.load_names`; in `navmesh_plan.py`'s `main()` only: `roverbench.scene.navmesh.{load_pathfinder, recompute_navmesh}` and `habitat_sim.ShortestPath`.
- Produces:
  - `steer/requests_eval.jsonl` rows `{key, sample_id, scene_id, floor_y, waypoints_xz}` (`steer-propose`)
  - `steer/routes_eval.json` = `{key: {"route_xz": [[x, z], ...] | null, "reason": str | null}}` (`navmesh_plan.py`)
  - `steer/candidates_eval.jsonl`, `steer/steer_meta.json` (as Task 5) and `steer/names_primary_eval.jsonl` (d1 Gemma + steer Gemma name rows — the compose job's `--names`) (`steer`)
  - `navmesh_plan.py` pure functions: `snap_xz(pf, xz, floor_y, tol)`, `plan_request(pf, waypoints_xz, floor_y, find_path, tol) -> (route | None, reason | None)`, `plan_all(rows, get_pf, find_path, tol) -> dict`, `find_scene_glb(root, scene_id)`
  - `_steer_init(names, fallback, kinds, routes)` and `steer_frame(item) -> (pairs, reasons, error, requests)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_navmesh_plan.py`:

```python
"""navmesh_plan: request planning with a fake PathFinder (habitat_sim is not imported here)."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "navmesh_plan", Path(__file__).resolve().parent.parent / "scripts" / "navmesh_plan.py")
npl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npl)


class FakePF:
    """Navigable floor: the square 0 <= x, z <= 2 at y = 1.0."""
    def snap_point(self, p):
        return np.array([min(max(p[0], 0), 2), 1.0, min(max(p[2], 0), 2)], np.float32)


class NanPF(FakePF):
    def snap_point(self, p):
        return np.array([np.nan, np.nan, np.nan], np.float32)


def straight(pf, a, b):
    return [a, (a + b) / 2, b]


def test_plan_request_joins_legs_and_reports_reasons():
    route, why = npl.plan_request(FakePF(), [(0.2, 0.2), (1.0, 1.8), (1.8, 0.2)], 1.0, straight)
    assert why is None and route[0] == [0.2, 0.2] and route[-1] == [1.8, 0.2]
    assert [1.0, 1.8] in route and len(route) == 5                      # legs joined, shared via kept once
    assert npl.plan_request(FakePF(), [(0.2, 0.2), (2.5, 1.0)], 1.0, straight) == (None, "off_navmesh")
    assert npl.plan_request(NanPF(), [(0.2, 0.2), (1.0, 1.0)], 1.0, straight) == (None, "off_navmesh")
    assert npl.plan_request(FakePF(), [(0.2, 0.2), (1.0, 1.0)], None, straight) == (None, "no_floor")
    assert npl.plan_request(FakePF(), [(0.2, 0.2), (1.0, 1.0)], 1.0, lambda pf, a, b: None) == (None, "no_path")


def test_plan_all_keys_and_missing_scene():
    rows = [{"key": "f|a", "scene_id": "s1", "floor_y": 1.0, "waypoints_xz": [[0.2, 0.2], [1.0, 1.0]]},
            {"key": "g|b", "scene_id": "gone", "floor_y": 1.0, "waypoints_xz": [[0.2, 0.2], [1.0, 1.0]]}]
    out = npl.plan_all(rows, lambda s: FakePF() if s == "s1" else None, straight)
    assert out["f|a"]["reason"] is None and out["f|a"]["route_xz"][-1] == [1.0, 1.0]
    assert out["g|b"] == {"route_xz": None, "reason": "no_navmesh"}


def test_find_scene_glb_skips_semantic(tmp_path):
    d = tmp_path / "val" / "00803-abc"
    d.mkdir(parents=True)
    (d / "abc.semantic.glb").write_text("")
    (d / "abc.basis.glb").write_text("")
    assert npl.find_scene_glb(tmp_path, "00803-abc") == d / "abc.basis.glb"
    assert npl.find_scene_glb(tmp_path, "nope") is None
```

In `tests/test_steering.py`, replace `test_steer_frame_turns_an_exception_into_a_reason` with:

```python
def _prep():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return prep


def test_steer_frame_turns_an_exception_into_a_reason(monkeypatch):
    prep = _prep()

    def boom(*a, **k):
        raise ValueError("bad map")
    monkeypatch.setattr(prep, "frame_pairs", boom)
    prep._steer_init({}, None, {}, None)
    pairs, reasons, err, reqs = prep.steer_frame({"sample_id": "x", "sample_dir": "/nope", "start_uv": [0.5, 1.0]})
    assert pairs == [] and reasons == {"error": 1} and "ValueError: bad map" in err and reqs == {}


def test_steer_frame_records_then_looks_up(tmp_path):
    prep = _prep()
    sd = _sample(tmp_path / "g", _dest_state())
    _, frame = load_frame(sd)
    a, b = int(frame.lab[70, 70]), int(frame.lab[70, 30])
    names = {f"g:b{a}": "armchair", f"g:b{b}": "sofa"}
    item = {"sample_id": "g", "sample_dir": str(sd), "start_uv": [0.5, 1.0], "viewpoint": "g", "is_aug": False,
            "scene_id": "sc", "image": "/img.png"}
    prep._steer_init(names, None, {}, None)
    pairs, _, err, reqs = prep.steer_frame(item)
    assert err is None and pairs == [] and reqs
    direct = st.map_planner()
    routes = {k: (lambda r: None if r is None else r.tolist())(direct(frame, w)) for k, w in reqs.items()}
    prep._steer_init(names, None, {}, routes)
    pairs, _, err, reqs = prep.steer_frame(item)
    assert err is None and reqs == {} and "dest" in [p["slice"] for p in pairs]
    assert pairs[0]["floor_y"] == pytest.approx(0.1) and pairs[0]["viewpoint"] == "g"
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_navmesh_plan.py tests/test_steering.py -q`
Expected: FAIL (script missing; `_steer_init` takes 3 arguments).

- [ ] **Step 3: Write `scripts/navmesh_plan.py`**

```python
"""Plan steering routes on the scene navmesh (habitat env; Addendum A1).

Runs in the `habitat` conda env (habitat_sim 0.3.3, Python 3.9), NOT the uv .venv. Call the env's
python directly — `conda run` inside this repo picks up .python-version (pyenv 3.12) and fails:

    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    ~/miniforge3/envs/habitat/bin/python scripts/navmesh_plan.py \
        --requests data/prepared_habitat_instr_v3/steer/requests_eval.jsonl \
        --out data/prepared_habitat_instr_v3/steer/routes_eval.json

Per scene the navmesh is baked at agent radius --clear-m (CPU, create_renderer=False, via
spaceitup-habitat's roverbench.scene.navmesh.recompute_navmesh; ~1 s per scene) and cached under
outputs/navmesh_cache/. Each request's waypoints are snapped onto the navmesh (within --snap-m,
at the frame's floor height) and joined by one geodesic per leg. Writes
{key: {"route_xz": [[x, z], ...] | null, "reason": null | "off_navmesh" | "no_path" | "no_floor" | "no_navmesh"}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HM3D_ROOT = Path("/nfs/projects/spaceitup/rover_navigation/data/hm3d")
ROVERBENCH_SRC = REPO_ROOT.parent / "spaceitup-habitat" / "src"
CACHE = REPO_ROOT / "outputs" / "navmesh_cache"
STEER = REPO_ROOT / "data" / "prepared_habitat_instr_v3" / "steer"
CLEAR_M = 0.30      # = rover_vlm.steering.CLEAR_M (this script runs outside the uv env)
SNAP_M = 0.3        # = rover_vlm.steering.SNAP_M
MAX_Y_DELTA = 0.5


def snap_xz(pf, xz, floor_y, tol=SNAP_M):
    """The navmesh point nearest to floor point xz, or None when it is farther than tol (or NaN)."""
    p = np.array([xz[0], floor_y, xz[1]], np.float32)
    s = np.asarray(pf.snap_point(p), float)
    if (not np.all(np.isfinite(s)) or abs(s[1] - floor_y) > MAX_Y_DELTA
            or np.hypot(s[0] - xz[0], s[2] - xz[1]) > tol):
        return None
    return s


def plan_request(pf, waypoints_xz, floor_y, find_path, tol=SNAP_M):
    """(route_xz, None) through the snapped waypoints, one geodesic per leg (find_path(pf, a, b) ->
    list of 3-D points or None); else (None, reason)."""
    if floor_y is None:
        return None, "no_floor"
    pts = [snap_xz(pf, w, floor_y, tol) for w in waypoints_xz]
    if any(p is None for p in pts):
        return None, "off_navmesh"
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        leg = find_path(pf, a, b)
        if leg is None or len(leg) < 2:
            return None, "no_path"
        leg = [[round(float(q[0]), 3), round(float(q[2]), 3)] for q in leg]
        out += leg if not out else leg[1:]
    return out, None


def plan_all(rows, get_pf, find_path, tol=SNAP_M):
    out = {}
    for row in rows:
        pf = get_pf(row["scene_id"])
        if pf is None:
            out[row["key"]] = {"route_xz": None, "reason": "no_navmesh"}
            continue
        route, why = plan_request(pf, row["waypoints_xz"], row.get("floor_y"), find_path, tol)
        out[row["key"]] = {"route_xz": route, "reason": why}
    return out


def find_scene_glb(root, scene_id):
    """The scene's render mesh (<split>/<scene_id>/*.basis.glb), first split alphabetically."""
    hits = sorted(Path(root).glob(f"*/{scene_id}/*.basis.glb"))
    return hits[0] if hits else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", type=Path, default=STEER / "requests_eval.jsonl")
    p.add_argument("--out", type=Path, default=STEER / "routes_eval.json")
    p.add_argument("--clear-m", type=float, default=CLEAR_M)
    p.add_argument("--snap-m", type=float, default=SNAP_M)
    p.add_argument("--hm3d-root", type=Path, default=HM3D_ROOT)
    p.add_argument("--roverbench-src", type=Path, default=ROVERBENCH_SRC)
    p.add_argument("--cache", type=Path, default=CACHE)
    args = p.parse_args()
    sys.path.insert(0, str(args.roverbench_src))
    import habitat_sim
    from roverbench.scene.navmesh import load_pathfinder, recompute_navmesh

    args.cache.mkdir(parents=True, exist_ok=True)
    pfs = {}

    def get_pf(scene_id):
        if scene_id not in pfs:
            nav = args.cache / f"{scene_id}_r{int(round(args.clear_m * 100)):03d}.navmesh"
            if nav.exists():
                pfs[scene_id] = load_pathfinder(str(nav))
            else:
                glb = find_scene_glb(args.hm3d_root, scene_id)
                pfs[scene_id] = None if glb is None else recompute_navmesh(
                    str(glb), agent_radius=args.clear_m, out_navmesh=str(nav))
            print(f"scene {scene_id}: {'ok' if pfs[scene_id] is not None else 'NO NAVMESH'}", flush=True)
        return pfs[scene_id]

    def find_path(pf, a, b):
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, np.float32)
        sp.requested_end = np.asarray(b, np.float32)
        return [np.asarray(q, float) for q in sp.points] if pf.find_path(sp) else None

    rows = [json.loads(x) for x in args.requests.read_text().splitlines() if x.strip()]
    routes = plan_all(rows, get_pf, find_path, args.snap_m)
    args.out.write_text(json.dumps(routes))
    counts = Counter("ok" if v["reason"] is None else v["reason"] for v in routes.values())
    print(f"{len(routes)} requests: {dict(counts)} -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rewrite the steering stages in `scripts/prepare_habitat_instr.py`**

```python
def _steer_init(names, fallback, kinds, routes):
    _STEER.update(names=names, fallback=fallback, kinds=kinds, routes=routes)


def steer_frame(item):
    """(candidate pairs, reason counts, error or None, recorded requests) for one eval frame
    (pool worker). With no routes loaded the builders only record what to plan (steer-propose);
    with routes they build pairs from them (steer)."""
    reqs = {}
    plan = (steering.recording_planner(reqs) if _STEER["routes"] is None
            else steering.lookup_planner(_STEER["routes"]))
    try:
        pairs, reasons = frame_pairs(item["sample_dir"], item["start_uv"], _STEER["names"],
                                     _STEER["fallback"], _STEER["kinds"], plan)
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted reason, not a dead stage
        return [], {"error": 1}, f"{item['sample_id']}: {type(e).__name__}: {e}", {}
    floor_y = steering.floor_height(item["sample_dir"])
    for p in pairs:
        p.update({k: item[k] for k in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")},
                 floor_y=floor_y)
    return pairs, dict(reasons), None, reqs


def _steer_names(args):
    """(primary name rows, (names, fallback, kinds)): primary = d1 Gemma + steer Gemma rows."""
    prim = _jsonl(args.names)
    if args.steer_names.exists():
        prim += _jsonl(args.steer_names)
    else:
        print(f"WARNING: {args.steer_names} missing — only d1's names (run STAGE=steer_names first)")
    return prim, steering.load_names(prim, _jsonl(args.fallback_names))


def _run_steer_pool(args, routes):
    items = steering.steer_frames(json.loads(args.frames.read_text()))
    if args.limit:
        items = items[: args.limit]
    prim, (names, fallback, kinds) = _steer_names(args)
    with Pool(STEER_WORKERS, initializer=_steer_init, initargs=(names, fallback, kinds, routes)) as pool:
        return items, prim, pool.map(steer_frame, items, chunksize=8)


def stage_steer_propose(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    items, _, results = _run_steer_pool(args, None)
    by_id = {i["sample_id"]: i for i in items}
    rows = {}
    for _, _, _, reqs in results:
        for key, wps in reqs.items():
            it = by_id[key.split("|")[0]]
            rows[key] = {"key": key, "sample_id": it["sample_id"], "scene_id": it["scene_id"],
                         "floor_y": steering.floor_height(it["sample_dir"]), "waypoints_xz": wps}
    (out / "requests_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    errors = [e for _, _, e, _ in results if e]
    print(json.dumps({"frames": len(items), "requests": len(rows), "n_errors": len(errors), "errors": errors[:20]},
                     indent=2))
```

`stage_steer` keeps Task 5's body (candidates file, `steer_meta.json`, reason counts, errors) with these changes: it loads `routes = {k: v["route_xz"] for k, v in json.loads(args.routes.read_text()).items()}` (SystemExit with the `navmesh_plan.py` command when the file is missing), runs `_run_steer_pool(args, routes)`, unpacks 4-tuples, and also writes `out / "names_primary_eval.jsonl"` from the primary rows. Arguments of both stages: `--frames`, `--names` (d1 Gemma), `--steer-names` (default `D1_LABELS / "names_steer_gemma_d1.jsonl"`), `--fallback-names`, `--out-dir`, `--limit`; `steer` also `--routes` (default `STEER_DIR / "routes_eval.json"`). Add the `steer-propose` subparser and dispatch entry; update the steering docstring block to the A1 run order:

```
    uv run scripts/prepare_habitat_instr.py steer-objects
    STAGE=steer_names sbatch slurm/label_descriptions.sbatch            # GPU, Gemma
    uv run scripts/prepare_habitat_instr.py steer-propose
    ~/miniforge3/envs/habitat/bin/python scripts/navmesh_plan.py        # CPU, habitat env
    uv run scripts/prepare_habitat_instr.py steer
    uv run scripts/prepare_habitat_instr.py steer-select
    # after STAGE=steer sbatch slurm/label_descriptions.sbatch
    uv run scripts/prepare_habitat_instr.py finalize-steer
```

Remove the Task 6 `# replaced in Task 8` map-planner line.

- [ ] **Step 5: Run the tests, then the whole chain on 40 real frames**

Run: `uv run pytest tests/test_navmesh_plan.py tests/test_steering.py -q && uv run pytest -q`. Then, with `SM=<scratchpad>/steer-smoke`:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run scripts/prepare_habitat_instr.py steer-propose --limit 40 --out-dir $SM
OMP_NUM_THREADS=4 ~/miniforge3/envs/habitat/bin/python scripts/navmesh_plan.py --requests $SM/requests_eval.jsonl \
    --out $SM/routes_eval.json --cache $SM/navmesh_cache
uv run scripts/prepare_habitat_instr.py steer --limit 40 --out-dir $SM --routes $SM/routes_eval.json
```

Expected: no errors; the navmesh line shows mostly `ok` and some `off_navmesh`; `steer_meta.json` lists `side:kept` / `dest:kept` (without the steer Gemma names, few anchors exist — non-zero is the check, not the yield). Paste the three outputs into the report.

- [ ] **Step 6: Commit**

```bash
git add scripts/navmesh_plan.py tests/test_navmesh_plan.py scripts/prepare_habitat_instr.py tests/test_steering.py
git commit -m "feat: steer-propose -> navmesh_plan.py (habitat env) -> steer; routes from the scene navmesh

Co-Authored-By: <your model's trailer>"
```

---

### Task 9: Stop pairs replace the side / destination builders (Addendum A2)

**Files:**
- Modify: `src/rover_vlm/steering.py` (remove the planners and the side / destination builders; add the stop-pair builders; rewrite `frame_pairs`, `name_pair`, `steer_frames`)
- Modify: `scripts/prepare_habitat_instr.py` (`steer` stage without routes; remove `steer-propose`)
- Delete: `scripts/navmesh_plan.py`, `tests/test_navmesh_plan.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: `route_events.{describe_xz, load_frame, first_blocker, ROUTE_STEP_M}`, `landmarks.{CAM_HEIGHT_M, MIN_FWD_M, densify}`, and from earlier steering tasks `answer_path`, `separation`, `exclusive`, `_rounded`, `stop_key`, `load_names`, `frame_clash`, `object_rows`, `route_length`.
- Produces (Tasks 10–15 rely on these exact names):
  - constants `MIN_ROUTE_M = 1.5`, `MIN_SEP_UV = 0.1`, `CUT_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 1.0)`
  - `in_view(cam, route_xz) -> np.ndarray` — the route from where it reaches the image's bottom edge
  - `cuts(route_xz, fractions=CUT_FRACTIONS, min_m=MIN_ROUTE_M) -> list[(fraction, prefix_xz)]`
  - `stop_pairs(frame, meta, route_xz) -> (list[pair], Counter)`
  - **route dict**: `{"tag": "A"|"B", "side": None, "route_xz": [[x, z], ...] (from the rover), "path": [[x, y, v], ...] (in view), "desc": <describe_xz dict>, "anchor_event": None, "target": <stop object id or None>, "fraction": float}`
  - **pair dict**: `{"slice": "stop", "anchor": None, "targets": [idA, idB], "separation": float (answer-end distance), "routes": [A, B], "pair_id": "steer_stop_<sample_id>_<fA>_<fB>"}` (fractions as 3-digit percents, e.g. `_050_100`)
  - `name_pair(pair, names, fallback, kinds) -> str | None` (stop objects may be None; adds namer / names / template)
  - `frame_pairs(sample_dir, candidate_index, names, fallback, kinds) -> (pairs, Counter)` — reasons `no_state`, `bad_candidate`, `stop:<reason>`, `stop:kept`
  - `steer_frames(records)` items gain `candidate_index`
  - stage: `_steer_init(names, fallback, kinds)`, `steer_frame(item) -> (pairs, reasons, error)`, `prepare_habitat_instr.py steer` (no `--routes`)

Why (spec Addendum A2): on the real eval frames neither the frame map nor the navmesh gives side or destination pairs; cutting the ground-truth route at two stop points gives pairs on ~55% of frames.

- [ ] **Step 1: Remove the dead tests, then write the failing new ones**

In `tests/test_steering.py` delete every test of removed code: `test_traversable_*`, `test_astar_*`, `test_string_pull_*`, `test_plan_route_*`, `test_entry_floor_*`, `test_side_pair_*`, `test_two_routes_*`, `test_anchor_too_close_*`, `test_side_difference`, `test_dest_*`, `test_anchor_blobs_*`, `test_frame_pairs_*`, `test_floor_height`, `test_planners_*`, `test_lookup_planner_*`, `test_steer_frame_records_then_looks_up`, and the helpers only they use (`_grid`, `cell`, `_ev`, `_dest_state`, `BOX`, `START`, `_frame`'s mask, `L_XZ`/`R_XZ` stay for the separation/exclusive tests). Keep `test_route_length`, both `test_answer_path_*` (rewrite `test_answer_path_marks_floor_behind_an_obstacle_hidden` to build the frame with `load_frame(_sample(...))` if it used `_frame`), `test_separation`, `test_exclusive_*`, `test_frame_clash_*`, `test_load_names_*`, `test_object_rows_*`, `test_label_descriptions_all_objects_*`, `test_steer_frames_*` (update below). Rewrite `test_name_pair_clash_with_an_object_named_only_in_the_fallback` on a hand-made stop pair (two route dicts with `desc = {"events": [], "stop": {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b1"}}` and `target = "f:b1"`).

`_sample` gains an optional `routes` argument that writes them as meta candidates:

```python
def _sample(d, state, routes=()):
    """A synthetic sample dir: state.npz (with floor_height) and meta.json (one candidate per route)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0, floor_height=0.1)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    (d / "meta.json").write_text(json.dumps({"pose_xz": [2.5, 0.2], "heading_xz": [0.0, 1.0],
                                             "flavor": "detour", "chord": {"crossing_xz": None},
                                             "candidates": cands}))
    return d
```

Append:

```python
def _stop_scene():
    s = _state()
    s[20:30, 52:58] = OCC          # box A: x 2.6..2.9, z 1.0..1.5 (passed on the left)
    s[70:80, 36:60] = OCC          # cabinet: x 1.8..3.0, z 3.5..4.0
    return s


STOP_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (2.5, 2.9)]    # ends 0.6 m before the cabinet (route_events tests)


def test_in_view_starts_at_the_bottom_edge():
    v = st.in_view(CAM, [(2.5, 0.2), (2.5, 3.0)])
    assert v[0][1] == pytest.approx(1.0, abs=0.06) and v[-1][1] == pytest.approx(3.0)
    assert len(st.in_view(CAM, [(2.5, 0.2), (2.5, 0.6)])) == 0          # never reaches the image


def test_cuts_keep_at_least_min_route_m():
    fr = [f for f, _ in st.cuts(STOP_ROUTE)]
    assert fr == [0.5, 0.65, 0.8, 1.0]                                   # 0.35 x 3.26 m < 1.5 m
    f, pts = st.cuts(STOP_ROUTE)[-1]
    assert st.route_length(pts) == pytest.approx(st.route_length(STOP_ROUTE), abs=0.06)


def test_stop_pairs_on_a_route_past_a_box_to_a_cabinet(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    meta, frame = load_frame(sd)
    pairs, reasons = st.stop_pairs(frame, meta, STOP_ROUTE)
    assert pairs, dict(reasons)
    for p in pairs:
        a, b = p["routes"]
        assert (p["slice"], a["tag"], b["tag"]) == ("stop", "A", "B") and a["fraction"] < b["fraction"]
        assert st.stop_key(a["desc"]["stop"]) != st.stop_key(b["desc"]["stop"])
        assert a["path"][0] == b["path"][0]                                # one entry
        assert p["separation"] >= st.MIN_SEP_UV and st.exclusive(a, b)
    full = [p for p in pairs if p["routes"][1]["fraction"] == 1.0][0]["routes"][1]
    assert (full["desc"]["stop"]["type"], full["desc"]["stop"]["relation"]) == ("near", "in front of")


def test_frame_pairs_names_and_reasons(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    _, frame = load_frame(sd)
    box, cab = int(frame.lab[25, 55]), int(frame.lab[75, 48])
    names = {f"f:b{box}": "armchair", f"f:b{cab}": "cabinet"}
    pairs, reasons = st.frame_pairs(sd, 0, names, None, {})
    assert pairs and all(p["pair_id"].startswith("steer_stop_f_") for p in pairs) and reasons["stop:kept"] == len(pairs)
    for p in pairs:
        assert all(r["template"] and r["namer"] == "primary" for r in p["routes"])
        assert p["routes"][0]["template"] != p["routes"][1]["template"]
    _, reasons = st.frame_pairs(sd, 0, {}, None, {})
    assert reasons.get("stop:unnamed_object", 0) > 0 and not reasons.get("stop:kept")
    assert st.frame_pairs(sd, 5, names, None, {})[1] == {"bad_candidate": 1}
    assert st.frame_pairs(tmp_path / "missing", 0, names, None, {})[1] == {"no_state": 1}
    json.dumps(pairs)
```

Update `test_steer_frames_one_item_per_frame_gt_record_first`: give each fake record `"candidate_index"` in `habitat_meta` and assert `items[0]["candidate_index"]` is the GT record's. Update `test_steer_frame_turns_an_exception_into_a_reason`: `prep._steer_init({}, None, {})` and a 3-tuple `(pairs, reasons, err)`; add `"candidate_index": 0` to the item.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_steering.py -q`
Expected: FAIL — `in_view` / `cuts` / `stop_pairs` missing; `frame_pairs` has the old signature.

- [ ] **Step 3: Implement in `src/rover_vlm/steering.py`**

Delete: `traversable`, `astar`, `line_clear`, `string_pull`, `snap`, `plan_route`, `entry_floor`, `map_planner`, `request_key`, `recording_planner`, `lookup_planner`, `enclosed_blobs`, `two_sided`, `_event_key`, `side_difference`, `side_routes`, `side_pair`, `dest_route`, `dest_pairs`, `anchor_blobs`, `floor_height`, and the constants `CLEAR_M`, `SNAP_M`, `VIA_EXTRA_M`, `STOP_BEYOND_M`, `MIN_AHEAD_M`, `INSIDE_FRAC`, `MIN_DEST_DU`, `DEST_GAP_M`, `_NEIGH`, `_CHUNK`; drop imports that become unused (`heapq`, `math`, PIL, `FREE`, `OCC`, `Camera`, `image_to_floor`, ...). Module docstring:

```python
"""Steering eval pairs (stop slice, spec Addendum A2): one ground-truth route cut at two points.

The d1 eval pairs could not test steering (2026-10-07): both routes of a pair end at the frame's
goal and diverge behind obstacles. Side pairs (one object passed on either side) and destination
pairs (two objects to stop in front of) proved not viable on these frames (A1/A2: no room beside
objects, nothing nameable far enough from the entry). A stop pair keeps the frame's d1
ground-truth route (planned by the generator on the navmesh) and cuts it at two points whose stop
phrases differ: same image, same entry, same route, two texts, two different ends. Each cut is
described from the rover's pose, as d1's routes; its answer starts where the route reaches the
image (hidden stretches v = 0). A pair is kept only when each cut fails the other's description
under the eval's own checker. Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""
```

New code (constants next to `MIN_ROUTE_M` / `MIN_SEP_UV`; add `CAM_HEIGHT_M` to the `.landmarks` import):

```python
CUT_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 1.0)   # where a route is cut, as shares of its length


def in_view(cam, route_xz):
    """The route from where it reaches the image's bottom edge (CAM_HEIGHT_M ahead of the camera):
    the generator's routes start at the rover, below the frame. Empty when it never gets there."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    _, fwd = cam.uv(pts)
    ok = fwd >= CAM_HEIGHT_M
    return pts[int(np.argmax(ok)):] if ok.any() else pts[:0]


def cuts(route_xz, fractions=CUT_FRACTIONS, min_m=MIN_ROUTE_M):
    """(fraction, prefix) of the route at each share of its length that keeps at least min_m."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    out = []
    for f in fractions:
        s = f * arc[-1]
        if s < min_m:
            continue
        out.append((f, pts[: max(2, int(np.searchsorted(arc, s - 1e-9)) + 1)]))
    return out


def stop_pairs(frame, meta, route_xz):
    """(pairs, reasons): every two cuts of one route whose stop phrases differ, whose answer ends lie
    at least MIN_SEP_UV apart, and whose answers each fail the other's description. Tags A (the
    earlier stop) / B."""
    reasons, routes = Counter(), []
    for f, cut in cuts(route_xz):
        desc = describe_xz(meta, frame, cut)
        if desc["reason"]:
            reasons[f"cut:{desc['reason']}"] += 1
            continue
        view = in_view(frame.cam, cut)
        path = answer_path(frame.cam, view, frame) if len(view) >= 2 else None
        if path is None or len(path) < 2:
            reasons["cut:out_of_view"] += 1
            continue
        routes.append({"tag": None, "side": None, "route_xz": _rounded(cut), "path": path, "desc": desc,
                       "anchor_event": None, "target": desc["stop"]["object"], "fraction": f})
    pairs = []
    for a, b in itertools.combinations(routes, 2):
        if stop_key(a["desc"]["stop"]) == stop_key(b["desc"]["stop"]):
            reasons["same_stop"] += 1
            continue
        sep = float(np.linalg.norm(np.subtract(a["path"][-1][:2], b["path"][-1][:2])))
        if sep < MIN_SEP_UV:
            reasons["too_close"] += 1
            continue
        ra, rb = {**a, "tag": "A"}, {**b, "tag": "B"}
        if not exclusive(ra, rb):
            reasons["not_exclusive"] += 1
            continue
        pairs.append({"slice": "stop", "anchor": None, "targets": [ra["target"], rb["target"]],
                      "separation": round(sep, 4), "routes": [ra, rb]})
    return pairs, reasons
```

`name_pair` (replace the destination head-noun rule; stop objects can be None):

```python
def name_pair(pair, names, fallback, kinds):
    """Pick each route's names (A5 rule), refuse a stop object whose name is ambiguous in the frame,
    and write each route's template sentence. Returns the drop reason, or None after adding namer /
    names / template to both route dicts. Two stops at one object ("just past" / "well past the
    chair") are fine; two identical sentences are not."""
    for r in pair["routes"]:
        row = {"events": r["desc"]["events"], "stop": r["desc"]["stop"]}
        chosen, namer, reject = choose_names(row, names, fallback)
        if chosen is None:
            return "unnamed_object" if reject == "unnamed_object" else "name_clash"
        anchor = pair["anchor"] or r["target"]
        if anchor and frame_clash(anchor, chosen, names, fallback):
            return "name_clash"
        r.update(namer=namer, names={o: chosen[o] for o in route_objects(row)},
                 template=template_description(row["events"], row["stop"], chosen, kinds))
    if pair["routes"][0]["template"] == pair["routes"][1]["template"]:
        return "same_text"
    return None
```

`frame_pairs`:

```python
def frame_pairs(sample_dir, candidate_index, names, fallback, kinds):
    """The stop pairs of one frame's ground-truth route, named, with drop reasons ("no_state",
    "bad_candidate", "stop:<reason>", "stop:kept")."""
    sd = Path(sample_dir)
    reasons = Counter()
    if not (sd / "state.npz").exists():
        reasons["no_state"] += 1
        return [], reasons
    meta, frame = load_frame(sd)
    cm = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cm)) or len(cm[candidate_index].get("polyline_xz") or []) < 2:
        reasons["bad_candidate"] += 1
        return [], reasons
    pre = f"{frame.name}:"      # naming only needs this frame's objects
    names = {o: n for o, n in names.items() if o.startswith(pre)}
    fallback = {o: n for o, n in (fallback or {}).items() if o.startswith(pre)}
    pairs, why = stop_pairs(frame, meta, cm[candidate_index]["polyline_xz"])
    reasons.update({f"stop:{k}": v for k, v in why.items()})
    kept = []
    for p in pairs:
        p["pair_id"] = f"steer_stop_{frame.name}_" + "_".join(f"{round(r['fraction'] * 100):03d}" for r in p["routes"])
        w = name_pair(p, names, fallback, kinds)
        reasons[f"stop:{w or 'kept'}"] += 1
        if w is None:
            kept.append(p)
    return kept, reasons
```

`steer_frames`: add `"candidate_index": r["habitat_meta"]["candidate_index"]` to each item.

- [ ] **Step 4: Rewire the stage in `scripts/prepare_habitat_instr.py`**

Delete `stage_steer_propose`, its subparser and dispatch entry, `steering.floor_height` uses, the `--routes` argument and the routes loading. Then:

```python
def _steer_init(names, fallback, kinds):
    _STEER.update(names=names, fallback=fallback, kinds=kinds)


def steer_frame(item):
    """(candidate pairs, reason counts, error or None) for one eval frame (pool worker)."""
    try:
        pairs, reasons = frame_pairs(item["sample_dir"], item["candidate_index"], _STEER["names"],
                                     _STEER["fallback"], _STEER["kinds"])
        for p in pairs:
            p.update({k: item[k] for k in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")})
    except Exception as e:  # noqa: BLE001 — one bad frame is a counted reason, not a dead stage
        return [], {"error": 1}, f"{item['sample_id']}: {type(e).__name__}: {e}"
    return pairs, dict(reasons), None
```

`_run_steer_pool(args)` (no routes; `initargs=(names, fallback, kinds)`); `stage_steer` keeps writing `candidates_eval.jsonl`, `names_primary_eval.jsonl` and `steer_meta.json` (`viewpoints_with_candidates` keyed by slice — now only `"stop"`), unpacking 3-tuples. The `steer` subparser: `--frames`, `--names`, `--steer-names`, `--fallback-names`, `--out-dir`, `--limit` (help: "steering: stop pairs per d1 eval frame (CPU)"). Docstring steering block — the A2 run order:

```
    uv run scripts/prepare_habitat_instr.py steer-objects
    STAGE=steer_names sbatch slurm/label_descriptions.sbatch            # GPU, Gemma
    uv run scripts/prepare_habitat_instr.py steer
    uv run scripts/prepare_habitat_instr.py steer-select
    # after STAGE=steer sbatch slurm/label_descriptions.sbatch
    uv run scripts/prepare_habitat_instr.py finalize-steer
```

`git rm scripts/navmesh_plan.py tests/test_navmesh_plan.py`.

- [ ] **Step 5: Run the tests, then the stage on real frames**

Run: `uv run pytest tests/test_steering.py -q && uv run pytest -q` (all pass; the count drops by the deleted tests). Then (CPU, OMP 1, 4 workers) `uv run scripts/prepare_habitat_instr.py steer --limit 300 --out-dir <scratchpad>/steer-stop` and paste `steer_meta.json`. Expected: no errors; `viewpoints_with_candidates.stop` well above 0 (a probe with names for every object gave pairs on ~55% of original frames; with d1's names only it is lower — the GPU naming job comes later).

- [ ] **Step 6: Commit**

```bash
git add src/rover_vlm/steering.py scripts/prepare_habitat_instr.py tests/test_steering.py
git rm scripts/navmesh_plan.py tests/test_navmesh_plan.py
git commit -m "feat: stop pairs (one GT route cut at two stop points) replace side/destination pairs; drop navmesh planning

Co-Authored-By: <your model's trailer>"
```

---

### Task 10: Selection — one pair per viewpoint (`steer-select`, Gate 0)

**Files:**
- Modify: `src/rover_vlm/steering.py`
- Modify: `scripts/prepare_habitat_instr.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: candidate pair rows (Task 9); `route_events.EVENTS_SCHEMA`; `descriptions.desc_check`.
- Produces:
  - `select_pairs(candidates) -> list[pair]` (sorted by `pair_id`)
  - `pair_rows(pair) -> [row, row]` — d1 events-row schema plus steering fields: `id, pair_id, sibling_id, slice, kind="steer", reason=None, sample_id, viewpoint, is_aug, scene_id, image, sample_dir, events, stop, goal_xz, pose_xz, heading_xz, route_len_m, route_xz, path, desc_check, anchor, anchor_event, side, template, namer, names, style_key, separation, outlines`
  - files `steer/events_steer_eval.jsonl`, `steer/events_meta.json` (`{"schema": EVENTS_SCHEMA, ...}`)
  - constant `GATE0_MIN_PAIRS = 150`

- [ ] **Step 1: Write the failing tests**

```python
def _cand(pid, vp, aug, sep):
    return {"pair_id": pid, "slice": "stop", "viewpoint": vp, "is_aug": aug, "separation": sep,
            "routes": [{"tag": "A"}, {"tag": "B"}]}


def test_select_pairs_one_per_viewpoint_original_first_then_separation():
    cands = [_cand("a1", "v1", True, 0.5), _cand("a2", "v1", False, 0.2),
             _cand("b1", "v2", False, 0.2), _cand("b2", "v2", False, 0.3), _cand("c1", "v3", True, 0.4)]
    assert [p["pair_id"] for p in st.select_pairs(cands)] == ["a2", "b2", "c1"]


def test_pair_rows_carry_the_d1_schema_and_steering_fields(tmp_path):
    sd = _sample(tmp_path / "f", _stop_scene(), [STOP_ROUTE])
    _, frame = load_frame(sd)
    box, cab = int(frame.lab[25, 55]), int(frame.lab[75, 48])
    pairs, _ = st.frame_pairs(sd, 0, {f"f:b{box}": "armchair", f"f:b{cab}": "cabinet"}, None, {})
    pair = pairs[0]
    pair.update(sample_id="f", viewpoint="f", is_aug=False, scene_id="sc", image="/x.png", sample_dir=str(sd))
    a, b = st.pair_rows(pair)
    pid = pair["pair_id"]
    assert (a["id"], a["sibling_id"], b["id"], b["sibling_id"]) == (f"{pid}_A", f"{pid}_B", f"{pid}_B", f"{pid}_A")
    for r, route in zip((a, b), pair["routes"]):
        assert r["reason"] is None and r["kind"] == "steer" and r["style_key"] == pid and r["slice"] == "stop"
        assert r["anchor"] == route["target"] and r["anchor_event"] is None and r["side"] is None
        assert {"events", "stop", "template", "path", "outlines", "desc_check"} <= set(r)
    json.dumps([a, b])
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_steering.py -q` → FAIL (`select_pairs` missing).

- [ ] **Step 3: Implement**

```python
GATE0_MIN_PAIRS = 150


def select_pairs(candidates):
    """One pair per (slice, viewpoint): the original frame before its yaw variants, then the
    largest end separation (pair_id breaks ties). Sorted by pair_id."""
    best = {}
    for p in candidates:
        key, rank = (p["slice"], p["viewpoint"]), (bool(p["is_aug"]), -p["separation"], p["pair_id"])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, p)
    return sorted((v[1] for v in best.values()), key=lambda p: p["pair_id"])


def pair_rows(pair):
    """The two route rows of a selected pair, in the d1 events-row schema (so compose,
    desc_check and the audit take them) plus the steering fields."""
    ids = [f"{pair['pair_id']}_{r['tag']}" for r in pair["routes"]]
    rows = []
    for k, r in enumerate(pair["routes"]):
        d = r["desc"]
        rows.append({
            "id": ids[k], "pair_id": pair["pair_id"], "sibling_id": ids[1 - k], "slice": pair["slice"],
            "kind": "steer", "reason": None,
            **{f: pair[f] for f in ("sample_id", "viewpoint", "is_aug", "scene_id", "image", "sample_dir")},
            **{f: d[f] for f in ("events", "stop", "goal_xz", "pose_xz", "heading_xz", "route_len_m")},
            "route_xz": r["route_xz"], "path": r["path"], "desc_check": desc_check(d, d["objects"]),
            "anchor": pair["anchor"] or r["target"], "anchor_event": r["anchor_event"], "side": r["side"],
            "template": r["template"], "namer": r["namer"], "names": r["names"],
            "style_key": pair["pair_id"], "separation": pair["separation"],
            "outlines": {oid: o["polygon"] for oid, o in d["objects"].items()},
        })
    return rows
```

In `scripts/prepare_habitat_instr.py`:

```python
def stage_steer_select(args):
    d = args.out_dir
    cands = [json.loads(x) for x in (d / "candidates_eval.jsonl").read_text().splitlines() if x.strip()]
    kept = steering.select_pairs(cands)
    rows = [row for p in kept for row in steering.pair_rows(p)]
    (d / "events_steer_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    per = Counter(p["slice"] for p in kept)
    meta = {"schema": EVENTS_SCHEMA, "candidates": len(cands), "pairs": dict(per), "routes": len(rows),
            "originals": dict(Counter(p["slice"] for p in kept if not p["is_aug"]))}
    (d / "events_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    n = per.get("stop", 0)
    print(f"GATE 0 stop: {n} pairs -> {'PASS' if n >= steering.GATE0_MIN_PAIRS else 'FAIL'} "
          f"(need >= {steering.GATE0_MIN_PAIRS})")
```

subparser `steer-select` (`--out-dir`, default `STEER_DIR`) and its dispatch entry.

- [ ] **Step 4: Run the tests and the stage** — `uv run pytest tests/test_steering.py -q && uv run pytest -q`; then `uv run scripts/prepare_habitat_instr.py steer-select --out-dir <scratchpad>/steer-stop` on Task 9's output; paste the meta and the GATE 0 line.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py scripts/prepare_habitat_instr.py tests/test_steering.py
git commit -m "feat: steer-select — one stop pair per viewpoint, d1-schema rows, Gate 0

Co-Authored-By: <your model's trailer>"
```

---

### Task 11: Geometry audit page (`--mode steer`, Gate 1)

**Files:**
- Modify: `scripts/audit_instruction_labels.py`, `scripts/audit_instruction_labels.html`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: `steer/events_steer_eval.jsonl` rows (Task 10): `pair_id, id, slice, image, scene_id, anchor (stop object or None), template, path, outlines`; `overlay.draw_polyline`; the script's `_draw_badge`, `_font`, `TEMPLATE`, `jsonl`, `check_events_schema`.
- Produces: `sample_steer(rows, n, seed) -> (list[[row, row]], population: int)`, `steer_item(a, b) -> dict`, `VERDICTS["steer"]`; page `outputs/steer_audit.html` titled "Steering Pair Audit".

- [ ] **Step 1: Write the failing test**

```python
def _audit():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_instruction_labels", Path(__file__).resolve().parent.parent / "scripts" / "audit_instruction_labels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arow(pid, tag, img, anchor):
    return {"id": f"{pid}_{tag}", "pair_id": pid, "slice": "stop", "image": str(img), "scene_id": "sc",
            "anchor": anchor, "template": f"Go ... stop ({tag})", "path": [[0.5, 1.0, 1], [0.3, 0.6, 1]],
            "outlines": {"f:b1": [[0.4, 0.5], [0.6, 0.5], [0.6, 0.6]], "f:b2": [[0.1, 0.5], [0.2, 0.5], [0.2, 0.6]]}}


def test_audit_samples_pairs_and_draws_both_routes(tmp_path):
    from PIL import Image
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64), (90, 90, 90)).save(img)
    rows = [r for i in range(4) for r in (_arow(f"p{i}", "A", img, "f:b1"), _arow(f"p{i}", "B", img, None if i % 2 else "f:b2"))]
    au = _audit()
    picked, pop = au.sample_steer(rows, 3, seed=0)
    assert pop == 4 and len(picked) == 3
    assert all(a["pair_id"] == b["pair_id"] and a["id"] < b["id"] for a, b in picked)
    item = au.steer_item(*picked[0])
    assert item["id"] == picked[0][0]["pair_id"] and item["img"].startswith("data:image/jpeg;base64,")
    assert picked[0][0]["template"] in item["instruction"] and picked[0][1]["template"] in item["instruction"]
    assert [v["v"] for v in au.VERDICTS["steer"]] == ["ok", "stop", "event", "object", "same", "unsure"]
```

- [ ] **Step 2: Run it to see it fail** — `uv run pytest tests/test_steering.py::test_audit_samples_pairs_and_draws_both_routes -q`.

- [ ] **Step 3: Implement** in `scripts/audit_instruction_labels.py`:

1. Docstring:
```
--mode steer (steering Gate 1, 2026-10-08): stop pairs from steer/events_steer_eval.jsonl; both cuts of the
route on the frame (magenta = first text, cyan = second), the stop objects outlined yellow, both template
texts. Is each text true of its own route (steps and stop), and are the two stops clearly apart?
Gate >= 90% of pairs true.
    uv run scripts/audit_instruction_labels.py --mode steer --n 50     # -> outputs/steer_audit.html
```
2. Imports/constants: `from rover_vlm.overlay import draw_polyline`; `V3 = REPO_ROOT / "data" / "prepared_habitat_instr_v3"`; `STEER_A, STEER_B, ANCHOR_COLOR = (255, 0, 200), (0, 200, 255), (255, 230, 0)`.
3. `VERDICTS["steer"]`:
```python
    "steer": [
        {"v": "ok", "label": "True pair", "help": "Both texts are true of their own route, and the two stops are clearly apart"},
        {"v": "stop", "label": "Wrong stop", "help": "A stop phrase does not match where its route ends"},
        {"v": "event", "label": "Wrong step", "help": "A step (pass / between / through / keep on your side) is not true of its route"},
        {"v": "object", "label": "Wrong object", "help": "A name does not fit the thing it refers to"},
        {"v": "same", "label": "Not visibly different", "help": "The two stops are too close to tell apart in the image"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
```
4. Functions:
```python
def sample_steer(rows, n, seed):
    """Up to n pairs (both rows, id order), shuffled; and the number of pairs available."""
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], []).append(r)
    pairs = sorted((sorted(v, key=lambda r: r["id"]) for v in by.values() if len(v) == 2), key=lambda p: p[0]["pair_id"])
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n, len(pairs)))
    rng.shuffle(picked)
    return picked, len(pairs)


def steer_item(a, b):
    """One audit item: the frame with both routes and the stop objects outlined, both texts."""
    img = Image.open(a["image"]).convert("RGB").resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = _font(max(16, IMG_PX // 30))
    for k, oid in enumerate(o for o in dict.fromkeys([a["anchor"], b["anchor"]]) if o):
        poly = a["outlines"].get(oid) or b["outlines"].get(oid) or []
        pts = [(u * IMG_PX, v * IMG_PX) for u, v in poly]
        if len(pts) >= 3:
            draw.polygon(pts, outline=ANCHOR_COLOR, width=4)
            _draw_badge(draw, (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)),
                        str(k + 1), ANCHOR_COLOR, font, 14)
    for row, colour in ((a, STEER_A), (b, STEER_B)):
        draw_polyline(draw, [(p[0], p[1]) for p in row["path"]], colour, IMG_PX, IMG_PX, width=6)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return {"id": a["pair_id"], "instruction": f"MAGENTA: {a['template']}   ·   CYAN: {b['template']}",
            "slice": a["slice"], "side": a["slice"], "kind": "steer", "has_alt": False, "scene": a["scene_id"],
            "target": "object", "reason": None,
            "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
```
5. `main()`: add `"steer"` to the `--mode` choices; before `if args.mode == "events":`:
```python
    if args.mode == "steer":
        events = args.events or V3 / "steer" / "events_steer_eval.jsonl"
        check_events_schema(events)
        picked, pop = sample_steer(jsonl(events), args.n, args.seed)
        items = [steer_item(a, b) for a, b in picked]
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "steer",
                   "verdicts": VERDICTS["steer"], "source": str(events)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Steering Pair Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "steer_audit.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"{len(items)} pairs (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
```

In `scripts/audit_instruction_labels.html`, after the `if (DATA.mode === "d1") {...}` block:
```js
if (DATA.mode === "steer") {
  document.title = "Steering Pair Audit";
  $("h1").textContent = "Steering pair audit";
  $("sub").textContent = "Each frame shows one route cut at two stop points: magenta follows the first text, cyan the second; yellow outlines are the stop objects. Is each text true of its own route (steps and stop), and are the two stops clearly apart?";
  $("tgtleg").hidden = false;
  $("tgtleg").innerHTML = '<i style="background:#ff00c8"></i>magenta = first text &nbsp; <i style="background:#00c8ff"></i>cyan = second text &nbsp; <i style="background:#ffe600"></i>yellow = stop object';
  $("altleg").hidden = true;
  document.querySelector(".stripcap span").textContent = `All samples: click to jump. Keys: 1–${DATA.verdicts.length} mark and advance, ← → move.`;
}
```
and in `render()`: the `tgtleg` condition gains `|| DATA.mode === "steer"`, the `altleg` one `DATA.mode === "steer" ||`, and after the d1 `kind` line:
```js
  if (DATA.mode === "steer") { $("side").textContent = `${it.slice} pair`; $("kind").textContent = "one route, two stops"; }
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_steering.py -q && uv run pytest -q`; then `uv run scripts/audit_instruction_labels.py --mode steer --n 10 --events <scratchpad>/steer-stop/events_steer_eval.jsonl --out <scratchpad>/steer-stop/audit.html` (the stage needs `events_meta.json` next to it — Task 10 writes it).

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_instruction_labels.py scripts/audit_instruction_labels.html tests/test_steering.py
git commit -m "feat: --mode steer audit page (Gate 1) — one route, two stops, both template texts

Co-Authored-By: <your model's trailer>"
```

---

### Task 12: Composed text — shared style per pair, `STAGE=steer`

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (`label_route`)
- Modify: `slurm/label_descriptions.sbatch`
- Test: `tests/test_descriptions.py`, `tests/test_slurm_thread_caps.py`

**Interfaces:**
- Consumes: steering rows carry `style_key = pair_id` (Task 10); `steer/names_primary_eval.jsonl` (d1 Gemma + steer Gemma name rows, written by the `steer` stage).
- Produces: `label_route` picks style and shape from `row.get("style_key") or row["id"]`; `STAGE=steer` composes `data/prepared_habitat_instr_v3/steer/events_steer_eval.jsonl` → `outputs/instr_labels_v3/desc_steer_eval_d1.jsonl` with `--names steer/names_primary_eval.jsonl` and `--fallback-names names_eval_d1.jsonl`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_descriptions.py`:

```python
def test_label_route_style_key_shares_style_and_shape_across_a_pair():
    from rover_vlm.descriptions import choose_d1_style, choose_shape, label_route
    row = {"pair_id": "p", "kind": "steer", "events": [PASS_L], "stop": NEAR, "style_key": "p"}
    seen = {}
    for rid in ("p_L", "p_R"):
        label_route(lambda pr, s, rid=rid: (seen.setdefault(rid, " ".join(pr.split())) or "", "stop"),
                    {**row, "id": rid}, NAMES, seed=0)
    for rid in ("p_L", "p_R"):
        assert choose_d1_style("p") in seen[rid] and choose_shape("p") in seen[rid]
```

Append to `tests/test_slurm_thread_caps.py`:

```python
def test_label_descriptions_has_a_steer_stage():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert '"$STAGE" = "steer"' in text and "events_steer_eval.jsonl" in text
    assert "desc_steer_eval_d1.jsonl" in text and "names_primary_eval.jsonl" in text and "--fallback-names" in text
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_descriptions.py::test_label_route_style_key_shares_style_and_shape_across_a_pair tests/test_slurm_thread_caps.py -q`.

- [ ] **Step 3: Implement**

In `label_route` replace `style, shape = choose_d1_style(row["id"]), choose_shape(row["id"])` with:

```python
    key = row.get("style_key") or row["id"]     # steering pairs share one style and shape
    style, shape = choose_d1_style(key), choose_shape(key)
```

In `slurm/label_descriptions.sbatch`, header comment after the steer_names block:

```bash
# STAGE=steer (2026-10-08, steering stop pairs): compose only, for prepare_habitat_instr.py steer-select's
# rows, with the merged primary names (d1 + steer Gemma, from the steer stage) and the d1 Cosmos fallback.
#     STAGE=steer sbatch slurm/label_descriptions.sbatch     # then: prepare_habitat_instr.py finalize-steer
```

and just before `if [ "$NAMER" = "cosmos" ]; then` (next to the `steer_names` block):

```bash
if [ "$STAGE" = "steer" ]; then
    SD=data/prepared_habitat_instr_v3/steer
    for f in events_steer_eval.jsonl names_primary_eval.jsonl; do
        [ -f "$SD/$f" ] || { echo "missing $SD/$f — run prepare_habitat_instr.py steer + steer-select"; exit 1; }
    done
    serve "$COSMOS_MODEL"
    steer_args=(--stage compose --events "$SD/events_steer_eval.jsonl" --names "$SD/names_primary_eval.jsonl"
                --fallback-names "$OUT/names_eval_d1.jsonl" --out "$OUT/desc_steer_eval_d1.jsonl"
                --endpoint "$BASE/v1" --concurrency "$CONCURRENCY")
    echo; echo "=============== compose: steer (pass 1) ==============="
    "$UV" run scripts/label_descriptions.py "${steer_args[@]}" --resume
    echo; echo "=============== compose: steer (pass 2, retries errors) ==============="
    "$UV" run scripts/label_descriptions.py "${steer_args[@]}" --resume
    echo "steer compose done: uv run scripts/prepare_habitat_instr.py finalize-steer"
    exit 0
fi
```

(The template fallback inside `label_route` still runs; `finalize-steer` treats a template row as a composed failure, so the composed files stay purely Cosmos.)

- [ ] **Step 4: Run** — `uv run pytest tests/test_descriptions.py tests/test_slurm_thread_caps.py -q && bash -n slurm/label_descriptions.sbatch && uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py slurm/label_descriptions.sbatch tests/test_descriptions.py tests/test_slurm_thread_caps.py
git commit -m "feat: STAGE=steer compose job; a pair shares speaking style and sentence shape

Co-Authored-By: <your model's trailer>"
```

---

### Task 13: The 4 eval files (`finalize-steer`, Gate 2)

**Files:**
- Modify: `src/rover_vlm/steering.py`
- Modify: `scripts/prepare_habitat_instr.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes: d1 entry records (base per `sample_id`); steering rows (Task 10); compose rows `desc_steer_eval_d1.jsonl` (`id, passed, description, instruction_source`); `descriptions.with_description`; `habitat_data.format_answer_goal`.
- Produces:
  - `STEER_FILES` = `eval_steer_stop_{template,composed}_{entry,goal}` (4 stems), `steer_record(base, row, text, source, form) -> record`, `finalize_steer(base_by_sample, rows, desc_by_id) -> {"files": {stem: [records]}, "stats": dict}`
  - record `habitat_meta`: kept base fields + `pair_id, sibling_id, kind="steer", start_uv_norm, goal_uv_norm, goal_visible (the answer's last v), steer={slice, anchor, anchor_event, side, viewpoint}` + `with_description`'s fields. No `alt_path`.
  - stage `finalize-steer` → `data/prepared_habitat_instr_v3/eval_steer_stop_*.json`, `steer/finalize_meta.json`; prints Gate 2.

- [ ] **Step 1: Write the failing tests**

```python
def _base(sid):
    return {"id": f"{sid}_L", "image": [f"/d/{sid}/fpv_enhanced.png"],
            "conversations": [{"from": "human", "value": "old"}, {"from": "gpt", "value": '{"path":[[0,1,1],[0.5,0.5,0]]}'}],
            "habitat_meta": {"scene_id": "sc", "sample_id": sid, "parent_sample_id": None, "source": "original",
                             "yaw_deg": None, "flavor": "doorway", "framing": "endpoints", "entry_edge": "bottom",
                             "alt_path": [[0, 1, 1]], "candidate_index": 2, "instruction": "old text"}}


def _srow(pid, tag, sib):
    return {"id": f"{pid}_{tag}", "pair_id": pid, "sibling_id": f"{pid}_{sib}", "slice": "stop", "sample_id": "f",
            "viewpoint": "f", "path": [[0.5, 1.0, 1], [0.4, 0.7, 1], [0.5, 0.6, 0 if tag == "B" else 1]],
            "desc_check": {"pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 3], "events": []},
            "anchor": "f:b1", "anchor_event": None, "side": None,
            "template": f"Go left of the sofa, and stop {tag} it."}


def test_finalize_steer_writes_four_records_per_route_and_drops_template_pairs_from_composed():
    rows = [_srow("p1", "A", "B"), _srow("p1", "B", "A"), _srow("p2", "A", "B"), _srow("p2", "B", "A")]
    desc = {"p1_A": {"passed": True, "description": "Keep left of the sofa, then stop just after it.", "instruction_source": "cosmos"},
            "p1_B": {"passed": True, "description": "Keep left of the sofa, then stop well past it.", "instruction_source": "cosmos"},
            "p2_A": {"passed": True, "description": "Go left of the sofa, and stop A it.", "instruction_source": "template"},
            "p2_B": {"passed": True, "description": "x", "instruction_source": "cosmos"}}
    out = st.finalize_steer({"f": _base("f")}, rows, desc)
    files = out["files"]
    assert set(files) == set(st.STEER_FILES) and len(st.STEER_FILES) == 4
    assert len(files["eval_steer_stop_template_entry"]) == 4 and len(files["eval_steer_stop_template_goal"]) == 4
    assert [r["id"] for r in files["eval_steer_stop_composed_entry"]] == ["p1_A", "p1_B"]
    assert out["stats"]["stop:composed_dropped"] == 1 and out["stats"]["stop:pairs"] == 2
    r = files["eval_steer_stop_composed_entry"][0]
    hm = r["habitat_meta"]
    assert hm["steer"] == {"slice": "stop", "anchor": "f:b1", "anchor_event": None, "side": None, "viewpoint": "f"}
    assert hm["sibling_id"] == "p1_B" and "alt_path" not in hm and "candidate_index" not in hm
    assert hm["prompt_form"] == "entry" and "stop just after it" in r["conversations"][0]["value"]
    assert json.loads(r["conversations"][1]["value"])["path"][0] == [0.5, 1.0, 1]
    g = files["eval_steer_stop_template_goal"][1]
    assert g["habitat_meta"]["goal_in_prompt"] and g["habitat_meta"]["goal_visible"] == 0
    assert "[0.5, 0.6]" in g["conversations"][0]["value"]


def test_finalize_steer_counts_unlabelled_pairs():
    rows = [_srow("p1", "A", "B"), _srow("p1", "B", "A")]
    out = st.finalize_steer({"f": _base("f")}, rows, {})
    assert out["stats"]["stop:unlabelled"] == 1 and out["files"]["eval_steer_stop_composed_entry"] == []
    assert len(out["files"]["eval_steer_stop_template_entry"]) == 2


def test_finalize_steer_stage_refuses_missing_compose_rows(tmp_path):
    import types
    prep = _prep()
    sd = tmp_path / "steer"
    sd.mkdir()
    (sd / "events_steer_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (_srow("p1", "A", "B"), _srow("p1", "B", "A"))))
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps([_base("f")]))
    args = types.SimpleNamespace(frames=frames, steer_dir=sd, desc=tmp_path / "none.jsonl", out_dir=tmp_path,
                                 allow_missing=False)
    with pytest.raises(SystemExit, match="no compose row"):
        prep.stage_finalize_steer(args)
    args.allow_missing = True
    prep.stage_finalize_steer(args)
    assert len(json.loads((tmp_path / "eval_steer_stop_template_entry.json").read_text())) == 2
```

(`_prep` is the importlib helper already in this file.)

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_steering.py -q` → FAIL (`finalize_steer` missing).

- [ ] **Step 3: Implement** in `src/rover_vlm/steering.py` (imports: `import copy`; `with_description` from `.descriptions`; `format_answer_goal` from `.habitat_data`):

```python
STEER_FILES = [f"eval_steer_stop_{t}_{f}" for t in ("template", "composed") for f in ("entry", "goal")]
# d1 record fields that still hold for a steering route (everything route-specific is rebuilt)
STEER_META_KEEP = ("scene_id", "sample_id", "parent_sample_id", "source", "yaw_deg", "flavor", "framing", "entry_edge")


def steer_record(base, row, text, source, form):
    """One eval record for a steering route: the frame's d1 record with the route's answer, fresh
    habitat_meta and the description prompt in `form` ("entry" | "goal")."""
    rec = copy.deepcopy(base)
    rec["id"] = row["id"]
    rec["conversations"][1]["value"] = format_answer_goal([tuple(w) for w in row["path"]])
    hm = {k: base["habitat_meta"].get(k) for k in STEER_META_KEEP}
    hm.update({"pair_id": row["pair_id"], "sibling_id": row["sibling_id"], "kind": "steer",
               "start_uv_norm": row["path"][0][:2], "goal_uv_norm": row["path"][-1][:2],
               "goal_visible": int(row["path"][-1][2]),
               "steer": {"slice": row["slice"], "anchor": row["anchor"], "anchor_event": row["anchor_event"],
                         "side": row["side"], "viewpoint": row["viewpoint"]}})
    rec["habitat_meta"] = hm
    return with_description(rec, text, source, form, row["desc_check"])


def finalize_steer(base_by_sample, rows, desc_by_id):
    """{"files": {stem: records}, "stats"} for the 4 eval files. Template text for every pair;
    composed text only when both routes' compose rows passed as Cosmos text (a template fallback
    drops the pair from the composed files only)."""
    files = {k: [] for k in STEER_FILES}
    stats = Counter()
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for pid in sorted(by_pair):
        pair = sorted(by_pair[pid], key=lambda r: r["id"])
        if len(pair) != 2:
            stats["incomplete_pair"] += 1
            continue
        sl, base = pair[0]["slice"], base_by_sample[pair[0]["sample_id"]]
        stats[f"{sl}:pairs"] += 1
        for form in ("entry", "goal"):
            files[f"eval_steer_{sl}_template_{form}"] += [steer_record(base, r, r["template"], "template", form)
                                                          for r in pair]
        labels = [desc_by_id.get(r["id"]) for r in pair]
        if any(lab is None for lab in labels):
            stats[f"{sl}:unlabelled"] += 1
            continue
        if not all(lab.get("passed") and lab.get("instruction_source") == "cosmos" for lab in labels):
            stats[f"{sl}:composed_dropped"] += 1
            continue
        stats[f"{sl}:composed"] += 1
        for form in ("entry", "goal"):
            files[f"eval_steer_{sl}_composed_{form}"] += [steer_record(base, r, lab["description"], "cosmos", form)
                                                          for r, lab in zip(pair, labels)]
    return {"files": files, "stats": dict(stats)}
```

In `scripts/prepare_habitat_instr.py`:

```python
GATE2_MIN_COMPOSED = 0.85


def stage_finalize_steer(args):
    d = args.steer_dir
    base = {}
    for r in json.loads(args.frames.read_text()):
        base.setdefault(r["habitat_meta"]["sample_id"], r)
    rows = [json.loads(x) for x in (d / "events_steer_eval.jsonl").read_text().splitlines() if x.strip()]
    out = steering.finalize_steer(base, rows, _label_rows([args.desc]) if args.desc.exists() else {})
    stats = out["stats"]
    pairs = sum(v for k, v in stats.items() if k.endswith(":pairs"))
    missing = sum(v for k, v in stats.items() if k.endswith(":unlabelled"))
    if missing > 0.01 * max(1, pairs) and not args.allow_missing:
        raise SystemExit(f"{missing} of {pairs} pairs have no compose row (> 1%) — finish "
                         "STAGE=steer sbatch slurm/label_descriptions.sbatch or pass --allow-missing")
    for stem, recs in out["files"].items():
        (args.out_dir / f"{stem}.json").write_text(json.dumps(recs, indent=1))
    meta = {"stats": stats, "files": {k: len(v) for k, v in out["files"].items()}}
    (d / "finalize_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    n, c = stats.get("stop:pairs", 0), stats.get("stop:composed", 0)
    share = c / n if n else 0.0
    print(f"GATE 2 stop: composed {c}/{n} = {share:.3f} -> {'PASS' if share >= GATE2_MIN_COMPOSED else 'FAIL'} "
          f"(need >= {GATE2_MIN_COMPOSED})")
```

subparser `finalize-steer` (`--frames` default `V3_OUT / "eval_desc_entry.json"`, `--steer-dir` default `STEER_DIR`, `--desc` default `D1_LABELS / "desc_steer_eval_d1.jsonl"`, `--out-dir` default `V3_OUT`, `--allow-missing`) and its dispatch entry.

- [ ] **Step 4: Run** — `uv run pytest tests/test_steering.py -q && uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/steering.py scripts/prepare_habitat_instr.py tests/test_steering.py
git commit -m "feat: finalize-steer — 4 stop-pair eval files (text x form), Gate 2

Co-Authored-By: <your model's trailer>"
```

---

### Task 14: Steering metrics in `eval.py`; `SETS=steer` in the eval job

**Files:**
- Modify: `src/rover_vlm/eval.py`
- Modify: `slurm/eval_descriptions.sbatch`
- Test: `tests/test_description_eval.py`, `tests/test_slurm_thread_caps.py`

**Interfaces:**
- Consumes: prediction records from `scripts/evaluate.py` (`id, generated, parsed{path}, gt{path}, metrics, habitat_meta` with `steer`, `sibling_id`, `desc_check`).
- Produces:
  - `description_follow(pred_path, check, gt_end_uv=None, stop_tol_uv=STOP_TOL_UV)` (new keyword, default unchanged)
  - `steer_scores(rec, sib) -> (own: int, swapped: int)` — the end of the prediction is closer to its own route's end than to the sibling's (ties neither); unparseable `(0, 0)`
  - `aggregate_steering_metrics(records) -> dict | None`; `aggregate_habitat_metrics` adds `summary["steering"]`
  - `eval_descriptions.sbatch`: `SETS=desc|steer|steer_goal`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_description_eval.py` (it defines `CAM` at the origin facing +z, `to_image`, `check`, `LEFT_ROUTE`):

```python
from rover_vlm.eval import aggregate_steering_metrics, steer_scores

A_GT, B_GT = [(0.0, 0.8), (0.5, 2.0)], [(0.0, 0.8), (0.5, 2.0), (0.0, 3.5)]   # one route, two stops


def _srec(rid, sib, pred_xz, gt_xz, vp="v1"):
    return {"id": rid, "generated": json.dumps(pred_xz) if pred_xz else "",
            "parsed": {"path": to_image(pred_xz)} if pred_xz else None, "gt": {"path": to_image(gt_xz)},
            "habitat_meta": {"pair_id": rid[:-2], "sibling_id": sib, "desc_check": check([]),
                             "steer": {"slice": "stop", "anchor": None, "anchor_event": None, "side": None, "viewpoint": vp}}}


def _pair(pa, pb, pid="p"):
    return [_srec(f"{pid}_A", f"{pid}_B", pa, A_GT), _srec(f"{pid}_B", f"{pid}_A", pb, B_GT)]


def test_stop_steering_perfect_blind_and_swapped():
    m = aggregate_steering_metrics(_pair(A_GT, B_GT))
    assert m["pair_success"] == 1.0 and m["swap_pair_rate"] == 0.0 and m["closer_to_own_gt"] == 1.0 and m["slice"] == "stop"
    assert aggregate_steering_metrics(_pair(B_GT, B_GT))["pair_success"] == 0.0          # text-blind
    m = aggregate_steering_metrics(_pair(B_GT, A_GT))
    assert m["pair_success"] == 0.0 and m["swap_pair_rate"] == 1.0
    ra, rb = _pair(B_GT, A_GT)
    assert steer_scores(ra, rb) == (0, 1)


def test_steering_unparseable_and_missing_sibling():
    recs = _pair(None, B_GT)
    m = aggregate_steering_metrics(recs)
    assert m["n_pairs"] == 1 and m["pair_success"] == 0.0 and m["record_hit_rate"] == 0.5
    assert aggregate_steering_metrics(recs[:1]) is None
    assert aggregate_steering_metrics([{"id": "x", "habitat_meta": {}}]) is None


def test_steering_block_in_habitat_summary_and_ci():
    recs = _pair(A_GT, B_GT) + _pair(B_GT, B_GT, pid="q")
    for r in recs:
        r["metrics"] = {"mean_point_error": 0.0, "frechet": 0.0, "path_visibility_acc": 1.0, "goal_point_error": 0.0,
                        "goal_visibility_correct": 1, "goal_copy_correct": 1, "start_point_error": 0.0,
                        "start_copy_correct": 1}
        r["gt"]["goal"] = r["gt"]["path"][-1]
    s = aggregate_habitat_metrics(recs)["steering"]
    assert s["n_pairs"] == 2 and s["pair_success"] == 0.5
    lo, hi = s["pair_success_ci"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert set(s) >= {"strict_pair_success_03", "strict_pair_success_05", "identical_prediction_pairs"}


def test_description_follow_stop_tolerance_keyword():
    c = check([])
    path = to_image(LEFT_ROUTE)
    end = [path[-1][0] + 0.04, path[-1][1]]
    assert description_follow(path, c, end)["stop"] == 0
    assert description_follow(path, c, end, stop_tol_uv=0.05)["stop"] == 1
```

(add `import json` at the top of that test file if it is not imported.)

Append to `tests/test_slurm_thread_caps.py`:

```python
def test_eval_descriptions_sets():
    text = (SLURM / "eval_descriptions.sbatch").read_text()
    assert 'SETS="${SETS:-desc}"' in text
    for s in ("eval_steer_stop_template_entry", "eval_steer_stop_composed_goal"):
        assert s in text
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_description_eval.py tests/test_slurm_thread_caps.py -q`.

- [ ] **Step 3: Implement in `eval.py`**

Refactor `description_follow` onto a shared floor helper and add the tolerance keyword:

```python
def _floor_path(path, check):
    """Floor xz of a predicted image path, consecutive duplicates removed (they give a zero
    tangent in _pass_side)."""
    floor = image_to_floor(path or [], check["pose_xz"], check["heading_xz"])
    if len(floor):
        keep = np.r_[True, np.linalg.norm(np.diff(floor, axis=0), axis=1) > 1e-9]
        floor = floor[keep]
    return floor


def description_follow(pred_path, check, gt_end_uv=None, stop_tol_uv=STOP_TOL_UV):
    """Did a predicted path do what its description says? Per event and the stop.

    With `gt_end_uv` (the record's GT terminus, image [x, y]) the stop is the path's last
    waypoint within stop_tol_uv of it; without it, the floor check against goal_xz."""
    floor = _floor_path(pred_path, check)
    n = len(check["events"])
    if len(floor) < 2:
        return {"follows": 0, "events": [0] * n, "stop": 0}
    res = [_event_followed(floor, e) for e in check["events"]]
    if gt_end_uv is not None:
        end = np.asarray(pred_path[-1][:2], float)
        stop = int(np.linalg.norm(end - np.asarray(gt_end_uv[:2], float)) <= stop_tol_uv)
    else:
        stop = int(np.linalg.norm(floor[-1] - np.asarray(check["goal_xz"], float)) <= STOP_TOL_M)
    return {"follows": int(all(res) and stop), "events": res, "stop": stop}
```

Append after `aggregate_description_metrics`:

```python
# --- steering stop pairs (2026-10-08, spec docs/superpowers/specs/2026-10-07-steering-eval-design.md A2) ---
def _xy(path):
    return [p[:2] for p in path]


def steer_scores(rec, sib):
    """(own, swapped) hits of one steering record against its pair sibling: the prediction's end is
    closer to this record's GT end than to the sibling's / the reverse (ties are neither).
    Unparseable = (0, 0). A text-blind model gives one end for both texts: at most one hit a pair."""
    path = (rec.get("parsed") or {}).get("path")
    if not path:
        return 0, 0
    end = np.asarray(path[-1][:2], float)
    d_own = float(np.linalg.norm(end - np.asarray(rec["gt"]["path"][-1][:2], float)))
    d_sib = float(np.linalg.norm(end - np.asarray(sib["gt"]["path"][-1][:2], float)))
    return int(d_own < d_sib), int(d_sib < d_own)


def aggregate_steering_metrics(records):
    """Steering summary for records carrying habitat_meta.steer, else None. Pairs are matched by
    sibling_id; a record whose sibling is absent is skipped. Headline: pair success (both texts
    steered their own way; text-blind = 0), with a 95% bootstrap interval over viewpoints. Also:
    swap control, closer-to-own-GT rate, strict follow (all events + stop at 0.03 and 0.05), and the
    share of pairs whose two generations are identical."""
    rows = [r for r in records if (r.get("habitat_meta") or {}).get("steer")]
    by_id = {r["id"]: r for r in rows}
    pairs, seen = [], set()
    for r in rows:
        sid = r["habitat_meta"].get("sibling_id")
        if r["id"] in seen or sid not in by_id:
            continue
        pairs.append(tuple(sorted((r, by_id[sid]), key=lambda x: x["id"])))
        seen |= {r["id"], sid}
    if not pairs:
        return None
    succ, swap_pair, own_hits, swap_hits, closer, s03, s05, same, groups = ([] for _ in range(9))
    for a, b in pairs:
        (oa, sa), (ob, sb) = steer_scores(a, b), steer_scores(b, a)
        succ.append(int(oa and ob))
        swap_pair.append(int(sa and sb))
        own_hits += [oa, ob]
        swap_hits += [sa, sb]
        for x, y in ((a, b), (b, a)):
            p = (x.get("parsed") or {}).get("path")
            closer.append(int(bool(p) and trajectory_metrics(_xy(p), _xy(x["gt"]["path"]))["mean_point_error"]
                              < trajectory_metrics(_xy(p), _xy(y["gt"]["path"]))["mean_point_error"]))
        for tol, acc in ((STOP_TOL_UV, s03), (0.05, s05)):
            acc.append(int(all((x.get("parsed") or {}).get("path") and description_follow(
                x["parsed"]["path"], x["habitat_meta"]["desc_check"], _gt_end_uv(x), stop_tol_uv=tol)["follows"]
                for x in (a, b))))
        same.append(int(bool(a.get("generated")) and a.get("generated", "").strip() == b.get("generated", "").strip()))
        groups.append(a["habitat_meta"]["steer"]["viewpoint"])
    lo, hi = clustered_bootstrap_ci(succ, groups)
    return {
        "slice": pairs[0][0]["habitat_meta"]["steer"]["slice"],
        "n_pairs": len(pairs),
        "pair_success": float(np.mean(succ)),
        "pair_success_ci": [lo, hi],
        "record_hit_rate": float(np.mean(own_hits)),
        "swap_hit_rate": float(np.mean(swap_hits)),
        "swap_pair_rate": float(np.mean(swap_pair)),
        "closer_to_own_gt": float(np.mean(closer)),
        "strict_pair_success_03": float(np.mean(s03)),
        "strict_pair_success_05": float(np.mean(s05)),
        "identical_prediction_pairs": float(np.mean(same)),
    }
```

In `aggregate_habitat_metrics`, after the `description` block:

```python
    steer = aggregate_steering_metrics(records)
    if steer is not None:
        summary["steering"] = steer
```

and add to its docstring: "Steering sets (habitat_meta.steer) also get a "steering" block (aggregate_steering_metrics)."

- [ ] **Step 4: `SETS` in `slurm/eval_descriptions.sbatch`**

Replace the final `for s in eval_desc_entry ...; do run "$s"; done` line and its echo with:

```bash
SETS="${SETS:-desc}"
STEER_GOAL="eval_steer_stop_template_goal eval_steer_stop_composed_goal"
case "$SETS" in
    desc)       LIST="eval_desc_entry eval_desc_goal eval_desc_pairs eval_desc_shuffled" ;;
    steer)      LIST="eval_steer_stop_template_entry eval_steer_stop_composed_entry $STEER_GOAL" ;;
    steer_goal) LIST="$STEER_GOAL" ;;
    *) echo "SETS must be desc, steer or steer_goal, got '$SETS'"; exit 1 ;;
esac
for s in $LIST; do run "$s"; done
echo "all cells done: uv run scripts/render_desc_videos.py --set <set> --tag $TAG; uv run scripts/visualize_steering.py"
```

and to its header comment:

```bash
#     SETS=steer sbatch slurm/eval_descriptions.sbatch              # the 4 steering stop-pair sets (d1)
#     SETS=steer_goal ADAPTER=outputs/runs_v4/habitat_train_full/adapter TAG=r4 sbatch slurm/eval_descriptions.sbatch
```

- [ ] **Step 5: Run** — `uv run pytest tests/test_description_eval.py tests/test_slurm_thread_caps.py -q && bash -n slurm/eval_descriptions.sbatch && uv run pytest -q`.

- [ ] **Step 6: Commit**

```bash
git add src/rover_vlm/eval.py slurm/eval_descriptions.sbatch tests/test_description_eval.py tests/test_slurm_thread_caps.py
git commit -m "feat: steering stop-pair metrics (end closer to own stop, swap control, CI); SETS=steer eval

Co-Authored-By: <your model's trailer>"
```

---

### Task 15: Report script and README

**Files:**
- Create: `scripts/visualize_steering.py`
- Modify: `src/rover_vlm/overlay.py` (new `description_panel`, moved out of `render_desc_videos.py`)
- Modify: `scripts/render_desc_videos.py` (`panel` becomes a thin wrapper; CLI and tests unchanged)
- Modify: `README.md` (index-only commit — see Step 5)
- Test: `tests/test_steering.py`, `tests/test_render_desc_videos.py` (existing, must still pass)

**Interfaces:**
- Consumes: `outputs/eval_desc/eval_steer_*/<tag>/predictions.json`; `eval.aggregate_steering_metrics`; data files for images.
- Controller Ruling 1 (pre-flight): the tile drawing is NOT copied from `render_desc_videos._tile` (verbatim duplication); it moves to `overlay.description_panel(image_path, rows, px, text_h, max_lines) -> PIL.Image` (frame with GT path thick grey, prediction orange, each row's description underneath, tiles side by side). `render_desc_videos.panel(image_path, rows, px=448)` returns `description_panel(image_path, rows, px, TEXT_H, 5)`; its `_tile` is deleted.
- Produces: `metric_cells(eval_dir, tags) -> list[dict]`, `pair_panel(image_path, a, b, px) -> PIL.Image`, page `outputs/steering_report.html`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering.py`:

```python
def _viz():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "visualize_steering", Path(__file__).resolve().parent.parent / "scripts" / "visualize_steering.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pred(rid, sib, pred, gt, side):
    return {"id": rid, "generated": json.dumps(pred), "parsed": {"path": pred}, "gt": {"path": gt},
            "habitat_meta": {"sibling_id": sib, "instruction": f"go {side}",
                             "desc_check": {"pose_xz": [0, 0], "heading_xz": [0, 1], "goal_xz": [0, 3], "events": []},
                             "steer": {"slice": "stop", "anchor": "x", "anchor_event": None, "side": None, "viewpoint": "v"}}}


def test_metric_cells_reads_predictions_per_set_and_tag(tmp_path):
    vz = _viz()
    run = tmp_path / "eval_steer_stop_template_entry" / "d1"
    run.mkdir(parents=True)
    a = [[0.5, 1.0, 1], [0.3, 0.6, 1]]
    b = [[0.5, 1.0, 1], [0.7, 0.6, 1]]
    (run / "predictions.json").write_text(json.dumps([_pred("p_A", "p_B", a, a, "a"), _pred("p_B", "p_A", b, b, "b")]))
    cells = vz.metric_cells(tmp_path, ["d1", "r4"])
    assert len(cells) == 1
    c = cells[0]
    assert (c["slice"], c["text"], c["form"], c["tag"]) == ("stop", "template", "entry", "d1")
    assert c["steering"]["pair_success"] == 1.0 and c["parse_rate"] == 1.0


def test_pair_panel_size(tmp_path):
    from PIL import Image
    vz = _viz()
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    a = _pred("p_A", "p_B", [[0.5, 1.0, 1], [0.3, 0.6, 1]], [[0.5, 1.0, 1], [0.35, 0.6, 1]], "a")
    b = _pred("p_B", "p_A", [[0.5, 1.0, 1], [0.7, 0.6, 1]], [[0.5, 1.0, 1], [0.65, 0.6, 1]], "b")
    assert vz.pair_panel(img, a, b, px=200).size == (400, 200 + vz.TEXT_H)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_steering.py -q`
Expected: FAIL — `scripts/visualize_steering.py` not found.

- [ ] **Step 3: Write the report script**

Create `scripts/visualize_steering.py`:

```python
"""Steering eval report: stop-pair success per text x prompt form x model, plus a gallery.

CPU only, login node, after `SETS=steer sbatch slurm/eval_descriptions.sbatch` (and the
round-4 control `SETS=steer_goal ... TAG=r4`). Metrics are recomputed from predictions.json
with rover_vlm.eval.aggregate_steering_metrics, so a metric fix never needs a GPU re-run.

    uv run scripts/visualize_steering.py                   # tags d1 and r4
    uv run scripts/visualize_steering.py --tags d1 --n 16

Writes outputs/steering_report.html (self-contained; images inline). Spec:
docs/superpowers/specs/2026-10-07-steering-eval-design.md
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path


from rover_vlm.eval import aggregate_steering_metrics
from rover_vlm.overlay import description_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
CELLS = [("stop", t, f) for t in ("template", "composed") for f in ("entry", "goal")]
TEXT_H = 70


def metric_cells(eval_dir, tags):
    """One row per (slice, text, form, tag) with predictions on disk."""
    out = []
    for s, t, f in CELLS:
        for tag in tags:
            p = Path(eval_dir) / f"eval_steer_{s}_{t}_{f}" / tag / "predictions.json"
            if not p.exists():
                continue
            recs = json.loads(p.read_text())
            out.append({"slice": s, "text": t, "form": f, "tag": tag, "n": len(recs),
                        "parse_rate": sum(bool((r.get("parsed") or {}).get("path")) for r in recs) / max(1, len(recs)),
                        "steering": aggregate_steering_metrics(recs) or {}})
    return out


def pair_panel(image_path, a, b, px=360):
    """Both routes of a pair side by side: GT grey, prediction orange, each text underneath."""
    return description_panel(image_path, [a, b], px, TEXT_H, 4)


def _uri(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def gallery(eval_dir, data_dir, set_name, tag, n):
    p = Path(eval_dir) / set_name / tag / "predictions.json"
    if not p.exists():
        return []
    recs = json.loads(p.read_text())
    images = {r["id"]: r["image"][0] for r in json.loads((Path(data_dir) / f"{set_name}.json").read_text())}
    by_id, seen, out = {r["id"]: r for r in recs}, set(), []
    for r in recs:
        sid = r["habitat_meta"].get("sibling_id")
        if r["id"] in seen or sid not in by_id:
            continue
        a, b = sorted((r, by_id[sid]), key=lambda x: x["id"])
        seen |= {a["id"], b["id"]}
        out.append((a["id"].rsplit("_", 1)[0], _uri(pair_panel(images[a["id"]], a, b))))
        if len(out) >= n:
            break
    return out


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Steering Eval Results</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddddd6;--card:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ececea;--muted:#a3a39c;--line:#33332f;--card:#1f1f1d}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ececea;--muted:#a3a39c;--line:#33332f;--card:#1f1f1d}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}
th:nth-child(-n+4),td:nth-child(-n+4){text-align:left}.muted{color:var(--muted)}
.wrap{overflow-x:auto}figure{margin:16px 0}img{max-width:100%;height:auto;border-radius:6px}
</style></head><body>
<h1>Steering eval</h1>
<p class="muted">Pair success = both texts of a pair steered their own way: each prediction ends closer to its own
stop than to the other text's stop. A text-blind model scores 0. Swap = both predictions followed the
<em>other</em> text. Strict = every event and the stop (0.03 / 0.05).</p>
<div class="wrap"><table><thead><tr><th>slice</th><th>text</th><th>form</th><th>model</th><th>pairs</th>
<th>pair success [95% CI]</th><th>swap</th><th>closer to own GT</th><th>strict .03</th><th>strict .05</th><th>parse</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
__GALLERY__
</body></html>"""


def render(cells, galleries):
    rows = []
    for c in cells:
        s = c["steering"]
        ci = s.get("pair_success_ci")
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            c["slice"], c["text"], c["form"], c["tag"], s.get("n_pairs", 0),
            f"{_pct(s.get('pair_success'))} [{_pct(ci[0]) if ci else '—'}, {_pct(ci[1]) if ci else '—'}]",
            _pct(s.get("swap_pair_rate")), _pct(s.get("closer_to_own_gt")),
            _pct(s.get("strict_pair_success_03")), _pct(s.get("strict_pair_success_05")), _pct(c["parse_rate"]))) + "</tr>")
    gal = "".join(f"<h2>{html.escape(title)}</h2>" + "".join(
        f'<figure><img src="{uri}" alt="{html.escape(pid)}"><figcaption class="muted">{html.escape(pid)}</figcaption></figure>'
        for pid, uri in items) for title, items in galleries)
    return PAGE.replace("__ROWS__", "".join(rows)).replace("__GALLERY__", gal)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=["d1", "r4"])
    p.add_argument("--n", type=int, default=16, help="gallery pairs")
    p.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_desc")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v3")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "steering_report.html")
    args = p.parse_args()
    cells = metric_cells(args.eval_dir, args.tags)
    if not cells:
        raise SystemExit(f"no eval_steer_* predictions under {args.eval_dir}")
    galleries = [("stop pairs — d1, template text, entry form",
                  gallery(args.eval_dir, args.data_dir, "eval_steer_stop_template_entry", "d1", args.n))]
    args.out.write_text(render(cells, galleries))
    for c in cells:
        print(f"{c['slice']:5s} {c['text']:9s} {c['form']:5s} {c['tag']:3s} pairs {c['steering'].get('n_pairs', 0):4d} "
              f"success {_pct(c['steering'].get('pair_success'))}")
    print(f"-> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

Add to `src/rover_vlm/overlay.py` (imports: `import textwrap`, `ImageFont` from PIL) the function below — the body of `render_desc_videos._tile` / `panel`, parameterised — and in `scripts/render_desc_videos.py` replace `_tile` and `panel` with `def panel(image_path, rows, px=448): return description_panel(image_path, rows, px, TEXT_H, 5)` (drop the now-unused imports and `PRED`):

```python
DESC_PRED = (255, 140, 0)


def description_panel(image_path, rows, px, text_h, max_lines):
    """Prediction rows side by side, one tile each: the frame with the GT path (thick grey) and the
    prediction (orange), the row's description underneath (render_desc_videos, visualize_steering)."""
    font = ImageFont.load_default()
    out = Image.new("RGB", (px * len(rows), px + text_h), (0, 0, 0))
    for i, row in enumerate(rows):
        img = Image.open(image_path).convert("RGB").resize((px, px), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        draw_polyline(d, [(p[0], p[1]) for p in row["gt"]["path"]], GT_GREY, px, px, width=8)
        if (row.get("parsed") or {}).get("path"):
            draw_polyline(d, [(p[0], p[1]) for p in row["parsed"]["path"]], DESC_PRED, px, px, width=4)
        tile = Image.new("RGB", (px, px + text_h), (20, 20, 20))
        tile.paste(img, (0, 0))
        text = "\n".join(textwrap.wrap(row["habitat_meta"].get("instruction") or "", width=max(20, px // 8))[:max_lines])
        ImageDraw.Draw(tile).multiline_text((6, px + 4), text, fill=(235, 235, 235), font=font, spacing=2)
        out.paste(tile, (i * px, 0))
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_steering.py tests/test_render_desc_videos.py -q && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: README section, index-only**

The working-tree `README.md` holds the user's uncommitted edits; only our section may be committed. Add this subsection right after the "Route descriptions (d1)" section (in BOTH the working tree and the index):

```markdown
### Steering eval (2026-10-07)

Does d1 *steer* — does changing only the text change the route the way the text says? The d1
pairs cannot tell (both routes end at the same goal and differ only behind obstacles). This
eval-only set keeps each eval frame's ground-truth route and cuts it at two points whose stop phrases
differ: same image, same entry, same route, two texts, two different ends; each route fails the other's
text. Every visible object is named first. Side pairs (one object passed on either side) proved not viable
on these frames — see the spec's Addenda A1/A2: `docs/superpowers/specs/2026-10-07-steering-eval-design.md`.

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py steer-objects               # objects to name (CPU)
    STAGE=steer_names sbatch slurm/label_descriptions.sbatch            # Gemma names them (GPU)
    uv run scripts/prepare_habitat_instr.py steer                       # candidate stop pairs (CPU)
    uv run scripts/prepare_habitat_instr.py steer-select                # one pair per viewpoint; Gate 0 (>= 150)
    uv run scripts/audit_instruction_labels.py --mode steer --n 50      # Gate 1 audit page (>= 90% true)
    STAGE=steer sbatch slurm/label_descriptions.sbatch                  # composed text (GPU, < 1 h)
    uv run scripts/prepare_habitat_instr.py finalize-steer              # 8 files; Gate 2 (>= 85% composed)
    SETS=steer sbatch slurm/eval_descriptions.sbatch                    # d1 on the 4 sets
    SETS=steer_goal ADAPTER=outputs/runs_v4/habitat_train_full/adapter TAG=r4 sbatch slurm/eval_descriptions.sbatch
    uv run scripts/visualize_steering.py                                # outputs/steering_report.html
    uv run scripts/render_desc_videos.py --set eval_steer_stop_template_entry --tag d1 --n 20

Outputs: `data/prepared_habitat_instr_v3/steer/` (objects, candidates, rows, meta),
`data/prepared_habitat_instr_v3/eval_steer_stop_{template,composed}_{entry,goal}.json`,
`outputs/eval_desc/eval_steer_*/<tag>/`. Headline: stop-pair success (text-blind = 0%).
```

Commit only that hunk:

```bash
SCR=<session scratchpad>/readme_idx      # controller Ruling 2: temp files under the scratchpad
mkdir -p "$SCR"
git show HEAD:README.md > "$SCR/README.head.md"
# insert the same subsection into $SCR/README.head.md at the same place (after the d1 section), with the Edit tool
git update-index --cacheinfo 100644,"$(git hash-object -w "$SCR/README.head.md")",README.md
git diff --cached README.md     # read it: only the new subsection, none of the user's edits
git add scripts/visualize_steering.py src/rover_vlm/overlay.py scripts/render_desc_videos.py tests/test_steering.py
git commit -m "feat: steering report script (shared description_panel); README steering eval section

Co-Authored-By: <your model's trailer>"
git diff --stat README.md       # the user's own edits are still unstaged
rm -rf "$SCR"
```

---

## After the code: run order and gates (the user runs the GPU parts)

1. **Objects (CPU, here):** `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; uv run scripts/prepare_habitat_instr.py steer-objects`.
2. **Names (user, GPU):** `STAGE=steer_names sbatch slurm/label_descriptions.sbatch` (Gemma).
3. **Stop pairs (CPU, here):** `uv run scripts/prepare_habitat_instr.py steer` — read `steer/steer_meta.json` (reasons, errors). Long runs → `setsid nohup ... > outputs/steer_stage.log 2>&1 &`.
4. **Select + Gate 0:** `uv run scripts/prepare_habitat_instr.py steer-select`. If < 150 pairs, stop and bring the drop-reason counts to the user (no silent threshold change — spec risk note).
5. **Gate 1 audit:** `uv run scripts/audit_instruction_labels.py --mode steer --n 50` → publish `outputs/steer_audit.html` as an artifact with a db `marks` capability (as earlier audits); ≥ 90% of pairs "true". Freeze thresholds.
6. **Compose (user):** `STAGE=steer sbatch slurm/label_descriptions.sbatch`.
7. **Finalize + Gate 2:** `uv run scripts/prepare_habitat_instr.py finalize-steer`.
8. **Eval (user):** `SETS=steer sbatch slurm/eval_descriptions.sbatch`; `SETS=steer_goal ADAPTER=outputs/runs_v4/habitat_train_full/adapter TAG=r4 sbatch slurm/eval_descriptions.sbatch` (≤ 3 SLURM jobs at once).
9. **Report:** `uv run scripts/visualize_steering.py` (publish as an artifact) and `render_desc_videos.py` for `eval_steer_stop_template_entry`, `--n 20`.
