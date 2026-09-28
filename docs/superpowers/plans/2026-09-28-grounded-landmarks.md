# Grounded Landmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v2 instruction labels' VLM-chosen landmarks with geometry-chosen targets (the opening the route uses, else the object it bends around) that Cosmos only *names*, with a geometry audit, an HM3D name check and a human audit as gates.

**Architecture:** A new pure-numpy module `rover_vlm/landmarks.py` finds each record's target on the rover's own occupancy map and projects it into the image; a new `landmarks` stage of `prepare_habitat_instr.py` writes one row per record. Prompt `i4` in `instructions.py` shows Cosmos only the outlined target and turns its reply into an instruction (paraphrase, else template). A standalone Habitat script renders per-pixel HM3D instance frames for the name check. Existing scripts gain modes instead of new scripts being added.

**Tech Stack:** Python 3.12 (uv), numpy, Pillow, pytest; Cosmos3-Nano over the existing vLLM serve lib; habitat-sim in the `habitat` conda env (semantic replay only).

**Spec:** `docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md`

## Global Constraints

- Side word = the route's side (bows that way from the straight entry→goal line); taken from candidate metadata, never from a VLM.
- Egocentric "on/to your left|right" stays rejected (`side:egocentric`).
- Camera model: level pinhole, hfov 90°, height 0.8 m, at `meta["pose_xz"]` facing `meta["heading_xz"]`; image right = `(-h_z, h_x)`; normalised `u = 0.5 + 0.5·lat/fwd`, `v = 0.5 + 0.5·(0.8 − height)/fwd`.
- Occupancy map `state.npz`: `state` uint8 (0 unknown, 1 free, 2 occupied), `mpp`, `x_min`, `z_min`; cell (row, col) ↔ world `x = x_min + (col+0.5)·mpp`, `z = z_min + (row+0.5)·mpp`.
- Openings are searched only in `flavor == "doorway"` frames (prototype 2026-09-28: detour frames' best landmark is the obstacle).
- Records with no target get **no instruction** and a counted reason code; their plain round-4 copy stays in training. Nothing is dropped silently.
- Thread caps in every compute job: `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`. The landmarks stage uses at most 4 worker processes.
- Never load Qwen/Cosmos/habitat on the login node; GPU work is sbatch, submitted by the user. SLURM max 3 jobs.
- Semantic rendering needs the **full A100** (`--gres=gpu:a100:1`); MIG slices cannot render.
- Never write into `spaceitup-habitat/` or the shared dataset dirs under `/nfs/projects/spaceitup/`; outputs go under this repo's `outputs/` and `data/`.
- Every script starts with a docstring saying what it does and how to run it (repo rule).

## Review Focus

1. A route whose polyline has fewer than 2 points or a `candidate_index` out of range → reason `bad_candidate`, never a crash that kills the stage (test in Task 2).
2. A sample directory without `state.npz` (older rounds / missing file) → reason `no_state` (test in Task 2).
3. Cosmos returns a name that itself contains a side word ("left cabinet") → the filled template trips `side:opposite`/`side:missing` and the record is rejected, not silently kept (test in Task 4).
4. A landmarks file that does not cover every labelset record → the label client refuses to start and names the missing count, instead of labelling with no target (test in Task 5).
5. Re-running `finalize-general` over i4 rows keeps the instruction's `source` (`cosmos`/`template`) in `habitat_meta.instruction_source` so later analysis can split by it (test in Task 6).

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/rover_vlm/landmarks.py` | create | camera + grid geometry, opening/object search, `select_target` |
| `tests/test_landmarks.py` | create | synthetic-grid tests + one real-sample projection test |
| `scripts/prepare_habitat_instr.py` | modify | new `landmarks` stage; finalize keeps instruction source |
| `src/rover_vlm/instructions.py` | modify | i4 prompt, target overlay, templates, reply parsing, i4 rules, rescore |
| `tests/test_instructions.py` | modify | i4 tests |
| `scripts/label_instructions.py` | modify | `--prompt i4 --landmarks ...` path |
| `slurm/label_instructions.sbatch` | modify | `PROMPT`/`STAGE` knobs; i4 gate split |
| `scripts/audit_instruction_labels.py` + `.html` | modify | `geometry` mode, target outline, `semantic` check |
| `scripts/render_semantic_frames.py` | create | habitat-env semantic replay → `outputs/landmark_semantics/*.npz` |
| `slurm/render_semantic_frames.sbatch` | create | full-A100 job for the above |
| `README.md`, `NOTES.md` | modify | how to run the new pipeline |

---

### Task 0: Commit today's pending v2 fixes

These are already written and tested (2026-09-28): the `side:egocentric` rule, finalize re-scoring, the audit page.

**Files:**
- Modify (already modified): `src/rover_vlm/instructions.py`, `tests/test_instructions.py`, `scripts/prepare_habitat_instr.py`
- Add (already present): `scripts/audit_instruction_labels.py`, `scripts/audit_instruction_labels.html`

- [ ] **Step 1: Run the test suite**

Run: `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; uv run pytest -q`
Expected: `396 passed`

- [ ] **Step 2: Commit only these files**

```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py scripts/prepare_habitat_instr.py \
        scripts/audit_instruction_labels.py scripts/audit_instruction_labels.html
git commit -m "feat: reject egocentric side phrasing, rescore in finalize, label audit page

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

(Do not stage `README.md`, `scripts/evaluate.py` or the untracked real-eval files — they belong to other work.)

---

### Task 1: Camera, grid and small geometry helpers

**Files:**
- Create: `src/rover_vlm/landmarks.py`
- Test: `tests/test_landmarks.py`

**Interfaces:**
- Produces: `Camera(pose_xz, heading_xz)` with `.pos`, `.h`, `.right`, `.uv(xz, height=0.0) -> (uv (N,2), fwd (N,))`, `.in_frame(uv, fwd) -> bool (N,)`; `Grid(state, mpp, x_min, z_min)` with `.load(path)`, `.rc(xz)`, `.xz(rc)`, `.cell(rc) -> int`, `.first_occupied(a_xz, b_xz) -> (row, col) | None`; `densify(points, step) -> (M,2)`; `components(mask) -> (labels, n)`; `convex_hull(points) -> list[tuple]`; constants `FREE, OCC, CAM_HEIGHT_M, MIN_FWD_M`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_landmarks.py
"""Grounded landmarks: geometry on synthetic occupancy grids + one real-sample check."""
import json
from pathlib import Path

import numpy as np
import pytest

from rover_vlm.landmarks import (
    FREE, OCC, Camera, Grid, components, convex_hull, densify,
)


def test_camera_centre_and_sides():
    cam = Camera([0.0, 0.0], [0.0, 1.0])
    uv, fwd = cam.uv([[0.0, 2.0]])
    assert fwd[0] == pytest.approx(2.0)
    assert uv[0, 0] == pytest.approx(0.5)
    assert uv[0, 1] == pytest.approx(0.5 + 0.5 * 0.8 / 2.0)
    right, _ = cam.uv([cam.pos + 2.0 * cam.h + 0.5 * cam.right])
    left, _ = cam.uv([cam.pos + 2.0 * cam.h - 0.5 * cam.right])
    assert right[0, 0] > 0.5 > left[0, 0]
    top, _ = cam.uv([[0.0, 2.0]], height=0.8)
    assert top[0, 1] == pytest.approx(0.5)             # camera height -> horizon


def test_camera_in_frame():
    cam = Camera([0.0, 0.0], [0.0, 1.0])
    uv, fwd = cam.uv([[0.0, 2.0], [0.0, -2.0], [0.0, 0.1]])
    assert cam.in_frame(uv, fwd).tolist() == [True, False, False]


def test_grid_roundtrip_and_ray():
    s = np.full((40, 40), FREE, np.uint8)
    s[20, 10:30] = OCC                                   # wall at z = 1.0..1.05
    g = Grid(s, 0.05, 0.0, 0.0)
    rc = g.rc([[0.52, 1.02]])[0]
    assert (round(rc[0]), round(rc[1])) == (20, 10)
    assert np.allclose(g.xz([[20, 10]])[0], [0.525, 1.025])
    assert g.cell((20, 15)) == OCC and g.cell((-1, 0)) == 0
    assert g.first_occupied([1.0, 0.2], [1.0, 1.8]) == (20, 20)
    assert g.first_occupied([0.2, 0.2], [0.2, 1.8]) is None


def test_densify_components_hull():
    d = densify(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.25)
    assert len(d) == 5 and np.allclose(d[-1], [1.0, 0.0])
    m = np.zeros((6, 6), bool); m[0:2, 0:2] = True; m[4:6, 4:6] = True
    lab, n = components(m)
    assert n == 2 and lab[0, 0] != lab[5, 5]
    h = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(h) == 4 and (0.5, 0.5) not in h
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_landmarks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rover_vlm.landmarks'`

- [ ] **Step 3: Implement**

```python
# src/rover_vlm/landmarks.py
"""Grounded landmarks: which thing an instruction should name, found by geometry.

A v2 human audit (2026-09-28) found Cosmos names real objects well but picks the wrong
one ~20% of the time (75% on the second route of a pair): choosing which object a drawn
route goes around is a geometric judgment VLMs fail. Here geometry chooses the target on
the rover's own occupancy map (`state.npz`, built from this very frame) and projects it
into the image; Cosmos only names it (prompt i4, instructions.py).

Pure numpy + PIL, CPU only. Spec: docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md
"""

from collections import deque
from pathlib import Path

import numpy as np

UNKNOWN, FREE, OCC = 0, 1, 2
CAM_HEIGHT_M = 0.8        # Habitat camera height above the floor
MIN_FWD_M = 0.3           # closer than this along the view axis = not usable in the image


class Camera:
    """Level pinhole, hfov 90°, at pose_xz facing heading_xz; normalised image coords.

    Verified 2026-09-28: projecting chord.goal reproduces fpv_paths' goal pixel
    (256.2, 282.0) vs (256.0, 282.0) on 00366-fxbzYAGkrtm_c012.
    """

    def __init__(self, pose_xz, heading_xz):
        self.pos = np.asarray(pose_xz, float)
        h = np.asarray(heading_xz, float)
        self.h = h / np.linalg.norm(h)
        self.right = np.array([-self.h[1], self.h[0]])

    def uv(self, xz, height=0.0):
        d = np.atleast_2d(np.asarray(xz, float)) - self.pos
        fwd, lat = d @ self.h, d @ self.right
        with np.errstate(divide="ignore", invalid="ignore"):
            u = 0.5 + 0.5 * lat / fwd
            v = 0.5 + 0.5 * (CAM_HEIGHT_M - height) / fwd
        return np.stack([u, v], 1), fwd

    @staticmethod
    def in_frame(uv, fwd):
        return (fwd > MIN_FWD_M) & (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)


class Grid:
    """The rover's occupancy map: row <- z, col <- x (same frame as roverbench's OccupancyMap)."""

    def __init__(self, state, mpp, x_min, z_min):
        self.s = np.asarray(state)
        self.mpp, self.x0, self.z0 = float(mpp), float(x_min), float(z_min)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["state"], z["mpp"], z["x_min"], z["z_min"])

    def rc(self, xz):
        xz = np.atleast_2d(np.asarray(xz, float))
        return np.stack([(xz[:, 1] - self.z0) / self.mpp - 0.5, (xz[:, 0] - self.x0) / self.mpp - 0.5], 1)

    def xz(self, rc):
        rc = np.atleast_2d(np.asarray(rc, float))
        return np.stack([self.x0 + (rc[:, 1] + 0.5) * self.mpp, self.z0 + (rc[:, 0] + 0.5) * self.mpp], 1)

    def cell(self, rc):
        r, c = int(round(rc[0])), int(round(rc[1]))
        if 0 <= r < self.s.shape[0] and 0 <= c < self.s.shape[1]:
            return int(self.s[r, c])
        return UNKNOWN

    def first_occupied(self, a_xz, b_xz):
        """First occupied cell on the segment a -> b (a's own cell excluded), or None."""
        for p in densify(np.array([a_xz, b_xz], float), self.mpp / 2)[1:]:
            rc = self.rc(p)[0]
            if self.cell(rc) == OCC:
                return int(round(rc[0])), int(round(rc[1]))
        return None


def densify(points, step):
    pts = np.asarray(points, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t / n for t in range(1, n + 1)]
    return np.array(out)


def components(mask):
    """8-connected labelling (no scipy in this env). Returns (labels, n)."""
    lab = np.zeros(mask.shape, int)
    n = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                            and mask[rr, cc] and not lab[rr, cc]):
                        lab[rr, cc] = n
                        q.append((rr, cc))
    return lab, n


def convex_hull(points):
    """Monotone-chain hull of 2D points, counter-clockwise, no repeated first point."""
    p = sorted({(round(float(x), 4), round(float(y), 4)) for x, y in points})
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], q) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, up = half(p), half(p[::-1])
    return lo[:-1] + up[:-1]
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_landmarks.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/landmarks.py tests/test_landmarks.py
git commit -m "feat: landmarks geometry helpers (camera, occupancy grid, hull)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Target selection (opening, object, reason codes)

**Files:**
- Modify: `src/rover_vlm/landmarks.py` (append)
- Test: `tests/test_landmarks.py` (append)

**Interfaces:**
- Consumes: Task 1 names.
- Produces: `select_target(sample_dir: Path, candidate_index: int) -> dict` with keys `target` (`"opening"|"object"|"none"`), `reason` (None or one of `REASONS`), `polygon` (list of `[u, v]`, normalised, may be empty), `bbox` (`[u0, v0, u1, v1]` or None), `world_xz` (`[x, z]` or None), `diag` (dict); `find_opening(grid, route) -> dict | None`; `enclosed_blobs(grid, cam, route) -> list[dict]`; `crossing_excursion(route) -> float`; constants `REASONS`, `MIN_OPENING_M=0.6`, `MAX_OPENING_M=2.5`, `DOOR_HEIGHT_M=2.0`, `OBJECT_TOP_M=1.0`, `MIN_VISIBLE_CELLS=4`, `MAX_CROSSING_M=0.3`, `MIN_POLY_AREA=0.002`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_landmarks.py`)

```python
from rover_vlm.landmarks import (
    REASONS, crossing_excursion, find_opening, select_target,
)


def _write(d, state, route, *, flavor="detour", pose=(2.5, 0.2), heading=(0.0, 1.0), extra_cands=()):
    """A synthetic sample dir: 5 m x 5 m map at 5 cm, camera at `pose` facing +z."""
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in route]}]
    cands += [{"label": "right", "polyline_xz": c} for c in extra_cands]
    meta = {"pose_xz": list(pose), "heading_xz": list(heading), "flavor": flavor,
            "chord": {"goal": [route[-1][0], 0.0, route[-1][1]], "floor_height": 0.0},
            "candidates": cands}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def _free():
    return np.full((100, 100), FREE, np.uint8)


def _left(cam_h=(0.0, 1.0)):
    # image-left direction for heading +z is -(right) = -(-1, 0) = (+1, 0)
    return np.array([1.0, 0.0])


def test_object_between_route_and_chord(tmp_path):
    s = _free()
    s[45:56, 45:56] = OCC                                # 0.55 m box centred (2.5, 2.5)
    L = _left()
    route = [(2.5, 0.2), tuple(np.array([2.5, 1.5]) + 0.9 * L), tuple(np.array([2.5, 3.5]) + 0.9 * L), (2.5, 4.6)]
    t = select_target(_write(tmp_path / "s", s, route), 0)
    assert t["target"] == "object" and t["reason"] is None
    assert np.allclose(t["world_xz"], [2.5, 2.5], atol=0.2)
    u0, v0, u1, v1 = t["bbox"]
    # only the part of the box between chord (x = 2.5, image centre) and route (image left)
    # is enclosed, so the outline sits just left of centre
    assert u0 < 0.5 and u1 <= 0.55
    assert len(t["polygon"]) >= 3


def test_opening_in_doorway_wall(tmp_path):
    s = _free()
    s[60:62, :] = OCC                                    # wall across z = 3.0
    s[60:62, 55:75] = FREE                               # 1.0 m gap, x = 2.75 .. 3.75
    route = [(2.5, 0.2), (3.25, 2.0), (3.25, 3.6), (2.5, 4.6)]
    t = select_target(_write(tmp_path / "s", s, route, flavor="doorway"), 0)
    assert t["target"] == "opening"
    assert t["diag"]["width_m"] == pytest.approx(1.0, abs=0.15)


def test_hidden_opening_falls_back_to_the_blocking_object(tmp_path):
    s = _free()
    s[60:62, :] = OCC
    s[60:62, 55:75] = FREE
    s[30:32, 55:72] = OCC                                # screen x 2.75..3.6 at z 1.5, on the camera->gap ray
    # route stays on the image-left side of the chord (x >= 2.5) and passes outside the screen
    route = [(2.5, 0.2), (3.9, 1.0), (3.9, 2.0), (3.25, 2.6), (3.25, 3.6), (2.5, 4.6)]
    assert crossing_excursion(np.array(route, float)) == 0.0
    t = select_target(_write(tmp_path / "s", s, route, flavor="doorway"), 0)
    assert t["target"] == "object"
    assert 2.6 < t["world_xz"][0] < 3.7 and 1.3 < t["world_xz"][1] < 1.8   # the screen, not the wall


def test_route_crossing_chord_dropped(tmp_path):
    s = _free()
    s[45:56, 45:56] = OCC
    route = [(2.5, 0.2), (3.5, 1.5), (1.5, 3.0), (2.5, 4.6)]   # bends right then left
    assert crossing_excursion(np.array(route, float)) > 0.3
    t = select_target(_write(tmp_path / "s", s, route), 0)
    assert t["target"] == "none" and t["reason"] == "route_crosses_chord"


def test_nothing_visible(tmp_path):
    route = [(2.5, 0.2), (3.4, 2.0), (2.5, 4.6)]
    t = select_target(_write(tmp_path / "s", _free(), route), 0)
    assert t == {**t, "target": "none", "reason": "no_visible_target"}


def test_bad_inputs_are_reasons_not_crashes(tmp_path):
    d = _write(tmp_path / "s", _free(), [(2.5, 0.2), (2.5, 4.6)])
    assert select_target(d, 7)["reason"] == "bad_candidate"
    (d / "state.npz").unlink()
    assert select_target(d, 0)["reason"] == "no_state"
    assert set(REASONS) >= {"bad_candidate", "no_state", "no_visible_target",
                            "route_crosses_chord", "both_sides", "target_too_small"}


REAL = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/train/"
            "00366-fxbzYAGkrtm/samples/00366-fxbzYAGkrtm_c012")


@pytest.mark.skipif(not REAL.exists(), reason="shared Habitat dataset not mounted")
def test_projection_matches_stored_goal_pixel():
    meta = json.loads((REAL / "meta.json").read_text())
    fpv = json.loads((REAL / "fpv_paths.json").read_text())
    g = np.array(meta["chord"]["goal"])[[0, 2]]
    uv, _ = Camera(meta["pose_xz"], meta["heading_xz"]).uv([g])
    W, H = fpv["image_size"]
    assert uv[0, 0] * W == pytest.approx(fpv["goal"]["uv"][0], abs=1.0)
    assert uv[0, 1] * H == pytest.approx(fpv["goal"]["uv"][1], abs=1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_landmarks.py -q`
Expected: FAIL — `ImportError: cannot import name 'REASONS'`

- [ ] **Step 3: Implement** (append to `src/rover_vlm/landmarks.py`; add `import json` to the imports)

```python
MIN_OPENING_M, MAX_OPENING_M = 0.6, 2.5   # a rover-passable gap, not a whole room
DOOR_HEIGHT_M = 2.0                       # top of an opening's outline
OBJECT_TOP_M = 1.0                        # top of an object's outline (map has no heights)
MIN_VISIBLE_CELLS = 4                     # visible occupied cells for an object to count
MAX_CROSSING_M = 0.3                      # route may stray this far to the other side of the chord
MIN_POLY_AREA = 0.002                     # outline area as a fraction of the image
REASONS = ("bad_candidate", "no_state", "route_crosses_chord", "no_visible_target",
           "both_sides", "target_too_small")


def _none(reason, **diag):
    return {"target": "none", "reason": reason, "polygon": [], "bbox": None,
            "world_xz": None, "diag": diag}


def _chord_frame(route):
    a, g = route[0], route[-1]
    ch = (g - a) / max(np.linalg.norm(g - a), 1e-9)
    return a, np.array([-ch[1], ch[0]])


def crossing_excursion(route):
    """How far (m) the route strays to the opposite side of the straight start->goal line
    from the side it mostly takes. Large = the route bends both ways (no single side word)."""
    pts = densify(route, 0.05)
    a, n = _chord_frame(route)
    off = (pts - a) @ n
    main = np.sign(off.mean()) or 1.0
    return float(max(0.0, (-main * off).max()))


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _finish(kind, poly, world_xz, **diag):
    poly = [[float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))] for u, v in poly]
    area = _poly_area(poly)
    if area < MIN_POLY_AREA:
        return _none("target_too_small", area=round(area, 5), **diag)
    us, vs = zip(*poly)
    return {"target": kind, "reason": None, "polygon": poly,
            "bbox": [min(us), min(vs), max(us), max(vs)],
            "world_xz": [float(world_xz[0]), float(world_xz[1])],
            "diag": {"area": round(area, 5), **diag}}


def find_opening(grid, route):
    """Narrowest rover-passable gap between occupied cells that the route passes through.

    At each route point, scan perpendicular both ways for the first occupied cell; a point
    with a hit on both sides within MAX_OPENING_M is a gap. Returns the narrowest one with
    width in [MIN_OPENING_M, MAX_OPENING_M] as {"width_m", "centre", "flanks"} or None.
    """
    pts = densify(route, 0.05)
    best = None
    for i in range(1, len(pts) - 1):
        d = pts[i + 1] - pts[i - 1]
        d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([-d[1], d[0]])
        hits = []
        for sgn in (1.0, -1.0):
            hit = None
            for t in np.arange(grid.mpp, MAX_OPENING_M, grid.mpp / 2):
                q = pts[i] + sgn * n * t
                if grid.cell(grid.rc(q)[0]) == OCC:
                    hit = q
                    break
            hits.append(hit)
        if hits[0] is None or hits[1] is None:
            continue
        w = float(np.linalg.norm(hits[0] - hits[1]))
        if MIN_OPENING_M <= w <= MAX_OPENING_M and (best is None or w < best["width_m"]):
            best = {"width_m": w, "centre": pts[i], "flanks": (hits[0], hits[1])}
    return best


def enclosed_blobs(grid, cam, route):
    """Occupied blobs inside the region between the route and its straight chord.

    Each: {"label", "cells", "visible_xz" (cells the camera sees first, in frame),
    "side" (mean signed offset from the chord; its sign is the side the blob is on)}.
    """
    from PIL import Image, ImageDraw

    H, W = grid.s.shape
    ring = grid.rc(np.vstack([route, route[:1]]))
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(float(c) + 0.5, float(r) + 0.5) for r, c in ring], fill=1)
    inside = (np.asarray(im) > 0) & (grid.s == OCC)
    lab, n = components(inside)
    a, nrm = _chord_frame(route)
    out = []
    for k in range(1, n + 1):
        rcs = np.argwhere(lab == k)
        xz = grid.xz(rcs)
        uv, fwd = cam.uv(xz)
        ok = cam.in_frame(uv, fwd)
        vis = np.zeros(len(xz), bool)
        for i in np.nonzero(ok)[0]:
            first = grid.first_occupied(cam.pos, xz[i])
            vis[i] = first is None or lab[first] == k
        out.append({"label": k, "cells": len(rcs), "visible_xz": xz[vis],
                    "side": float(((xz - a) @ nrm).mean())})
    return out


def select_target(sample_dir, candidate_index):
    """The thing an instruction for this route should name, or a reason there is none.

    Order: the opening the route passes through (doorway frames, if the camera sees it),
    else the most visible occupied blob between the route and its chord, else none.
    """
    sd = Path(sample_dir)
    meta = json.loads((sd / "meta.json").read_text())
    cands = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cands)) or len(cands[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    if not (sd / "state.npz").exists():
        return _none("no_state")
    route = np.asarray(cands[candidate_index]["polyline_xz"], float)
    exc = crossing_excursion(route)
    if exc > MAX_CROSSING_M:
        return _none("route_crosses_chord", excursion_m=round(exc, 3))
    grid = Grid.load(sd / "state.npz")
    cam = Camera(meta["pose_xz"], meta["heading_xz"])

    if meta.get("flavor") == "doorway":
        op = find_opening(grid, route)
        if op is not None:
            c = op["centre"]
            uvc, fc = cam.uv([c])
            a, b = op["flanks"]
            _, fab = cam.uv([a, b])
            if (cam.in_frame(uvc, fc)[0] and (fab > MIN_FWD_M).all()
                    and grid.first_occupied(cam.pos, c) is None):
                quad = np.vstack([cam.uv([a])[0], cam.uv([b])[0],
                                  cam.uv([b], DOOR_HEIGHT_M)[0], cam.uv([a], DOOR_HEIGHT_M)[0]])
                return _finish("opening", quad, c, width_m=round(op["width_m"], 3))

    blobs = [b for b in enclosed_blobs(grid, cam, route) if len(b["visible_xz"]) >= MIN_VISIBLE_CELLS]
    if not blobs:
        return _none("no_visible_target")
    if len({np.sign(b["side"]) for b in blobs if abs(b["side"]) > 0.05}) > 1:
        return _none("both_sides", n_blobs=len(blobs))
    best = max(blobs, key=lambda b: len(b["visible_xz"]))
    xz = best["visible_xz"]
    pts = np.vstack([cam.uv(xz)[0], cam.uv(xz, OBJECT_TOP_M)[0]])
    return _finish("object", convex_hull(pts), xz.mean(0),
                   visible_cells=len(xz), blob_cells=best["cells"])
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_landmarks.py -q`
Expected: `11 passed` (or `10 passed, 1 skipped` where the shared dataset is not mounted)

If `test_hidden_opening_falls_back` or `test_opening_in_doorway_wall` fails, print `select_target(...)` in the test and adjust the synthetic geometry (not the thresholds) — the thresholds are tuned at the geometry audit (Task 3), not here.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/landmarks.py tests/test_landmarks.py
git commit -m "feat: select_target - opening, enclosed object, or a reason code

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `landmarks` stage + geometry audit page

**Files:**
- Modify: `scripts/prepare_habitat_instr.py` (new sub-command, module docstring)
- Modify: `scripts/audit_instruction_labels.py`, `scripts/audit_instruction_labels.html`
- Test: `tests/test_landmarks.py` (append)

**Interfaces:**
- Consumes: `select_target` (Task 2).
- Produces: `data/prepared_habitat_instr_v2/landmarks_{train,eval}.jsonl` — one JSON row per labelset record: `{"id", "sample_dir", "candidate_index", "side", "kind", "pair_id", **select_target(...)}`; `landmarks_meta.json` with per-split reason counts. `prepare_habitat_instr.landmark_row(rec) -> dict` (importable for tests). Audit script `--mode geometry --landmarks <jsonl>`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_landmarks.py`)

```python
def test_landmark_row_carries_record_fields(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prep", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    s = _free()
    s[45:56, 45:56] = OCC
    L = _left()
    route = [(2.5, 0.2), tuple(np.array([2.5, 1.5]) + 0.9 * L), tuple(np.array([2.5, 3.5]) + 0.9 * L), (2.5, 4.6)]
    d = _write(tmp_path / "scene_c000", s, route)
    rec = {"id": "scene_c000_L", "image": [str(d / "fpv_enhanced.png")],
           "habitat_meta": {"candidate_index": 0, "side": "left", "kind": "instructed", "pair_id": None}}
    row = prep.landmark_row(rec)
    assert row["id"] == "scene_c000_L" and row["side"] == "left" and row["target"] == "object"
    json.dumps(row)                                      # serialisable as a JSONL row
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_landmarks.py::test_landmark_row_carries_record_fields -q`
Expected: FAIL — `AttributeError: module 'prep' has no attribute 'landmark_row'`

- [ ] **Step 3: Implement the stage** in `scripts/prepare_habitat_instr.py`

Add to the imports: `from multiprocessing import Pool` and `from rover_vlm.landmarks import select_target`. Add these functions above `def main()`:

```python
LANDMARK_WORKERS = 4   # shared 96-core box: modest, and each worker is single-threaded numpy


def landmark_row(rec):
    """One landmarks JSONL row for a labelset record (see rover_vlm.landmarks.select_target)."""
    hm = rec["habitat_meta"]
    sample_dir = Path(rec["image"][0]).parent
    try:
        t = select_target(sample_dir, hm["candidate_index"])
    except Exception as e:  # noqa: BLE001 — one bad sample is a counted reason, not a dead stage
        t = {"target": "none", "reason": "error", "polygon": [], "bbox": None,
             "world_xz": None, "diag": {"error": f"{type(e).__name__}: {e}"}}
    return {"id": rec["id"], "sample_dir": str(sample_dir), "candidate_index": hm["candidate_index"],
            "side": hm["side"], "kind": hm.get("kind"), "pair_id": hm.get("pair_id"), **t}


def stage_landmarks(args):
    d = args.out_dir
    meta = {}
    for name in SPLITS:
        recs = json.loads((d / f"labelset_{name}.json").read_text())
        if args.limit:
            recs = recs[: args.limit]
        with Pool(LANDMARK_WORKERS) as pool:
            rows = pool.map(landmark_row, recs, chunksize=32)
        out = d / f"landmarks_{name}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts = Counter(r["target"] if r["target"] != "none" else f"none:{r['reason']}" for r in rows)
        pairs = Counter(r["target"] for r in rows if r["pair_id"])
        meta[name] = {"records": len(rows), "targets": dict(counts), "pair_targets": dict(pairs)}
        print(f"{name}: {json.dumps(meta[name])} -> {out}")
    (d / "landmarks_meta.json").write_text(json.dumps(meta, indent=2))
```

Register the sub-command in `main()` after the `finalize-general` parser, and add it to the dispatch dict at the end of `main()`:

```python
    g = sub.add_parser("landmarks", help="v2: grounded landmark target per labelset record (CPU)")
    g.add_argument("--out-dir", type=Path, default=V2_OUT)
    g.add_argument("--limit", type=int, default=0, help="first N records per split (smoke run)")
    args = p.parse_args()
    {"pairs": stage_pairs, "finalize": stage_finalize, "general": stage_general,
     "finalize-general": stage_finalize_general, "landmarks": stage_landmarks}[args.stage](args)
```

Add to the module docstring's usage block:

```
    # grounded landmarks (2026-09-28): one target per labelset record, CPU, ~40 min
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py landmarks
```

(Each of the 4 workers runs single-threaded numpy, so the per-process caps are 1: 4 processes × 1 thread.)

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_landmarks.py -q`
Expected: all pass.

- [ ] **Step 5: Smoke-run on real data (CPU)**

Run:
```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run scripts/prepare_habitat_instr.py landmarks --limit 200
```
Expected: two lines like `train: {"records": 200, "targets": {"object": ~120, "opening": ~65, "none:...": ~15}, ...}` (prototype 2026-09-28 on 200 train records: object 119, opening 67, none 14). A split with >30% `none:*` means a bug — stop and inspect before Step 6.

- [ ] **Step 6: Extend the audit page with a geometry mode**

In `scripts/audit_instruction_labels.py`:
- Add `--mode {labels,geometry}` (default `labels`) and `--landmarks PATH ...` (landmarks JSONL files).
- Load landmarks into `{id: row}`. In `geometry` mode, draw from `labelset_train.json` instead of `train_general.json`, stratified: 30 records with `pair_id` (both kinds), 70 single-route; skip records whose landmark row is missing.
- In `item()`, when a landmark row with a non-empty `polygon` exists, draw it after the route: `ImageDraw.polygon([(u*IMG_PX, v*IMG_PX) for u, v in poly], outline=(0, 230, 255), width=4)`. Add `"target": row["target"], "reason": row["reason"]` to the item.
- Put the verdict set in the payload so the page renders it: labels mode keeps the current four; geometry mode uses
  `[{"v":"ok","label":"Right target","help":"The outline is the opening the route uses / the object it goes around"}, {"v":"wrong","label":"Wrong target","help":"The outline is on something else"}, {"v":"none_ok","label":"No target, correctly","help":"Nothing nameable; dropping this record is right"}, {"v":"unsure","label":"Can't tell","help":"Too ambiguous from this frame"}]`.
- Default output for geometry mode: `outputs/instr_geometry_audit.html`.

In `scripts/audit_instruction_labels.html`: replace the four hard-coded verdict `<button>`s with an empty `<div class="verdicts" id="verdicts"></div>` and build the buttons from `DATA.verdicts` at load (keys 1–4 map to `DATA.verdicts[0..3].v`; the strip colours use `ok` → ok colour, `unsure` → unsure colour, anything else → bad colour). Show `it.instruction` if present, else `target: ${it.target}` (and the reason when `it.target === "none"`). In the legend add a cyan swatch "outlined target (geometry's choice)".

- [ ] **Step 7: Build and check the page locally**

Run:
```bash
uv run scripts/prepare_habitat_instr.py landmarks        # full run, ~40 min, 4 workers
uv run scripts/audit_instruction_labels.py --mode geometry \
    --landmarks data/prepared_habitat_instr_v2/landmarks_train.jsonl
```
Expected: `100 items ... -> outputs/instr_geometry_audit.html`. Extract 2 images from the page and look at them (as done on 2026-09-28) to confirm the cyan outline, route and dashed chord all draw.

- [ ] **Step 8: Commit**

```bash
git add scripts/prepare_habitat_instr.py scripts/audit_instruction_labels.py \
        scripts/audit_instruction_labels.html tests/test_landmarks.py
git commit -m "feat: landmarks stage and geometry audit mode

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: USER GATE — geometry audit**

Publish `outputs/instr_geometry_audit.html` as a **new** artifact (capabilities `{"db": {}}`, icon `checklist`), give the user the link, and wait. Read marks with `Artifact read_db` (collection `marks`). Pass: "Right target" + "No target, correctly" ≥ 95% of marked records. If it fails, group the failures by `target`/`reason`/`diag`, adjust the constants in `landmarks.py` (or the opening/object logic) with a test for each fix, re-run Step 7, and re-audit. Do not start Task 4's GPU steps before this passes.

---

### Task 4: Prompt i4 — target overlay, naming prompt, templates, reply resolution

**Files:**
- Modify: `src/rover_vlm/instructions.py`
- Test: `tests/test_instructions.py` (append)

**Interfaces:**
- Consumes: landmarks rows (Task 3) — `polygon`, `target`.
- Produces: `render_target_jpeg(record, polygon) -> bytes`; `naming_prompt(target, side) -> str`; `parse_target_reply(raw) -> dict | None` (keys `name`, `beside`, `single`, `instruction`); `choose_template(record_id, target, has_beside) -> str`; `fill_template(template, side, name, beside) -> str`; `head_noun(name) -> str | None`; `resolve_i4(reply, side, template) -> dict` (keys `instruction`, `instruction_source`, `rejects`, `passed`); `TARGET_TEMPLATES`; `rescore_row` handles `prompt_version == "i4"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_instructions.py`)

```python
from rover_vlm.instructions import (
    TARGET_TEMPLATES, choose_template, fill_template, head_noun, naming_prompt,
    parse_target_reply, render_target_jpeg, resolve_i4,
)


def test_naming_prompt_hides_the_route_and_hands_over_the_side():
    p = naming_prompt("object", "left")
    assert "outlined" in p and "left" in p and "magenta" not in p
    assert '"name"' in p and '"single"' in p
    assert "doorway" in naming_prompt("opening", "right").lower()


def test_parse_target_reply():
    raw = ('<think>x</think>\n<answer>{"name": "Grey Sofa", "beside": null, "single": true, '
           '"instruction": "Go left around the grey sofa."}</answer>')
    r = parse_target_reply(raw)
    assert r == {"name": "grey sofa", "beside": None, "single": True,
                 "instruction": "Go left around the grey sofa."}
    assert parse_target_reply("<answer>no json</answer>") is None
    assert parse_target_reply('<answer>{"single": "false", "name": "chair"}</answer>')["single"] is False


def test_templates_are_deterministic_and_filled():
    t = choose_template("abc_L", "object", False)
    assert t == choose_template("abc_L", "object", False) and t in TARGET_TEMPLATES["object"]
    assert "{beside}" not in choose_template("abc_L", "opening", False)
    s = fill_template("Pass the {name} on the {side}.", "right", "coffee table", None)
    assert s == "Pass the coffee table on the right."


def test_head_noun():
    assert head_noun("dark wooden cabinet") == "cabinet"
    assert head_noun("  ") is None


def test_resolve_i4_prefers_valid_paraphrase_then_template():
    tpl = "Go {side} around the {name}."
    good = {"name": "grey sofa", "beside": None, "single": True,
            "instruction": "Keep left and pass around the grey sofa."}
    r = resolve_i4(good, "left", tpl)
    assert r["passed"] and r["instruction_source"] == "cosmos"
    bad_para = {**good, "instruction": "Go around the outlined thing on the left."}
    r = resolve_i4(bad_para, "left", tpl)
    assert r["passed"] and r["instruction_source"] == "template"
    assert r["instruction"] == "Go left around the grey sofa."
    assert resolve_i4({**good, "single": False}, "left", tpl)["rejects"] == ["mask_ambiguous"]
    assert resolve_i4(None, "left", tpl)["rejects"] == ["parse"]


def test_resolve_i4_rejects_a_name_with_a_side_word():
    reply = {"name": "right cabinet", "beside": None, "single": True, "instruction": "Go left."}
    r = resolve_i4(reply, "left", "Go {side} around the {name}.")
    assert not r["passed"] and "side:opposite" in r["rejects"]


def test_render_target_jpeg(tmp_path):
    from PIL import Image
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(img)
    rec = {"image": [str(img)]}
    out = render_target_jpeg(rec, [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    im = Image.open(io.BytesIO(out)).convert("RGB")
    assert im.getpixel((5, 5))[0] < im.getpixel((im.width // 2, im.height // 2))[0]   # outside dimmed


def test_rescore_row_i4():
    row = {"id": "a_L", "side": "left", "prompt_version": "i4", "template": "Go {side} around the {name}.",
           "reply": {"name": "grey sofa", "beside": None, "single": True,
                     "instruction": "Go left around the grey sofa."}, "passed": False}
    new = rescore_row(row)
    assert new["passed"] and new["instruction_source"] == "cosmos"
    none_row = {"id": "b_L", "side": "left", "prompt_version": "i4", "reply": None,
                "rejects": ["no_target:no_visible_target"], "passed": False}
    assert rescore_row(none_row) == none_row
```

Add `import io` at the top of `tests/test_instructions.py` if it is not already imported.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_instructions.py -q`
Expected: FAIL — `ImportError: cannot import name 'TARGET_TEMPLATES'`

- [ ] **Step 3: Implement** in `src/rover_vlm/instructions.py`

(a) Extend `_DRAWING` so an i4 instruction cannot lean on the outline — add `outlin\w*|contour|dimmed|cyan` inside the alternation:

```python
_DRAWING = re.compile(
    r"\b(?:lines?|stroke|overlay|drawn|drawing|highlight\w*|marked|arrows?|badges?|magenta|"
    r"numbered|coordinates?|options?|outlin\w*|contour|dimmed|cyan|(?:route|path) (?:1|2|one|two)|"
    r"(?:bold|thin|grey|gray) (?:route|path))\b",
    re.I,
)
```

(b) Append the i4 block after the `rescore_row` definition:

```python
# --- i4 (2026-09-28): geometry picks the target, Cosmos only names it ------------------
# The v2 human audit found i3's landmarks ~20% wrong (75% on pairs' second route): Cosmos
# named real objects but not the one the route goes around. i4 shows Cosmos only the target
# that rover_vlm.landmarks chose — outlined, rest dimmed, no route — and hands it the side
# and a sentence pattern. The name is Cosmos's; the side and the referent are geometry's.

TARGET_COLOR = (0, 230, 255)
TARGET_TEMPLATES = {
    "object": [
        "Go {side} around the {name}.",
        "Pass the {name} on the {side}.",
        "Take the {side} way around the {name}.",
        "Keep {side} of the {name}.",
        "Go around the {name} on the {side} side.",
        "Steer {side} around the {name}.",
        "Pass {side} of the {name}.",
        "Head {side} around the {name}.",
    ],
    "opening": [
        "Go through the {name} on the {side}.",
        "Take the {name} on the {side}.",
        "Head {side} through the {name}.",
        "Use the {name} on the {side}.",
        "Go {side} through the {name}.",
        "Go through the {name} to the {side} of the {beside}.",
        "Take the {name} {side} of the {beside}.",
        "Head for the {name} on the {side}, by the {beside}.",
    ],
}


def _stable_index(key, n):
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n


def choose_template(record_id, target, has_beside):
    """A sentence pattern for the record, fixed by its id (reruns agree)."""
    opts = [t for t in TARGET_TEMPLATES[target] if has_beside or "{beside}" not in t]
    return opts[_stable_index(record_id, len(opts))]


def fill_template(template, side, name, beside):
    return template.format(side=side, name=name, beside=beside or "")


def head_noun(name):
    words = re.findall(r"[a-z]+", (name or "").lower())
    return words[-1] if words else None


def render_target_jpeg(record, polygon):
    """The frame with the target outlined and everything else dimmed; no route drawn."""
    img, _ = _canvas(record)
    W, H = img.size
    pts = [(u * W, v * H) for u, v in polygon]
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    dim = Image.eval(img, lambda p: int(p * 0.45))
    img = Image.composite(img, dim, mask)
    ImageDraw.Draw(img).polygon(pts, outline=TARGET_COLOR, width=6)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def naming_prompt(target, side):
    """Ask Cosmos to name the outlined target and phrase the instruction (prompt i4)."""
    what = ("an opening the rover drives through — a doorway, a passage, a hallway or a gap "
            "between two things" if target == "opening" else "the object the rover drives around")
    pattern = TARGET_TEMPLATES[target][0].format(side=side, name="<name>", beside="<beside>")
    beside = ('"beside": "<the most prominent object right next to the opening, or null>", '
              if target == "opening" else '"beside": null, ')
    return f"""{SCENE_INTRO}

One region of the image is outlined; everything outside it is dimmed. The outlined region is
{what}. The rover will pass it on its {side}.

1. Name what is inside the outline in 1 to 5 words, the way a person would say it
   ("grey sofa", "open doorway", "kitchen island"). Never use the words left or right in
   the name.
2. Say whether the outline covers one single thing (true) or several unrelated things (false).
3. Write one instruction a person standing behind the rover would say, in your own words,
   meaning the same as: "{pattern}". It must contain the word '{side}' and the name, and must
   not mention the outline, the dimming, colours of the drawing, or "your left/right".

{_answer_block('{"name": "<name>", ' + beside + '"single": true, "instruction": "<the instruction>"}')}"""


def parse_target_reply(raw):
    """{"name", "beside", "single", "instruction"} from a Cosmos i4 reply, or None."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is None or not isinstance(payload.get("name"), str) or not payload["name"].strip():
        return None
    single = payload.get("single", True)
    if isinstance(single, str):
        single = single.strip().lower() not in ("false", "no", "0")
    beside = payload.get("beside")
    beside = " ".join(beside.lower().split()) if isinstance(beside, str) and beside.strip() else None
    text = payload.get("instruction")
    return {"name": " ".join(payload["name"].lower().split()), "beside": beside,
            "single": bool(single),
            "instruction": " ".join(text.split()) if isinstance(text, str) and text.strip() else None}


def _i4_rejects(text, side, name):
    out = instruction_rejects(text, side=side)
    noun = head_noun(name)
    if text and noun and not re.search(rf"\b{re.escape(noun)}(?:e?s)?\b", text, re.I):
        out.append("name:missing")
    return out


def resolve_i4(reply, side, template):
    """The instruction for a record from its i4 reply: Cosmos's paraphrase when it passes the
    rules, else the filled template (which must pass them too)."""
    if reply is None:
        return {"instruction": None, "instruction_source": None, "rejects": ["parse"], "passed": False}
    if not reply["single"]:
        return {"instruction": None, "instruction_source": None, "rejects": ["mask_ambiguous"], "passed": False}
    if len(reply["name"].split()) > 5 or _DRAWING.search(reply["name"]):
        return {"instruction": None, "instruction_source": None, "rejects": ["name:bad"], "passed": False}
    para = reply.get("instruction")
    if para and not _i4_rejects(para, side, reply["name"]):
        return {"instruction": para, "instruction_source": "cosmos", "rejects": [], "passed": True}
    filled = fill_template(template, side, reply["name"], reply.get("beside"))
    rejects = _i4_rejects(filled, side, reply["name"])
    return {"instruction": filled if not rejects else None, "instruction_source": "template",
            "rejects": rejects, "passed": not rejects}
```

(c) Make `rescore_row` handle i4 — add as the first check after the `error` test:

```python
    if row.get("prompt_version") == "i4":
        if row.get("reply") is None and any(r.startswith("no_target") for r in row.get("rejects", [])):
            return row
        new = dict(row)
        new.update(resolve_i4(row.get("reply"), row["side"], row["template"]))
        return new
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_instructions.py -q`
Expected: all pass. The `_DRAWING` change must not break older tests; if one does, it is because an old fixture sentence uses one of the new words — change the fixture sentence, not the rule.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: prompt i4 - name the outlined target, paraphrase or template

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Label client i4 path + sbatch knobs

**Files:**
- Modify: `scripts/label_instructions.py`
- Modify: `slurm/label_instructions.sbatch`
- Test: `tests/test_instructions.py` (append; the client's pure helper lives in `instructions.py`)

**Interfaces:**
- Consumes: Task 4 functions; landmarks JSONL (Task 3).
- Produces: label rows `{"id", "pair_id", "kind", "side", "prompt_version": "i4", "target", "template", "reply", "instruction", "instruction_source", "rejects", "passed", "finish", "latency_s"}`; for `target == "none"`: `{"reply": None, "rejects": ["no_target:<reason>"], "passed": False}` with no Cosmos call. `load_landmarks(paths, ids) -> dict` in `instructions.py`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_instructions.py`)

```python
from rover_vlm.instructions import load_landmarks


def test_load_landmarks_refuses_incomplete_coverage(tmp_path):
    p = tmp_path / "lm.jsonl"
    p.write_text(json.dumps({"id": "a", "target": "object", "polygon": [[0, 0], [1, 0], [1, 1]]}) + "\n")
    assert set(load_landmarks([p], {"a"})) == {"a"}
    with pytest.raises(SystemExit, match="1 of 2 records have no landmark row"):
        load_landmarks([p], {"a", "b"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_instructions.py::test_load_landmarks_refuses_incomplete_coverage -q`
Expected: FAIL — `ImportError: cannot import name 'load_landmarks'`

- [ ] **Step 3: Implement `load_landmarks`** (append to the i4 block in `instructions.py`)

```python
def load_landmarks(paths, ids):
    """{id: landmarks row} for `ids`; exits when any id has no row — labelling a record
    without its target would silently fall back to nothing."""
    rows = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(ids)} records have no landmark row "
                         f"(e.g. {missing[:3]}) — run prepare_habitat_instr.py landmarks")
    return {i: rows[i] for i in ids}
```

- [ ] **Step 4: Wire the i4 path into `scripts/label_instructions.py`**

Add CLI flags:

```python
    p.add_argument("--prompt", choices=["i3", "i4"], default="i4",
                   help="i4 (2026-09-28): name the geometry-chosen target; i3: the v2 prompt")
    p.add_argument("--landmarks", type=Path, nargs="+", default=None,
                   help="landmarks_*.jsonl from prepare_habitat_instr.py landmarks (i4)")
```

After `records` are loaded (and limited), add:

```python
    landmarks = None
    if args.prompt == "i4":
        if not args.landmarks:
            raise SystemExit("--prompt i4 needs --landmarks data/prepared_habitat_instr_v2/landmarks_*.jsonl")
        landmarks = load_landmarks(args.landmarks, {r["id"] for r in records})
```

Replace the start of `label(rec)` so i4 records branch before the i3 code (keep the i3 body unchanged below it):

```python
        def label(rec):
            """One record. Never raises: a bad sample is a recorded failure."""
            hm = rec["habitat_meta"]
            row = {"id": rec["id"], "pair_id": hm.get("pair_id"), "kind": hm.get("kind"),
                   "side": hm["side"], "prompt_version": args.prompt}
            if args.retry_rejected:
                row["retried"] = True
            if args.prompt == "i4":
                lm = landmarks[rec["id"]]
                row["target"] = lm["target"]
                if lm["target"] == "none":
                    return {**row, "reply": None, "rejects": [f"no_target:{lm['reason']}"], "passed": False}
                template = choose_template(rec["id"], lm["target"], has_beside=True)
                t0 = time.time()
                try:
                    raw, fin = chat_with_image(client, model, render_target_jpeg(rec, lm["polygon"]),
                                               naming_prompt(lm["target"], hm["side"]),
                                               max_tokens=args.max_tokens, seed=seed)
                except Exception as e:  # noqa: BLE001
                    return {**row, "error": f"{type(e).__name__}: {e}"}
                reply = parse_target_reply(raw)
                # a template needing {beside} is only usable when Cosmos gave one
                if "{beside}" in template and not (reply and reply.get("beside")):
                    template = choose_template(rec["id"], lm["target"], has_beside=False)
                row.update({"template": template, "reply": reply,
                            **resolve_i4(reply, hm["side"], template),
                            "finish": fin, "latency_s": round(time.time() - t0, 2)})
                if args.show or not row["passed"]:
                    row["raw_gen"] = raw
                return row
            # --- i3 (unchanged) ---
```

Import the new names at the top: `choose_template, load_landmarks, naming_prompt, parse_target_reply, render_target_jpeg, resolve_i4`. Replace `INSTRUCTION_PROMPT_VERSION` in the start-up print with `args.prompt`. In the progress print, `row.get('instruction')` already works for i4 rows. Update the module docstring's usage block with:

```
    # i4 (2026-09-28): geometry-chosen targets, Cosmos names them
    uv run scripts/label_instructions.py --prompt i4 \
        --split data/prepared_habitat_instr_v2/labelset_eval.json \
        --landmarks data/prepared_habitat_instr_v2/landmarks_eval.jsonl \
        --out outputs/instr_labels_v2/eval_i4.jsonl --concurrency 8 --resume
```

- [ ] **Step 5: Add `PROMPT` / `STAGE` knobs to `slurm/label_instructions.sbatch`**

After the existing variable block add:

```bash
PROMPT="${PROMPT:-i4}"
# STAGE=gate: only the gate set, then stop (a human audits it before the full run)
# STAGE=full: skip the gate, label eval + train
STAGE="${STAGE:-gate}"
LM_ARGS=()
if [ "$PROMPT" = "i4" ]; then
    for s in eval train; do
        [ -f "$D/landmarks_$s.jsonl" ] || { echo "missing $D/landmarks_$s.jsonl — run prepare_habitat_instr.py landmarks"; exit 1; }
    done
fi
```

Replace the gate block's record selection and call with:

```bash
if [ "$STAGE" = "gate" ]; then
    echo; echo "=============== gate ($PROMPT): 100 eval pairs + 100 single-route eval records ==============="
    "$UV" run python - "$D/labelset_eval.json" "$OUT/gate_split_$PROMPT.json" <<'PY'
import json, sys
recs = json.load(open(sys.argv[1]))
pairs = list(dict.fromkeys(r["habitat_meta"]["pair_id"] for r in recs if r["habitat_meta"].get("pair_id")))[:100]
keep = [r for r in recs if r["habitat_meta"].get("pair_id") in set(pairs)]
keep += [r for r in recs if not r["habitat_meta"].get("pair_id")][:100]
json.dump(keep, open(sys.argv[2], "w"))
print(len(keep), "gate records")
PY
    "$UV" run scripts/label_instructions.py --prompt "$PROMPT" --split "$OUT/gate_split_$PROMPT.json" \
        --landmarks "$D/landmarks_eval.jsonl" --out "$OUT/gate_$PROMPT.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume
    echo "gate labelled — audit it before STAGE=full"; exit 0
fi
```

and the full loop's two calls gain `--prompt "$PROMPT" --landmarks "$D/landmarks_$s.jsonl"` with outputs `"$OUT/${s}_${PROMPT}.jsonl"`. Update the header comment: submit the gate with `sbatch slurm/label_instructions.sbatch`, the full run with `STAGE=full sbatch slurm/label_instructions.sbatch`, and afterwards `finalize-general --instructions outputs/instr_labels_v2/train_i4.jsonl outputs/instr_labels_v2/eval_i4.jsonl`. (Keep `PROMPT=i3` working: the landmark check and `--landmarks` only apply when `PROMPT=i4`; for i3 pass no `--landmarks`.)

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (including `test_slurm_thread_caps.py`).

- [ ] **Step 7: Commit**

```bash
git add scripts/label_instructions.py slurm/label_instructions.sbatch src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: i4 labelling path in the client and the unattended job

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: finalize-general keeps the instruction source

**Files:**
- Modify: `src/rover_vlm/instructions.py` (`finalize_general`)
- Test: `tests/test_instructions.py` (append)

**Interfaces:**
- Consumes: i4 rows (Task 5).
- Produces: kept records carry `habitat_meta.instruction_source` = the row's `instruction_source` (`cosmos`/`template`), defaulting to `"cosmos"` for i3 rows.

- [ ] **Step 1: Write the failing test** (append; `_src`, `build_frame` and `finalize_general` are already imported/defined in `tests/test_instructions.py` — `test_finalize_general` uses them)

```python
def test_finalize_general_keeps_instruction_source(two_sided_sample):
    f = build_frame(_src(two_sided_sample))
    labels = f["labelset"]
    rows = {r["id"]: {"id": r["id"], "passed": True,
                      "instruction": f"Go {r['habitat_meta']['side']} around the sofa.",
                      "instruction_source": "template", "prompt_version": "i4"} for r in labels}
    out = finalize_general([f["plain"]], labels, rows)
    kept = [r for r in out["train_or_all"] if r["habitat_meta"].get("instruction")]
    assert len(kept) == 2
    assert all(r["habitat_meta"]["instruction_source"] == "template" for r in kept)
    rows = {i: {k: v for k, v in r.items() if k != "instruction_source"} for i, r in rows.items()}
    out = finalize_general([f["plain"]], labels, rows)                 # i3 rows: no source field
    assert all(r["habitat_meta"]["instruction_source"] == "cosmos" for r in out["pairs"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_instructions.py::test_finalize_general_keeps_instruction_source -q`
Expected: FAIL — `assert ... == "template"` (currently hard-coded `"cosmos"`).

- [ ] **Step 3: Implement** — in `finalize_general`, change the `kept = [...]` line to

```python
    kept = [with_instruction(r, rows_by_id[r["id"]]["instruction"],
                             rows_by_id[r["id"]].get("instruction_source") or "cosmos")
            for r in label_recs
            if r["id"] in passed and r["habitat_meta"].get("pair_id") not in bad_pairs]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/instructions.py tests/test_instructions.py
git commit -m "feat: finalize keeps whether the instruction is Cosmos's or the template

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Semantic replay script (Habitat env) + sbatch

**Files:**
- Create: `scripts/render_semantic_frames.py`
- Create: `slurm/render_semantic_frames.sbatch`
- Modify: `tests/test_slurm_thread_caps.py` (add the new sbatch to the list)

**Interfaces:**
- Consumes: `labelset_{train,eval}.json`; `spaceitup-habitat/src/roverbench/sim.py` (`make_simulator`, `set_agent_pose`).
- Produces: `outputs/landmark_semantics/<sample_dir_name>.npz` with `ids` (uint16 or uint32, H×W instance ids at 512²) and `categories` (JSON string `{instance_id: category}`).

This runs only on the GPU in the `habitat` conda env, so it has no CPU unit test (like training). Keep it small and loud.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Render one HM3D SEMANTIC frame per labelset sample, for the grounded-landmark name check.

For every sample behind data/prepared_habitat_instr_v2/labelset_{train,eval}.json whose
scene ships HM3D semantics, replay the SEMANTIC sensor at the sample's stored camera pose
(same world frame, 512x512, hfov 90, camera 0.8 m) and save
outputs/landmark_semantics/<sample>.npz: `ids` (instance id per pixel) + `categories`
(JSON {id: category}). Resumes: existing files are skipped.

Runs in the `habitat` conda env (NOT uv) on the full A100 (MIG slices cannot render):
    sbatch slurm/render_semantic_frames.sbatch
Imports the generator's roverbench.sim from ../spaceitup-habitat/src; writes nothing there.
"""
import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HABITAT_SRC = REPO_ROOT.parent / "spaceitup-habitat" / "src"
HM3D = Path("/nfs/projects/spaceitup/rover_navigation/data/hm3d")
sys.path.insert(0, str(HABITAT_SRC))
from roverbench.sim import make_simulator, set_agent_pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v2")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "landmark_semantics")
    ap.add_argument("--config", default=str(HM3D / "hm3d_annotated_basis.scene_dataset_config.json"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_scene = defaultdict(set)
    for split in ("train", "eval"):
        for r in json.loads((args.data / f"labelset_{split}.json").read_text()):
            by_scene[r["habitat_meta"]["scene_id"]].add(str(Path(r["image"][0]).parent))
    semantic = {p.parent.name for p in HM3D.glob("*/*/*.semantic.glb")}
    scenes = sorted(s for s in by_scene if s in semantic)
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    print(f"{len(by_scene)} scenes in the labelsets, {len(scenes)} with HM3D semantics", flush=True)

    done = fail = 0
    t0 = time.time()
    for n, scene in enumerate(scenes, 1):
        todo = [Path(d) for d in sorted(by_scene[scene]) if not (args.out / f"{Path(d).name}.npz").exists()]
        if not todo:
            continue
        try:
            sim = make_simulator(scene.split("-", 1)[1], None, width=args.res, height=args.res,
                                 camera_height_m=0.8, hfov_deg=90.0, with_depth=False,
                                 with_semantic=True, scene_dataset_config=args.config)
        except Exception:
            fail += len(todo)
            print(f"[{scene}] SIM FAILED\n{traceback.format_exc()}", flush=True)
            continue
        try:
            cats = json.dumps({int(o.semantic_id): o.category.name()
                               for o in sim.semantic_scene.objects if o is not None})
            for d in todo:
                meta = json.loads((d / "meta.json").read_text())
                fh = meta["chord"]["floor_height"]
                set_agent_pose(sim, [meta["pose_xz"][0], fh, meta["pose_xz"][1]],
                               [meta["heading_xz"][0], 0.0, meta["heading_xz"][1]])
                ids = np.asarray(sim.get_sensor_observations()["semantic"])
                ids = ids.astype(np.uint16 if ids.max() < 65536 else np.uint32)
                np.savez_compressed(args.out / f"{d.name}.npz", ids=ids, categories=cats)
                done += 1
        finally:
            sim.close()
        print(f"[{n}/{len(scenes)}] {scene}: {len(todo)} frames  total {done}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {done} frames written, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the sbatch**

```bash
#!/bin/bash
# HM3D semantic frames for the grounded-landmark name check (scripts/render_semantic_frames.py).
#
#     sbatch slurm/render_semantic_frames.sbatch
#
# Full A100 only: habitat's EGL renderer cannot use a MIG slice. ~6.3k frames in ~150
# scenes; resumable (existing .npz are skipped), so a timeout is fixed by re-submitting.
#SBATCH --job-name=semantic-frames
#SBATCH --partition=A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=telegram:422627024
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/landmark_semantics
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/miniforge3/etc/profile.d/conda.sh
conda activate habitat
python -u scripts/render_semantic_frames.py "$@"
```

- [ ] **Step 3: Add the sbatch to the thread-cap test**

In `tests/test_slurm_thread_caps.py`, add `"render_semantic_frames.sbatch"` to the parametrize list.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_slurm_thread_caps.py -q`
Expected: all pass.

- [ ] **Step 5: Check the habitat env can import the script's dependencies (login node, no GPU)**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate habitat && python -c "import sys; sys.path.insert(0, '../spaceitup-habitat/src'); import roverbench.sim; print('ok')"`
Expected: `ok`. (If `habitat_sim` fails to import on the login node for lack of a GPU, that is fine — note it and rely on the GPU run; the import error message must not be about a missing module.)

- [ ] **Step 6: Commit**

```bash
git add scripts/render_semantic_frames.py slurm/render_semantic_frames.sbatch tests/test_slurm_thread_caps.py
git commit -m "feat: HM3D semantic replay for the landmark name check

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: USER STEP** — ask the user to `sbatch slurm/render_semantic_frames.sbatch` (it can run while the geometry audit is open) and paste back the last lines of `outputs/slurm/semantic-frames-<id>.out`. Expected final line: `done: ~6000 frames written, 0 failed`.

---

### Task 8: HM3D name agreement check

**Files:**
- Modify: `scripts/audit_instruction_labels.py` (new `semantic` sub-mode)
- Create: `src/rover_vlm/landmark_check.py` (pure matching logic, testable)
- Test: `tests/test_landmarks.py` (append)

**Interfaces:**
- Consumes: i4 label rows (Task 5), landmarks rows (Task 3), semantic npz (Task 7).
- Produces: `canonical(word) -> str`; `name_matches(name, category) -> bool`; `target_category(ids, categories, polygon) -> str | None`; `beside_categories(ids, categories, bbox) -> set[str]`; `OPENING_WORDS`; report `outputs/landmark_name_check.json` + `outputs/landmark_name_mismatches.jsonl`.

- [ ] **Step 1: Write the failing tests**

```python
from rover_vlm.landmark_check import (
    OPENING_WORDS, beside_categories, canonical, name_matches, target_category,
)


def test_name_matching():
    assert canonical("couch") == canonical("sofa")
    assert name_matches("grey sofa", "couch")
    assert name_matches("kitchen island", "kitchen island")
    assert name_matches("wooden cabinet", "cabinet")
    assert not name_matches("armchair", "table")
    assert {"doorway", "hallway"} <= OPENING_WORDS


def test_target_category_majority_inside_polygon():
    ids = np.zeros((10, 10), np.uint16)
    ids[2:8, 2:8] = 5
    ids[0:10, 0:2] = 9
    cats = json.dumps({"5": "couch", "9": "wall", "0": "floor"})
    assert target_category(ids, cats, [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]) == "couch"
    assert "wall" in beside_categories(ids, cats, [0.2, 0.2, 0.8, 0.8])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_landmarks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rover_vlm.landmark_check'`

- [ ] **Step 3: Implement `src/rover_vlm/landmark_check.py`**

```python
"""Does an i4 landmark name agree with the HM3D category under its target? (CPU)

Used by scripts/audit_instruction_labels.py semantic. HM3D categories are raw annotator
strings ("couch", "kitchen cabinet", "unknown"), so matching goes through a small synonym
table and head-noun containment rather than exact equality.
"""
import json

import numpy as np
from PIL import Image, ImageDraw

IGNORED = {"floor", "ceiling", "unknown", "carpet", "rug", "floor mat", "mat"}
OPENING_WORDS = {"door", "doorway", "opening", "hallway", "corridor", "passage", "entrance",
                 "entry", "archway", "arch", "gap", "hall", "passageway"}
_SYNONYMS = [
    {"sofa", "couch", "sectional", "loveseat", "settee"},
    {"armchair", "chair", "seat"},
    {"table", "desk"},
    {"cabinet", "cupboard", "dresser", "sideboard", "drawers", "wardrobe", "closet"},
    {"counter", "countertop", "island"},
    {"shelf", "shelves", "bookshelf", "bookcase", "shelving"},
    {"column", "pillar", "post", "pole"},
    {"wall", "partition"},
    {"railing", "banister", "balustrade", "rail"},
    {"plant", "planter", "pot", "flowerpot"},
    {"bed", "mattress"},
    {"fridge", "refrigerator"},
    {"tv", "television", "monitor", "screen"},
]
_CANON = {w: sorted(g)[0] for g in _SYNONYMS for w in g}


def canonical(word):
    w = word.lower().strip()
    w = w[:-1] if w.endswith("s") and w[:-1] in _CANON else w
    return _CANON.get(w, w)


def name_matches(name, category):
    nw = {canonical(w) for w in name.lower().split()}
    cw = {canonical(w) for w in category.lower().split()}
    return bool(nw & cw)


def _mask(shape, polygon):
    H, W = shape
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(u * W, v * H) for u, v in polygon], fill=1)
    return np.asarray(im) > 0


def target_category(ids, categories, polygon):
    """Most frequent non-floor category inside the target outline, or None."""
    cats = json.loads(categories) if isinstance(categories, str) else categories
    inside = ids[_mask(ids.shape, polygon)]
    counts = {}
    for i, n in zip(*np.unique(inside, return_counts=True)):
        c = cats.get(str(int(i)), "unknown")
        if c not in IGNORED:
            counts[c] = counts.get(c, 0) + int(n)
    return max(counts, key=counts.get) if counts else None


def beside_categories(ids, categories, bbox, grow=0.5, min_frac=0.02):
    """Categories covering >= min_frac of the bbox grown by `grow` of its size each side."""
    cats = json.loads(categories) if isinstance(categories, str) else categories
    H, W = ids.shape
    u0, v0, u1, v1 = bbox
    du, dv = (u1 - u0) * grow, (v1 - v0) * grow
    r0, r1 = int(max(0, v0 - dv) * H), int(min(1, v1 + dv) * H) + 1
    c0, c1 = int(max(0, u0 - du) * W), int(min(1, u1 + du) * W) + 1
    region = ids[r0:r1, c0:c1]
    out = set()
    for i, n in zip(*np.unique(region, return_counts=True)):
        c = cats.get(str(int(i)), "unknown")
        if c not in IGNORED and n >= min_frac * region.size:
            out.add(c)
    return out
```

- [ ] **Step 4: Add the `semantic` mode to `scripts/audit_instruction_labels.py`**

`--mode semantic --labels outputs/instr_labels_v2/gate_i4.jsonl --landmarks data/prepared_habitat_instr_v2/landmarks_eval.jsonl [--semantics outputs/landmark_semantics]`. For each passed i4 row with a semantic npz for its `Path(landmark["sample_dir"]).name`:
- object target: `cat = target_category(...)`; `ok = cat is not None and name_matches(reply["name"], cat)`.
- opening target: `ok_name = any(w in OPENING_WORDS for w in reply["name"].split())`; if `reply["beside"]`: `ok_beside = any(name_matches(reply["beside"], c) for c in beside_categories(ids, cats, lm["bbox"]))`, else `ok_beside = None`.
Print a table (target type × pair/single: n, agreement %) and write `outputs/landmark_name_check.json` (the table) plus `outputs/landmark_name_mismatches.jsonl` (id, target, name, beside, hm3d category / nearby categories, instruction). Print how many rows had no semantic frame (not an error: most scenes have no HM3D semantics).

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/rover_vlm/landmark_check.py scripts/audit_instruction_labels.py tests/test_landmarks.py
git commit -m "feat: HM3D name agreement check for i4 landmarks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Gate run, label audit, full run, docs

**Files:**
- Modify: `README.md` (v2 instruction section), `NOTES.md`

- [ ] **Step 1: USER STEP — i4 gate.** Ask the user to run `sbatch slurm/label_instructions.sbatch` (defaults `PROMPT=i4 STAGE=gate`) and paste the tail of `outputs/slurm/cosmos3-label-instr-<id>.out`. Expected: ~300 rows in `outputs/instr_labels_v2/gate_i4.jsonl`, summary printed.

- [ ] **Step 2: Automatic checks on the gate**

Run:
```bash
uv run python -c "
import json, collections
rows=[json.loads(l) for l in open('outputs/instr_labels_v2/gate_i4.jsonl')]
print(collections.Counter(r['rejects'][0] if r['rejects'] else 'passed' for r in rows if not r.get('error')))
print(collections.Counter(r.get('instruction_source') for r in rows if r.get('passed')))"
uv run scripts/audit_instruction_labels.py --mode semantic --labels outputs/instr_labels_v2/gate_i4.jsonl \
    --landmarks data/prepared_habitat_instr_v2/landmarks_eval.jsonl
```
Report both to the user, with 5 random mismatches from `outputs/landmark_name_mismatches.jsonl`.

- [ ] **Step 3: USER GATE — label audit.** Build the labels-mode audit page from the gate rows (`--mode labels` reading `gate_i4.jsonl` + `labelset_eval.json` + landmarks, target outline drawn), publish it as a new artifact, and wait for the marks. Pass: < 5% wrong overall **and** < 5% wrong on pair records. If it fails, bring the failure breakdown to the user before changing anything (the next lever named in the spec is Gemma-3-12B on the same gate set).

- [ ] **Step 4: USER STEP — full run.** `STAGE=full sbatch slurm/label_instructions.sbatch` (~8 h, Telegram-notified), then on the login node:
```bash
uv run scripts/prepare_habitat_instr.py finalize-general --instructions \
    outputs/instr_labels_v2/train_i4.jsonl outputs/instr_labels_v2/eval_i4.jsonl
```
Expected: refuses only if > 1% of records lack a row; otherwise prints per-split stats (instructed/pairs kept) — report them vs the i3 numbers (train_general 48,455; 1,963 pairs).

- [ ] **Step 5: Docs**

README v2-instruction section: add the grounded-landmark pipeline in run order (landmarks → geometry audit → semantic frames → i4 gate → name check + label audit → full run → finalize), one command each, and where outputs land (`data/prepared_habitat_instr_v2/landmarks_*.jsonl`, `outputs/landmark_semantics/`, `outputs/instr_labels_v2/*_i4.jsonl`, `outputs/landmark_name_check.json`). NOTES.md: current state and next steps (≤ 40 lines).

- [ ] **Step 6: Commit**

```bash
git add README.md NOTES.md
git commit -m "docs: grounded-landmark pipeline

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
