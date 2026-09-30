# Route Descriptions (prompt d1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-landmark i4 instructions with geometry-grounded route descriptions (≤3 events + a stop phrase), labelled by Cosmos in two stages (name each object on a crop, then compose the sentence text-only), and build train/eval sets where the model gets the entry point and the description but not the goal.

**Architecture:** A pure-numpy geometry module (`route_events.py`) finds, per route, every salient obstacle it passes, merges them into pass / between / through events, picks the top 3, adds a stop phrase, and trims each object's outline with a per-frame depth calibration. A second module (`descriptions.py`) holds templates, text rules, the two Cosmos prompts and the finalize logic. A new labelling script drives both Cosmos stages; `prepare_habitat_instr.py` gets `events` and `finalize-descriptions` stages; `eval.py` gets a "description followed" metric that maps predicted image paths back onto the frame's floor.

**Tech Stack:** Python 3.12 (uv), numpy, Pillow, pytest; Cosmos3-Nano over an OpenAI-compatible endpoint (`openai` client) on SLURM; ffmpeg for videos.

**Spec:** `docs/superpowers/specs/2026-09-30-route-descriptions-design.md` (read it first; this plan argues from it).

## Global Constraints

- Shared 96-core login node: every CPU command that runs numpy/torch exports `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` (multiprocessing stages: `=1` per worker, 4 workers).
- Never load Qwen or Cosmos on the login node. GPU work is an sbatch file the user submits; at most 3 SLURM jobs at once.
- Test-first for every code change (`uv run pytest tests/<file> -v`); the full suite (`uv run pytest -q`) must stay green.
- The user's uncommitted files (`README.md` framing row, `scripts/evaluate.py`, untracked `scripts/*`, `slurm/*`, `NOTES.md`) are NOT ours: never `git add -A` / `git commit -a`; add files by name. Before editing `README.md` or `scripts/evaluate.py`, check `git diff` on them and stop to ask the user if unrelated hunks are still uncommitted.
- Side convention (unchanged since i2): "go left of X" / "pass X on the left" = the route passes X on the route's left, i.e. X is on the route's right.
- Human audit pages are the gates: geometry audit ≥ 90% "true" before any GPU run; label audit ≥ 90% "correct" before the full labelling run.
- Every script/module starts with a docstring saying what it does and how to run it; outputs go under `outputs/` or `data/prepared_*` (gitignored), never the repo root.
- Commit messages end with: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **A frame with no `depth.png`, or where the depth fit is poor** (few free-floor pixels, SDXL-enhanced frames) → the column outline is used and the object flagged `depth_fallback`; never a crash. Pinned in Task 2.
2. **Two events whose objects share a head noun** ("go left of the table, then right of the table") → the text-rule order search must match the second name after the first, not the same word twice. Pinned in Task 4.
3. **A pair where one route has no stop anchor or fails labelling** → the other route stays in training, the pair leaves `eval_desc_pairs.json`; nothing silently shrinks. Pinned in Task 9.
4. **A predicted path with points above the horizon, fewer than 2 floor points, or an unparseable answer** → counts as not followed, stays in the denominator, never raises. Pinned in Task 10.
5. **Cosmos replying with an odd kind ("Opening", "doorway", missing) or extra keys** → the kind is normalised; unknown kinds become "object"; a missing name is a rejection, not a crash. Pinned in Task 6.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/rover_vlm/route_events.py` | create | Geometry: frame loading, fixed visibility, candidates, depth calibration + trimmed outlines, events, stop phrase, `describe_route`, `collect_objects` |
| `src/rover_vlm/descriptions.py` | create | Templates, text rules, name rules, naming image/prompt/parser, compose prompt/parser, `label_object` / `label_route`, finalize helpers |
| `src/rover_vlm/labelling_io.py` | modify | add `chat_text` |
| `src/rover_vlm/habitat_data.py` | modify | add `HABITAT_PROMPT_DESCRIPTION`, `description_entry_prompt` |
| `src/rover_vlm/eval.py` | modify | add `image_to_floor`, `description_follow`, `aggregate_description_metrics` |
| `scripts/prepare_habitat_instr.py` | modify | stages `events`, `finalize-descriptions` |
| `scripts/label_descriptions.py` | create | Cosmos stages `name` / `compose`, resumable, gate |
| `scripts/audit_instruction_labels.py` | modify | modes `events` (geometry gate) and `d1` (label gate) |
| `scripts/evaluate.py` | modify | attach `summary["description"]` |
| `scripts/render_desc_videos.py` | create | mp4 of predicted vs GT paths with the description |
| `slurm/label_descriptions.sbatch` | create | serve Cosmos + both stages, `STAGE=gate|full`, resumable |
| `slurm/eval_descriptions.sbatch` | create | eval cells on the four `eval_desc_*` sets |
| `tests/test_route_events.py` | create | geometry tests (synthetic grids + 3 real-frame regressions) |
| `tests/test_descriptions.py` | create | text rules, templates, parsers, labelling logic with fake clients, finalize |
| `tests/test_description_eval.py` | create | back-projection and follow metric |
| `tests/test_render_desc_videos.py` | create | panel and frame picking |
| `README.md`, `NOTES.md` | modify | how to run the d1 pipeline |

---

### Task 1: Frame loading, fixed visibility and candidate obstacles

**Files:**
- Create: `src/rover_vlm/route_events.py`
- Test: `tests/test_route_events.py`

**Interfaces:**
- Consumes (from `rover_vlm.landmarks`): `Grid`, `Camera`, `components`, `convex_hull`, `densify`, `column_in_frame`, `_dilate`, `_poly_area`, constants `OCC`, `FREE`, `OBJECT_TOP_M`, `CAM_HEIGHT_M`.
- Produces:
  - `Frame` dataclass: `name: str, grid: Grid, cam: Camera, lab: np.ndarray, depth: np.ndarray | None, depth_fit: tuple | None`
  - `load_frame(sample_dir) -> (meta: dict, frame: Frame)`
  - `first_blocker(frame, xz) -> int | None` (blob label of the first occupied cell on the ray, ignoring cells within `NEAR_CAMERA_M` of the camera)
  - `visible_cells(frame, cells_rc, label) -> np.ndarray (k, 2) world xz`
  - `route_geometry(route_xz) -> {"pts", "arc", "right", "offset"}`
  - `column_outline(frame, vis_xz, top=OBJECT_TOP_M) -> outline dict` and `outline_dict(poly, depth_fallback) -> {"polygon", "bbox", "crop_box", "area", "depth_fallback"}`
  - `candidates(frame, route_xz, outline=None) -> list[dict]` sorted by `arc_m`; each: `blob, pass_m, arc_m, side ("left"|"right" = side the route passes it on), offset_m, extent_m, gap (None | (p_xz, q_xz)), vis_xz, outline, score`
  - constants `PASS_M=0.7, START_SKIP_M=0.3, NEAR_CAMERA_M=0.35, MIN_VISIBLE_CELLS=4, MIN_AREA=0.01, KEEP_NEAR_ROUTE_M=1.5, MAX_VIS_SAMPLES=200, OFFSET_FLOOR_M=0.1, ROUTE_STEP_M=0.05, CROP_MARGIN=0.15, PAIR_ARC_M=0.5`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_route_events.py`:

```python
"""Route events (prompt d1): geometry on synthetic occupancy grids + real-frame regressions.

Synthetic maps are 5 m x 5 m at 5 cm; the camera sits at (2.5, 0.2) facing +z, so the
camera's right is -x and image-left is +x (LEFT below).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rover_vlm.landmarks import FREE, OCC
from rover_vlm.route_events import (
    NEAR_CAMERA_M, candidates, first_blocker, load_frame, route_geometry, visible_cells,
)

LEFT = np.array([1.0, 0.0])


def _free():
    return np.full((100, 100), FREE, np.uint8)


def write_sample(d, state, routes, *, pose=(2.5, 0.2), heading=(0.0, 1.0), depth=None):
    """A synthetic sample dir: state.npz, meta.json (one candidate per route), optional depth.png."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "state.npz", state=state, mpp=0.05, x_min=0.0, z_min=0.0)
    cands = [{"label": "left", "polyline_xz": [list(map(float, p)) for p in r]} for r in routes]
    meta = {"pose_xz": list(pose), "heading_xz": list(heading), "flavor": "detour",
            "chord": {"crossing_xz": None}, "candidates": cands}
    (d / "meta.json").write_text(json.dumps(meta))
    if depth is not None:
        Image.fromarray(depth.astype(np.uint8), "L").save(d / "depth.png")
    return d


# S-route: bows to +x (left) past box A, then to -x (right) past box B
S_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (1.8, 2.4), (1.8, 3.0), (2.5, 3.6)]


def _s_state():
    s = _free()
    s[20:30, 52:58] = OCC        # box A: x 2.6..2.9, z 1.0..1.5 (on the route's right there)
    s[46:56, 44:54] = OCC        # box B: x 2.2..2.7, z 2.3..2.8 (on the route's left there)
    return s


def test_route_geometry_right_vector_points_right_of_travel():
    rg = route_geometry([(0.0, 0.0), (0.0, 2.0)])
    assert np.allclose(rg["right"][5], [-1.0, 0.0])      # travel +z -> right is -x (Camera.right)
    assert rg["arc"][-1] == pytest.approx(2.0)
    assert rg["offset"].max() == pytest.approx(0.0)


def test_each_obstacle_gets_its_own_side_on_an_s_route(tmp_path):
    _, frame = load_frame(write_sample(tmp_path / "s", _s_state(), [S_ROUTE]))
    cs = candidates(frame, np.array(S_ROUTE))
    sides = [c["side"] for c in cs]
    assert sides == ["left", "right"]                    # route order: A then B
    assert all(c["pass_m"] <= 0.7 for c in cs)
    assert cs[0]["outline"]["area"] >= 0.01 and len(cs[0]["outline"]["polygon"]) >= 3
    assert cs[0]["score"] > 0


def test_obstacle_beyond_pass_distance_is_not_a_candidate(tmp_path):
    s = _free()
    s[20:30, 52:58] = OCC                                # box A as above
    route = [(2.5, 0.2), (4.2, 1.2), (4.2, 1.6), (2.5, 3.6)]   # passes A 1.3 m away
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    assert candidates(frame, np.array(route)) == []


def test_occupied_cells_at_the_camera_do_not_hide_what_is_behind(tmp_path):
    # 00372 (2026-09-30): a strip 0.31 m ahead of the rover, bare floor in the image, was
    # the first occupied cell on every ray and hid the armchair.
    s = _s_state()
    s[9:11, 46:54] = OCC                                 # strip z 0.45..0.55, 0.25-0.35 m ahead
    _, frame = load_frame(write_sample(tmp_path / "s", s, [S_ROUTE]))
    target = frame.grid.xz([[25, 55]])[0]                # a cell of box A
    assert first_blocker(frame, target) == int(frame.lab[25, 55])
    cs = candidates(frame, np.array(S_ROUTE))
    assert [c["side"] for c in cs] == ["left", "right"]
    strip = int(frame.lab[10, 50])
    assert strip not in {c["blob"] for c in cs}


def test_hidden_cells_are_not_visible(tmp_path):
    s = _free()
    s[20:22, 40:60] = OCC                                # screen x 2.0..3.0 at z 1.0..1.1
    s[40:44, 48:52] = OCC                                # post behind it, x 2.4..2.6, z 2.0..2.2
    _, frame = load_frame(write_sample(tmp_path / "s", s, [S_ROUTE]))
    post = np.argwhere(frame.lab == frame.lab[41, 49])
    assert len(visible_cells(frame, post, int(frame.lab[41, 49]))) == 0


def test_near_camera_constant():
    assert NEAR_CAMERA_M == pytest.approx(0.35)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; uv run pytest tests/test_route_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rover_vlm.route_events'`.

- [ ] **Step 3: Write the implementation**

Create `src/rover_vlm/route_events.py`:

```python
"""Route descriptions (prompt d1): what a route passes, in order, and where it stops.

The i4 spot check (2026-09-30) showed one geometry-chosen landmark per route is often the
wrong one — the straight entry->goal line grazes a wall while the route bends round a pillar —
and one landmark under-specifies a path once the goal point leaves the prompt. Here every
salient obstacle the route passes is a candidate; the best three become events (pass /
between / through), a stop phrase says where the route ends, and each object's outline is
trimmed with the frame's depth so it does not swallow what stands behind. Cosmos later only
names the objects and phrases the sentence (rover_vlm.descriptions).

Pure numpy, CPU only. Spec: docs/superpowers/specs/2026-09-30-route-descriptions-design.md
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .landmarks import (
    OBJECT_TOP_M, OCC, Camera, Grid, _dilate, _poly_area, column_in_frame, components,
    convex_hull, densify,
)

PASS_M = 0.7              # an obstacle this close to the route is something it passes
START_SKIP_M = 0.3        # ignore what the route passes in its first metres (the rover's own spot)
# The rover's start pose is collision-free: occupied cells this close to the camera are map
# artefacts or things below the lens, never occluders or landmarks (00372, 2026-09-30).
NEAR_CAMERA_M = 0.35
MIN_VISIBLE_CELLS = 4
MIN_AREA = 0.01           # outline area as a fraction of the image
KEEP_NEAR_ROUTE_M = 1.5   # outline only the part of a long blob (a wall) near the route
MAX_VIS_SAMPLES = 200     # visibility rays per blob (each ray is a python loop)
OFFSET_FLOOR_M = 0.1      # score floor so obstacles near the start still count
ROUTE_STEP_M = 0.05
CROP_MARGIN = 0.15        # naming crop = outline box + this fraction of its size
PAIR_ARC_M = 0.5          # two obstacles on opposite sides this close along the route = "between"


@dataclass
class Frame:
    name: str
    grid: Grid
    cam: Camera
    lab: np.ndarray                  # gap-merged blob label per occupied cell, 0 elsewhere
    depth: np.ndarray | None         # depth.png as float (0-255, per-frame min-max), or None
    depth_fit: tuple | None = None   # (scale, offset, resid_m): metres = scale * pixel + offset


def load_frame(sample_dir):
    """(meta, Frame) for one sample dir. Blobs one empty cell apart are one obstacle."""
    sd = Path(sample_dir)
    meta = json.loads((sd / "meta.json").read_text())
    grid = Grid.load(sd / "state.npz")
    occ = grid.s == OCC
    merged, _ = components(_dilate(occ))
    dp = sd / "depth.png"
    depth = np.asarray(Image.open(dp), dtype=float) if dp.exists() else None
    frame = Frame(sd.name, grid, Camera(meta["pose_xz"], meta["heading_xz"]),
                  np.where(occ, merged, 0), depth)
    return meta, frame


def first_blocker(frame, xz):
    """Blob label of the first occupied cell between the camera and xz, or None.

    Cells within NEAR_CAMERA_M of the camera are skipped (see NEAR_CAMERA_M)."""
    for p in densify(np.array([frame.cam.pos, xz], float), frame.grid.mpp / 2)[1:]:
        if np.linalg.norm(p - frame.cam.pos) <= NEAR_CAMERA_M:
            continue
        r, c = (int(round(v)) for v in frame.grid.rc(p)[0])
        if frame.grid.cell((r, c)) == OCC:
            return int(frame.lab[r, c])
    return None


def visible_cells(frame, cells_rc, label):
    """World xz of the cells of blob `label` the camera sees (column in frame, not hidden)."""
    cells_rc = np.asarray(cells_rc)
    if len(cells_rc) > MAX_VIS_SAMPLES:
        cells_rc = cells_rc[np.linspace(0, len(cells_rc) - 1, MAX_VIS_SAMPLES).astype(int)]
    xz = frame.grid.xz(cells_rc)
    ok = column_in_frame(frame.cam, xz) & (np.linalg.norm(xz - frame.cam.pos, axis=1) > NEAR_CAMERA_M)
    vis = [q for q, k in zip(xz, ok) if k and first_blocker(frame, q) in (None, label)]
    return np.array(vis, float).reshape(-1, 2)


def route_geometry(route_xz):
    """Resampled route: points, arclength, right-of-travel normals, offset from entry->goal line."""
    pts = densify(np.asarray(route_xz, float), ROUTE_STEP_M)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    right = np.stack([-d[:, 1], d[:, 0]], 1)        # Camera.right's handedness
    a, g = pts[0], pts[-1]
    ch = (g - a) / max(np.linalg.norm(g - a), 1e-9)
    offset = np.abs((pts - a) @ np.array([-ch[1], ch[0]]))
    return {"pts": pts, "arc": arc, "right": right, "offset": offset}


def crop_box(bbox, margin=CROP_MARGIN):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    m = margin * max(w, h, 0.05)
    return [max(0.0, bbox[0] - m), max(0.0, bbox[1] - m), min(1.0, bbox[2] + m), min(1.0, bbox[3] + m)]


def outline_dict(poly, depth_fallback):
    poly = [[float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))] for u, v in poly]
    if len(poly) < 3:
        return {"polygon": poly, "bbox": None, "crop_box": None, "area": 0.0, "depth_fallback": depth_fallback}
    us, vs = zip(*poly)
    bbox = [min(us), min(vs), max(us), max(vs)]
    return {"polygon": poly, "bbox": bbox, "crop_box": crop_box(bbox),
            "area": round(_poly_area(poly), 5), "depth_fallback": depth_fallback}


def column_outline(frame, vis_xz, top=OBJECT_TOP_M):
    """i4's outline: hull of the visible cells' floor-to-`top` columns (no depth)."""
    pts = np.vstack([frame.cam.uv(vis_xz)[0], frame.cam.uv(vis_xz, top)[0]])
    return outline_dict(convex_hull(pts), depth_fallback=True)


def candidates(frame, route_xz, outline=None):
    """Every salient obstacle the route passes, in route order (see module docstring)."""
    outline = outline or column_outline
    rg = route_geometry(route_xz)
    usable = np.nonzero(rg["arc"] > START_SKIP_M)[0]
    if not len(usable):
        return []
    pts = rg["pts"][usable]
    out = []
    for b in np.unique(frame.lab[frame.lab > 0]):
        cells = np.argwhere(frame.lab == b)
        xz = frame.grid.xz(cells)
        dist = np.linalg.norm(xz[:, None] - pts[None], axis=2)
        j, k = np.unravel_index(dist.argmin(), dist.shape)
        if dist[j, k] > PASS_M:
            continue
        near = dist.min(1) <= KEEP_NEAR_ROUTE_M
        vis = visible_cells(frame, cells[near], int(b))
        if len(vis) < MIN_VISIBLE_CELLS:
            continue
        o = outline(frame, vis)
        if o["area"] < MIN_AREA:
            continue
        i = usable[k]
        on_right = (xz[j] - rg["pts"][i]) @ rg["right"][i] > 0
        ext = xz.max(0) - xz.min(0)
        out.append({
            "blob": int(b), "pass_m": float(dist[j, k]), "arc_m": float(rg["arc"][i]),
            "side": "left" if on_right else "right",
            "offset_m": float(rg["offset"][i]), "extent_m": float(np.hypot(*ext)),
            "gap": _threaded_gap(xz, dist, usable, rg), "vis_xz": vis, "outline": o,
            "score": float(np.sqrt(o["area"]) * max(rg["offset"][i], OFFSET_FLOOR_M)),
        })
    return sorted(out, key=lambda c: c["arc_m"])


def _threaded_gap(xz, dist, usable, rg):
    """(p, q) when the route threads through this blob (cells within PASS_M on both sides at
    about the same arclength — a doorway in one connected wall), else None."""
    close = np.nonzero(dist.min(1) <= PASS_M)[0]
    if not len(close):
        return None
    k = dist[close].argmin(1)
    i = usable[k]
    s = np.einsum("ij,ij->i", xz[close] - rg["pts"][i], rg["right"][i])
    left, right = close[s < 0], close[s > 0]
    if not (len(left) and len(right)):
        return None
    arc = dict(zip(close, rg["arc"][i]))
    d = np.linalg.norm(xz[left][:, None] - xz[right][None], axis=2)
    a, b = np.unravel_index(d.argmin(), d.shape)
    if abs(arc[left[a]] - arc[right[b]]) > PAIR_ARC_M:
        return None
    return (xz[left[a]], xz[right[b]])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_route_events.py -v`
Expected: 6 passed. If `test_each_obstacle_gets_its_own_side_on_an_s_route` fails on the area check, print `cs` and check the box geometry against the comments before changing any constant.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py tests/test_route_events.py
git commit -m "feat: route_events candidates — per-obstacle side, near-camera occlusion fix

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Depth calibration and depth-trimmed outlines (group B)

**Files:**
- Modify: `src/rover_vlm/route_events.py` (`load_frame`, new `calibrate_depth`, `object_outline`; `candidates` default outline)
- Test: `tests/test_route_events.py`

**Interfaces:**
- Consumes: Task 1's `Frame`, `load_frame`, `column_outline`, `outline_dict`, `candidates`.
- Produces:
  - `calibrate_depth(depth, cam, grid) -> (scale, offset, resid_m) | None`
  - `object_outline(frame, vis_xz, top=OBJECT_TOP_M) -> outline dict` (trimmed; falls back to `column_outline` with `depth_fallback=True`)
  - `load_frame` now sets `frame.depth_fit`; `candidates(..., outline=None)` defaults to `object_outline`.
  - constants `DEPTH_BAND_M=0.3, DEPTH_FIT_MAX_RESID_M=0.3, FIT_MAX_M=8.0, MIN_MASK_PX=20`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_route_events.py`)

```python
from PIL import ImageDraw  # noqa: E402

from rover_vlm.route_events import calibrate_depth, column_outline, object_outline  # noqa: E402

LO, HI = 0.5, 6.0     # the synthetic frame's min-max range (the generator does not store it)


def _depth_image(cam, obj_poly_uv, obj_fwd, H=128, W=128):
    """Planar depth of a level camera over a flat floor, far wall at HI above the horizon,
    one object region at obj_fwd; min-max normalised to 0..255 like the generator."""
    v = (np.arange(H) + 0.5) / H
    floor = np.where(v > 0.5, 0.5 * 0.8 / np.maximum(v - 0.5, 1e-6), HI)
    m = np.repeat(np.minimum(floor, HI)[:, None], W, 1)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(u * W, q * H) for u, q in obj_poly_uv], fill=1)
    m[np.asarray(mask, bool)] = obj_fwd
    return np.round((m - LO) / (HI - LO) * 255)


def _box_frame(tmp_path, with_depth=True):
    s = _free()
    s[28:34, 48:54] = OCC                        # box x 2.4..2.7, z 1.4..1.7 (1.2-1.5 m ahead)
    route = [(2.5, 0.2), (3.2, 1.2), (3.2, 1.9), (2.5, 3.0)]
    d = write_sample(tmp_path / "noimg", s, [route])
    _, f0 = load_frame(d)
    vis = visible_cells(f0, np.argwhere(f0.lab == f0.lab[30, 50]), int(f0.lab[30, 50]))
    depth = None
    if with_depth:
        half = column_outline(f0, vis, top=0.5)["polygon"]   # the object is 0.5 m tall
        depth = _depth_image(f0.cam, half, 1.3)
    d = write_sample(tmp_path / ("d" if with_depth else "n"), s, [route], depth=depth)
    return load_frame(d)[1], vis


def test_calibration_recovers_the_frame_range(tmp_path):
    frame, _ = _box_frame(tmp_path)
    scale, offset, resid = frame.depth_fit
    assert scale == pytest.approx((HI - LO) / 255, rel=0.05)
    assert offset == pytest.approx(LO, abs=0.1)
    assert resid < 0.1


def test_trimmed_outline_drops_the_background_above_a_low_object(tmp_path):
    frame, vis = _box_frame(tmp_path)
    col = column_outline(frame, vis)
    trim = object_outline(frame, vis)
    assert trim["depth_fallback"] is False
    assert trim["bbox"][1] > col["bbox"][1] + 0.02        # top edge moved down: the far wall is gone
    assert trim["area"] < col["area"]


def test_no_depth_png_falls_back_to_the_column_outline(tmp_path):
    frame, vis = _box_frame(tmp_path, with_depth=False)
    assert frame.depth is None and frame.depth_fit is None
    o = object_outline(frame, vis)
    assert o["depth_fallback"] is True and o["polygon"] == column_outline(frame, vis)["polygon"]


def test_poor_fit_falls_back(tmp_path):
    frame, vis = _box_frame(tmp_path)
    frame.depth = np.random.default_rng(0).uniform(0, 255, frame.depth.shape)
    frame.depth_fit = calibrate_depth(frame.depth, frame.cam, frame.grid)
    assert frame.depth_fit is None or frame.depth_fit[2] > 0.3
    assert object_outline(frame, vis)["depth_fallback"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_route_events.py -v -k "calibration or trimmed or depth or poor_fit"`
Expected: FAIL — `ImportError: cannot import name 'calibrate_depth'`.

- [ ] **Step 3: Write the implementation**

In `src/rover_vlm/route_events.py`, add `ImageDraw` to the PIL import, `CAM_HEIGHT_M, FREE` to the landmarks import, then add after `column_outline`:

```python
DEPTH_BAND_M = 0.3            # object pixels lie within its own depth range +- this
DEPTH_FIT_MAX_RESID_M = 0.3   # worse fit -> column outline, flagged depth_fallback
FIT_MAX_M = 8.0               # floor pixels farther than this are too coarse to fit on
MIN_MASK_PX = 20


def calibrate_depth(depth, cam, grid, step=8):
    """(scale, offset, resid_m) with metres = scale * pixel + offset, or None.

    depth.png is min-max normalised per frame and the range is not stored
    (spaceitup-habitat/src/roverbench/dataset/writer.py:_depth_png). A floor pixel's planar
    depth is known from the camera height alone, so the range is re-fitted on pixels whose
    floor point is a FREE map cell. Pixels showing an object in front of free floor are
    outliers; two trimmed refits drop them."""
    H, W = depth.shape
    vals, metres = [], []
    for r in range(int(H * 0.55), H, max(1, H // 64)):
        v = (r + 0.5) / H
        fwd = 0.5 * CAM_HEIGHT_M / (v - 0.5)
        if fwd > FIT_MAX_M:
            continue
        for c in range(0, W, max(1, W // 64)):
            u = (c + 0.5) / W
            xz = cam.pos + fwd * cam.h + (u - 0.5) * 2 * fwd * cam.right
            if depth[r, c] > 0 and grid.cell(grid.rc(xz)[0]) == FREE:
                vals.append(depth[r, c])
                metres.append(fwd)
    if len(vals) < 50:
        return None
    x, y = np.array(vals), np.array(metres)
    for _ in range(2):
        a, b = np.polyfit(x, y, 1)
        res = np.abs(a * x + b - y)
        keep = res <= max(DEPTH_BAND_M, np.percentile(res, 80))
        if keep.sum() < 20:
            break
        x, y = x[keep], y[keep]
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b), float(np.median(np.abs(a * x + b - y)))


def object_outline(frame, vis_xz, top=OBJECT_TOP_M):
    """The column outline trimmed to pixels at the object's own depth (group B: the column
    of a coffee table also covered the sofa behind it, and Cosmos named the sofa)."""
    col = column_outline(frame, vis_xz, top)
    fit = frame.depth_fit
    if frame.depth is None or fit is None or fit[2] > DEPTH_FIT_MAX_RESID_M or len(col["polygon"]) < 3:
        return col
    H, W = frame.depth.shape
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(u * W, v * H) for u, v in col["polygon"]], fill=1)
    metres = fit[0] * frame.depth + fit[1]
    fwd = (np.asarray(vis_xz) - frame.cam.pos) @ frame.cam.h
    keep = (np.asarray(mask, bool) & (frame.depth > 0)
            & (metres >= fwd.min() - DEPTH_BAND_M) & (metres <= fwd.max() + DEPTH_BAND_M))
    if keep.sum() < MIN_MASK_PX:
        return col
    ys, xs = np.nonzero(keep)
    pts = np.stack([(xs + 0.5) / W, (ys + 0.5) / H], 1)[:: max(1, len(xs) // 4000)]
    return outline_dict(convex_hull(pts), depth_fallback=False)
```

In `load_frame`, after building `frame`, add:

```python
    if depth is not None:
        frame.depth_fit = calibrate_depth(depth, frame.cam, grid)
```

In `candidates`, change the first line to `outline = outline or object_outline`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_route_events.py -v`
Expected: all passed (Task 1 tests still pass: synthetic frames without `depth.png` use the column outline).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py tests/test_route_events.py
git commit -m "feat: route_events depth calibration and trimmed outlines

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Events, stop phrase, describe_route and real-frame regressions

**Files:**
- Modify: `src/rover_vlm/route_events.py`
- Test: `tests/test_route_events.py`

**Interfaces:**
- Consumes: Tasks 1–2 (`candidates`, `object_outline`, `outline_dict`, `route_geometry`, `visible_cells`, `load_frame`).
- Produces:
  - `opening_outline(frame, p_xz, q_xz) -> outline dict + "gap_xz": [[x, z], [x, z]]`
  - `build_events(frame, cands) -> list[dict]` (route order, ≤ `MAX_EVENTS`): `type ("pass"|"between"|"through"), side ("left"|"right"|None), members [cand indices], arc_m, score, opening (through only)`
  - `stop_phrase(frame, route_xz, events) -> dict | None`: `type ("near"|"after_last"|"distance"), relation, distance_m (distance only), anchor (dict with blob, vis_xz, outline) | None`
  - `describe_route(sample_dir, candidate_index) -> dict`: on success `{"reason": None, "events": [{"type", "side", "objects": [object_id], "arc_m", "gap_xz"?}], "stop": {"type", "relation", "distance_m", "object"}, "objects": {object_id: {"object_id", "kind_hint" ("object"|"opening"), "polygon", "bbox", "crop_box", "area", "depth_fallback", "cells_xz"}}, "goal_xz", "pose_xz", "heading_xz", "route_len_m"}`; on failure `{"reason": <REASONS>, "events": [], "stop": None, "objects": {}, "diag": {...}}`
  - `collect_objects(rows) -> (rows_without_objects, objects_list)` — first occurrence of an object id wins; adds `"image"` from the row.
  - constants `MAX_EVENTS=3, WALL_EXTENT_M=1.5, OPENING_TOP_M=2.0, STOP_NEAR_M=1.0, AFTER_LAST_M=1.5, STOP_FAR_M=4.0, APPROACH_COS=0.866, DISTANCE_STEPS=(1.0, 1.5, 2.0, 3.0, 4.0), REASONS=("no_state", "bad_candidate", "no_stop_anchor", "error")`
  - Object ids: `f"{frame.name}:b{blob}"` for objects, `f"{frame.name}:gap{blob}"` / `f"{frame.name}:gap{a}-{b}"` for openings — both routes of a pair share them.

- [ ] **Step 1: Write the failing tests** (append)

```python
from rover_vlm.route_events import (  # noqa: E402
    MAX_EVENTS, build_events, collect_objects, describe_route, stop_phrase,
)


def test_opposite_sides_close_along_the_route_merge_into_between(tmp_path):
    s = _free()
    s[40:46, 62:68] = OCC        # x 3.1..3.4, z 2.0..2.3 : route's left there (0.625 m)
    s[40:46, 32:38] = OCC        # x 1.6..1.9, z 2.0..2.3 : route's right there (0.625 m)
    route = [(2.5, 0.2), (2.5, 1.5), (2.5, 2.8), (2.6, 3.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert [e["type"] for e in ev] == ["between"] and ev[0]["side"] is None
    assert len(ev[0]["members"]) == 2


def test_doorway_in_one_wall_is_a_through_event(tmp_path):
    s = _free()
    s[40:42, 10:47] = OCC        # wall z 2.0..2.1, x 0.5..2.35
    s[40:42, 53:90] = OCC        # wall x 2.65..4.5 (gap 2.35..2.65 = doorway)
    s[10:42, 8:10] = OCC         # side walls joining both halves into one blob
    s[10:42, 90:92] = OCC
    s[8:10, 8:92] = OCC
    route = [(2.5, 0.8), (2.5, 2.05), (2.5, 3.6)]
    d = write_sample(tmp_path / "s", s, [route], pose=(2.5, 0.8))
    _, frame = load_frame(d)
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert [e["type"] for e in ev] == ["through"]
    p, q = ev[0]["opening"]["gap_xz"]
    assert abs(p[0] - q[0]) == pytest.approx(0.3, abs=0.1)


def test_top_three_by_score_then_route_order(tmp_path):
    s = _free()
    for z0 in (16, 30, 44, 58, 72):                      # five posts, alternating sides
        x0 = 56 if (z0 // 14) % 2 else 40
        s[z0:z0 + 4, x0:x0 + 4] = OCC
    route = [(2.5, 0.2), (2.5, 4.6)]
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert len(ev) <= MAX_EVENTS
    assert [e["arc_m"] for e in ev] == sorted(e["arc_m"] for e in ev)


def _stop_state():
    s = _free()
    s[20:30, 52:58] = OCC                                # box A (passed on the left)
    s[70:80, 36:60] = OCC                                # cabinet x 1.8..3.0, z 3.5..4.0 (>= 1% of the image)
    return s


STOP_ROUTE = [(2.5, 0.2), (3.3, 1.0), (3.3, 1.6), (2.5, 2.9)]   # ends 0.6 m before the cabinet


def test_stop_near_an_object_ahead_is_in_front_of(tmp_path):
    route = STOP_ROUTE
    _, frame = load_frame(write_sample(tmp_path / "s", _stop_state(), [route]))
    st = stop_phrase(frame, np.array(route), [])
    assert st["type"] == "near" and st["relation"] == "in front of"


def test_stop_just_after_the_last_event(tmp_path):
    s = _free()
    s[20:30, 52:58] = OCC
    route = [(2.5, 0.2), (3.3, 1.0), (3.3, 2.7)]      # goal 1.3 m from box A, 1.2 m of route after it
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    ev = build_events(frame, candidates(frame, np.array(route)))
    assert stop_phrase(frame, np.array(route), ev)["type"] == "after_last"


def test_stop_at_a_rough_distance_before_an_object_ahead(tmp_path):
    s = _free()
    s[90:94, 30:70] = OCC                                # sideboard x 1.5..3.5, z 4.5..4.7
    route = [(2.5, 0.2), (2.5, 2.5)]                     # goal 2.0 m short of it, nothing passed
    _, frame = load_frame(write_sample(tmp_path / "s", s, [route]))
    st = stop_phrase(frame, np.array(route), [])
    assert st["type"] == "distance" and st["relation"] == "before" and st["distance_m"] == 2.0


def test_open_floor_goal_with_nothing_around_has_no_stop(tmp_path):
    route = [(2.5, 0.2), (2.5, 2.5)]
    _, frame = load_frame(write_sample(tmp_path / "s", _free(), [route]))
    assert stop_phrase(frame, np.array(route), []) is None


def test_describe_route_rows_and_object_ids(tmp_path):
    route = STOP_ROUTE
    d = write_sample(tmp_path / "s00001", _stop_state(), [route])
    out = describe_route(d, 0)
    assert out["reason"] is None
    assert out["events"][0]["type"] == "pass" and out["events"][0]["side"] == "left"
    oid = out["events"][0]["objects"][0]
    assert oid.startswith("s00001:b") and out["objects"][oid]["kind_hint"] == "object"
    assert out["stop"]["object"] in out["objects"]
    assert 4 <= len(out["objects"][oid]["cells_xz"]) <= 40
    assert describe_route(d, 5)["reason"] == "bad_candidate"
    (d / "state.npz").unlink()
    assert describe_route(d, 0)["reason"] == "no_state"


def test_collect_objects_dedupes_shared_objects():
    o = {"object_id": "f:b1", "polygon": [], "kind_hint": "object"}
    rows = [{"id": "f_L", "image": "img.png", "objects": {"f:b1": o}},
            {"id": "f_R", "image": "img.png", "objects": {"f:b1": dict(o, polygon=[[0, 0]])}}]
    kept, objects = collect_objects(rows)
    assert [r.get("objects") for r in kept] == [None, None]
    assert len(objects) == 1 and objects[0]["polygon"] == [] and objects[0]["image"] == "img.png"


# --- real frames (NFS); skipped where the dataset is not mounted --------------------------
AUG = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated_aug/train")
ORIG = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/train")
REAL = [
    # (sample dir, candidate, world xz of the obstacle a person names, audit group)
    (AUG / "00372-GMwtBqNLGBs/samples/00372-GMwtBqNLGBs_c030_y1", 0, (-0.79, 10.86), "C: armchair hidden by a strip at the camera"),
    (ORIG / "00206-uhkqDVMtEnn/samples/00206-uhkqDVMtEnn_c040", 4, (6.82, -1.21), "A: pillar, not the grazed wall"),
    (AUG / "00475-g7hUFVNac26/samples/00475-g7hUFVNac26_c013_y1", 2, (5.34, 1.97), "A: post on the line, not the sofa"),
]


@pytest.mark.parametrize("sd,cand,xz,why", REAL, ids=[r[3] for r in REAL])
def test_real_frame_names_the_obstacle_a_person_would(sd, cand, xz, why):
    if not (sd / "state.npz").exists():
        pytest.skip("dataset not mounted")
    out = describe_route(sd, cand)
    assert out["reason"] is None, out
    event_objects = [o for e in out["events"] for o in e["objects"]]
    near = [o for o in event_objects
            if np.linalg.norm(np.array(out["objects"][o]["cells_xz"]) - xz, axis=1).min() <= 0.4]
    assert near, f"{why}: events {out['events']}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_route_events.py -v`
Expected: FAIL — `ImportError: cannot import name 'MAX_EVENTS'`.

- [ ] **Step 3: Write the implementation** (append to `src/rover_vlm/route_events.py`)

```python
MAX_EVENTS = 3
WALL_EXTENT_M = 1.5        # two blobs this long on both sides of the route = an opening
OPENING_TOP_M = 2.0        # an opening's outline reaches this high (a door)
STOP_NEAR_M = 1.0
AFTER_LAST_M = 1.5
STOP_FAR_M = 4.0
APPROACH_COS = 0.866       # within 30 degrees of the final heading, ahead or behind
DISTANCE_STEPS = (1.0, 1.5, 2.0, 3.0, 4.0)
REASONS = ("no_state", "bad_candidate", "no_stop_anchor", "error")
CHECK_CELLS = 40           # cells per object kept for the eval's follow check


def opening_outline(frame, p, q):
    """Floor-to-OPENING_TOP_M quad between the two cells flanking a gap."""
    ends = np.array([p, q], float)
    lo, _ = frame.cam.uv(ends)
    hi, _ = frame.cam.uv(ends, OPENING_TOP_M)
    o = outline_dict(convex_hull(np.vstack([lo, hi])), depth_fallback=True)
    o["gap_xz"] = [[round(float(v), 3) for v in p], [round(float(v), 3) for v in q]]
    return o


def build_events(frame, cands):
    """Top-MAX_EVENTS events by score, returned in route order."""
    events, used = [], set()
    for k, a in enumerate(cands):
        if k in used:
            continue
        used.add(k)
        if a["gap"] is not None and a["extent_m"] >= WALL_EXTENT_M:
            events.append({"type": "through", "side": None, "members": [k], "arc_m": a["arc_m"],
                           "score": a["score"], "opening": opening_outline(frame, *a["gap"])})
            continue
        m = next((m for m, b in enumerate(cands) if m not in used and b["side"] != a["side"]
                  and abs(b["arc_m"] - a["arc_m"]) <= PAIR_ARC_M), None)
        if m is None:
            events.append({"type": "pass", "side": a["side"], "members": [k], "arc_m": a["arc_m"],
                           "score": a["score"]})
            continue
        b = cands[m]
        used.add(m)
        ev = {"type": "between", "side": None, "members": [k, m],
              "arc_m": min(a["arc_m"], b["arc_m"]), "score": a["score"] + b["score"]}
        if a["extent_m"] >= WALL_EXTENT_M and b["extent_m"] >= WALL_EXTENT_M:
            d = np.linalg.norm(a["vis_xz"][:, None] - b["vis_xz"][None], axis=2)
            i, j = np.unravel_index(d.argmin(), d.shape)
            ev.update(type="through", opening=opening_outline(frame, a["vis_xz"][i], b["vis_xz"][j]))
        events.append(ev)
    top = sorted(events, key=lambda e: -e["score"])[:MAX_EVENTS]
    return sorted(top, key=lambda e: e["arc_m"])


def _stop_anchors(frame, goal):
    """Visible, salient blobs with a cell within STOP_FAR_M of the goal."""
    out = []
    for b in np.unique(frame.lab[frame.lab > 0]):
        cells = np.argwhere(frame.lab == b)
        xz = frame.grid.xz(cells)
        d = np.linalg.norm(xz - goal, axis=1)
        if d.min() > STOP_FAR_M:
            continue
        vis = visible_cells(frame, cells[d <= STOP_FAR_M + KEEP_NEAR_ROUTE_M], int(b))
        if len(vis) < MIN_VISIBLE_CELLS:
            continue
        o = object_outline(frame, vis)
        if o["area"] < MIN_AREA:
            continue
        out.append({"blob": int(b), "near_xz": xz[d.argmin()], "dist": float(d.min()),
                    "vis_xz": vis, "outline": o})
    return out


def stop_phrase(frame, route_xz, events):
    """Where the route ends: near an object, just after the last event, or a rough distance
    from an object on the line of approach; None when nothing fits (the route is dropped)."""
    rg = route_geometry(route_xz)
    goal = rg["pts"][-1]
    back = rg["pts"][max(0, len(rg["pts"]) - 11)]          # ~0.5 m before the goal
    head = goal - back
    head /= max(np.linalg.norm(head), 1e-9)
    right = np.array([-head[1], head[0]])
    anchors = _stop_anchors(frame, goal)
    near = [x for x in anchors if x["dist"] <= STOP_NEAR_M]
    if near:
        x = min(near, key=lambda x: x["dist"])
        v = x["near_xz"] - goal
        along, lat = float(v @ head), float(v @ right)
        rel = "in front of" if along > abs(lat) else "just past" if along < -abs(lat) else "next to"
        return {"type": "near", "relation": rel, "anchor": x}
    if events and rg["arc"][-1] - events[-1]["arc_m"] <= AFTER_LAST_M:
        return {"type": "after_last", "relation": "just after", "anchor": None}
    far = []
    for x in anchors:
        v = x["near_xz"] - goal
        if abs(v @ head) / max(np.linalg.norm(v), 1e-9) >= APPROACH_COS:
            far.append((x["dist"], x, bool(v @ head > 0)))
    if far:
        dist, x, ahead = min(far, key=lambda t: t[0])
        step = min(DISTANCE_STEPS, key=lambda s: abs(s - dist))
        return {"type": "distance", "relation": "before" if ahead else "past", "distance_m": step, "anchor": x}
    return None


def _none(reason, **diag):
    return {"reason": reason, "events": [], "stop": None, "objects": {}, "diag": diag}


def _cells(xz, n=CHECK_CELLS):
    xz = np.asarray(xz, float)
    if len(xz) > n:
        xz = xz[np.linspace(0, len(xz) - 1, n).astype(int)]
    return [[round(float(a), 3), round(float(b), 3)] for a, b in xz]


def describe_route(sample_dir, candidate_index):
    """The events + stop phrase for one candidate route of one frame (see module docstring)."""
    sd = Path(sample_dir)
    if not (sd / "state.npz").exists():
        return _none("no_state")
    meta, frame = load_frame(sd)
    cm = meta.get("candidates", [])
    if not (0 <= candidate_index < len(cm)) or len(cm[candidate_index].get("polyline_xz") or []) < 2:
        return _none("bad_candidate")
    route = np.asarray(cm[candidate_index]["polyline_xz"], float)
    cands = candidates(frame, route)
    events = build_events(frame, cands)
    stop = stop_phrase(frame, route, events)
    if stop is None:
        return _none("no_stop_anchor", n_events=len(events))
    objects = {}

    def add(key, kind, outline, cells):
        oid = f"{frame.name}:{key}"
        objects.setdefault(oid, {"object_id": oid, "kind_hint": kind,
                                 **{k: outline[k] for k in ("polygon", "bbox", "crop_box", "area", "depth_fallback")},
                                 "cells_xz": _cells(cells)})
        return oid

    rows = []
    for e in events:
        blobs = [cands[m]["blob"] for m in e["members"]]
        if e["type"] == "through":
            ids = [add("gap" + "-".join(map(str, blobs)), "opening", e["opening"], e["opening"]["gap_xz"])]
        else:
            ids = [add(f"b{cands[m]['blob']}", "object", cands[m]["outline"], cands[m]["vis_xz"]) for m in e["members"]]
        row = {"type": e["type"], "side": e["side"], "objects": ids, "arc_m": round(e["arc_m"], 2)}
        if e["type"] == "through":
            row["gap_xz"] = e["opening"]["gap_xz"]
        rows.append(row)
    st = {"type": stop["type"], "relation": stop["relation"], "distance_m": stop.get("distance_m"), "object": None}
    if stop["anchor"] is not None:
        a = stop["anchor"]
        st["object"] = add(f"b{a['blob']}", "object", a["outline"], a["vis_xz"])
    return {"reason": None, "events": rows, "stop": st, "objects": objects,
            "goal_xz": [round(float(v), 3) for v in route[-1]],
            "pose_xz": [float(v) for v in meta["pose_xz"]], "heading_xz": [float(v) for v in meta["heading_xz"]],
            "route_len_m": round(float(route_geometry(route)["arc"][-1]), 2)}


def collect_objects(rows):
    """(rows without their "objects", one object row per id). Both routes of a pair share
    their frame's objects; the first route's outline wins (its name serves both)."""
    objects, kept = {}, []
    for r in rows:
        r = dict(r)
        for oid, o in (r.pop("objects", None) or {}).items():
            objects.setdefault(oid, {**o, "image": r["image"]})
        kept.append(r)
    return kept, list(objects.values())
```

Note for the implementer: `test_describe_route_rows_and_object_ids` expects `kept` rows to have **no** `"objects"` key after `collect_objects` (`r.get("objects")` is `None`); `describe_route` itself returns them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_route_events.py -v`
Expected: all passed; the 3 real-frame tests run on the login node (NFS mounted). If a real-frame test fails, print `out["events"]` with each object's `cells_xz` mean and look at the frame (`scratchpad`-style render) before touching thresholds — a threshold change must be justified by the frame, and all synthetic tests must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py tests/test_route_events.py
git commit -m "feat: route_events events, stop phrase, describe_route

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Templates, text rules and name rules (`descriptions.py`)

**Files:**
- Create: `src/rover_vlm/descriptions.py`
- Test: `tests/test_descriptions.py`

**Interfaces:**
- Consumes (from `rover_vlm.instructions`): `head_noun`, `normalize_name`, `_DRAWING`, `_EGOCENTRIC`, `_stable_index`, `STYLES`, `choose_style`.
- Produces:
  - `DISTANCE_WORDS: dict[float, str]`, `DESC_WORDS = (5, 45)`
  - `event_phrase(event, names_list) -> str`, `stop_text(stop, name) -> str`
  - `template_description(events, stop, names) -> str` (`names`: `{object_id: name}`)
  - `description_rejects(text, events, stop, names) -> list[str]`
  - `name_rejects(name, kind, need) -> list[str]` (`need`: `"object"|"opening"`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_descriptions.py`:

```python
"""Prompt d1: templates, text rules, name rules, Cosmos reply handling, finalize."""
import pytest

from rover_vlm.descriptions import (
    description_rejects, name_rejects, template_description,
)

NAMES = {"f:b1": "coffee table", "f:b2": "stone column", "f:b3": "grey sofa",
         "f:gap4": "open doorway", "f:b5": "fridge", "f:b6": "table"}
PASS_L = {"type": "pass", "side": "left", "objects": ["f:b1"]}
PASS_R = {"type": "pass", "side": "right", "objects": ["f:b3"]}
BETWEEN = {"type": "between", "side": None, "objects": ["f:b2", "f:b3"]}
THROUGH = {"type": "through", "side": None, "objects": ["f:gap4"]}
NEAR = {"type": "near", "relation": "in front of", "distance_m": None, "object": "f:b5"}
AFTER = {"type": "after_last", "relation": "just after", "distance_m": None, "object": None}
DIST = {"type": "distance", "relation": "before", "distance_m": 2.0, "object": "f:b5"}


@pytest.mark.parametrize("events,stop", [
    ([PASS_L], NEAR), ([PASS_L, THROUGH], NEAR), ([BETWEEN], DIST), ([PASS_L, PASS_R], AFTER), ([], NEAR),
])
def test_template_always_passes_its_own_rules(events, stop):
    text = template_description(events, stop, NAMES)
    assert description_rejects(text, events, stop, NAMES) == [], text


def test_template_wording():
    assert template_description([PASS_L, THROUGH], NEAR, NAMES) == (
        "Go left of the coffee table, then go through the open doorway, and stop in front of the fridge.")
    assert template_description([BETWEEN], DIST, NAMES).endswith("stop about two metres before the fridge.")


def test_paraphrases_with_both_side_word_positions_pass():
    ev, st = [PASS_L, PASS_R], NEAR
    ok = ["Keep left of the coffee table, then pass the grey sofa on the right and stop in front of the fridge.",
          "Pass the coffee table on the left, then the grey sofa on the right, and stop in front of the fridge."]
    for t in ok:
        assert description_rejects(t, ev, st, NAMES) == [], t


@pytest.mark.parametrize("text,reason", [
    ("Go right of the coffee table and stop in front of the fridge.", "side:missing:0"),
    ("Go left of the coffee table and stop by the fridge on the left.", "side:stray"),
    ("Stop in front of the fridge.", "name:missing:0"),
    ("Go left of the coffee table and stop.", "stop:name"),
    ("Go left of the coffee table, then the fridge.", "stop:missing"),
    ("Go left of the coffee table on your left and stop in front of the fridge.", "side:egocentric"),
    ("Follow the magenta line left of the coffee table and stop at the fridge.", "drawing:magenta"),
])
def test_rejections(text, reason):
    assert reason in description_rejects(text, [PASS_L], NEAR, NAMES)


def test_names_must_come_in_route_order():
    t = "Stop in front of the fridge after going left of the coffee table."
    assert "stop:name" in description_rejects(t, [PASS_L], NEAR, NAMES)


def test_same_head_noun_twice_matches_in_order():
    names = dict(NAMES, **{"f:b1": "coffee table", "f:b3": "dining table"})
    ev = [PASS_L, {"type": "pass", "side": "right", "objects": ["f:b3"]}]
    t = "Go left of the coffee table, then right of the dining table, and stop in front of the fridge."
    assert description_rejects(t, ev, NEAR, names) == []
    t2 = "Go left of the coffee table and stop in front of the fridge."
    assert "name:missing:1" in description_rejects(t2, ev, NEAR, names)


def test_distance_and_after_rules():
    assert "stop:distance" in description_rejects(
        "Go between the stone column and the grey sofa and stop before the fridge.", [BETWEEN], DIST, NAMES)
    assert description_rejects(
        "Go between the stone column and the grey sofa and stop about two meters before the fridge.",
        [BETWEEN], DIST, NAMES) == []
    assert "stop:after" in description_rejects(
        "Go left of the coffee table, then right of the grey sofa, and stop.", [PASS_L, PASS_R], AFTER, NAMES)


def test_between_and_through_words_required():
    assert "between:missing:0" in description_rejects(
        "Go past the stone column and the grey sofa and stop about two metres before the fridge.",
        [BETWEEN], DIST, NAMES)
    assert "through:missing:0" in description_rejects(
        "Head to the open doorway and stop in front of the fridge.", [THROUGH], NEAR, NAMES)


@pytest.mark.parametrize("name,kind,need,reason", [
    ("big wooden cabinet with doors", "object", "object", "name:long"),
    ("object", "object", "object", "name:generic"),
    ("left cabinet", "object", "object", "name:side"),
    ("outlined region", "object", "object", "name:drawing"),
    ("open doorway", "opening", "object", "kind:mismatch"),
    ("grey sofa", "object", "opening", "kind:mismatch"),
    ("", "object", "object", "name:empty"),
])
def test_name_rejects(name, kind, need, reason):
    assert reason in name_rejects(name, kind, need)


def test_good_names_pass():
    assert name_rejects("grey sofa", "object", "object") == []
    assert name_rejects("white wall", "wall", "object") == []
    assert name_rejects("arched doorway", "opening", "opening") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_descriptions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rover_vlm.descriptions'`.

- [ ] **Step 3: Write the implementation**

Create `src/rover_vlm/descriptions.py`:

```python
"""Route descriptions (prompt d1): turning geometry's events into checked sentences.

rover_vlm.route_events finds what a route passes and where it stops. This module holds the
canonical template sentence, the text rules every description must pass (all names present
in route order, each pass with its own side word, the stop phrase present), the Cosmos
prompts for the two labelling stages (name one object on a crop; compose the sentence from
text only) and the finalize helpers that turn labels into training/eval records.

Pure and CPU-only. Spec: docs/superpowers/specs/2026-09-30-route-descriptions-design.md
"""

import re

from .instructions import _DRAWING, _EGOCENTRIC, head_noun

DISTANCE_WORDS = {1.0: "one metre", 1.5: "one and a half metres", 2.0: "two metres",
                  3.0: "three metres", 4.0: "four metres"}
DESC_WORDS = (5, 45)
_METRE = r"met(?:re|er)s?"
_STOP = re.compile(r"\b(?:stop|stops|stopping|halt|park|come to rest|end up)\b", re.I)
_AFTER = re.compile(r"\b(?:just|right) (?:after|past|beyond)\b", re.I)
_SIDE = re.compile(r"\b(left|right)\b", re.I)
_BREAK = re.compile(r"[,;.]|\bthen\b|\band\b", re.I)
GENERIC_NAMES = {"object", "thing", "item", "area", "region", "stuff", "something", "space", "structure"}


def event_phrase(ev, names):
    if ev["type"] == "pass":
        return f"go {ev['side']} of the {names[0]}"
    if ev["type"] == "between":
        return f"go between the {names[0]} and the {names[1]}"
    return f"go through the {names[0]}"


def stop_text(stop, name):
    if stop["type"] == "near":
        return f"stop {stop['relation']} the {name}"
    if stop["type"] == "after_last":
        return "stop just after it"
    return f"stop about {DISTANCE_WORDS[stop['distance_m']]} {stop['relation']} the {name}"


def template_description(events, stop, names):
    """The canonical sentence for a route; always passes description_rejects."""
    parts = [event_phrase(e, [names[o] for o in e["objects"]]) for e in events]
    body = ", then ".join(parts) if parts else "go straight ahead"
    text = f"{body}, and {stop_text(stop, names.get(stop['object']))}."
    return text[0].upper() + text[1:]


def _find_name(text, name, start):
    noun = head_noun(name)
    if not noun:
        return None
    return re.compile(rf"\b{re.escape(noun)}(?:e?s)?\b", re.I).search(text, start)


def _side_rejects(text, events, spans):
    """Each side word belongs to the next name in its clause, else to the previous name."""
    got = {}
    for m in _SIDE.finditer(text):
        after = [s for s in spans if s[0] >= m.end()]
        nxt = min(after, key=lambda s: s[0]) if after else None
        if nxt is not None and not _BREAK.search(text[m.end():nxt[0]]):
            owner = nxt[2]
        else:
            before = [s for s in spans if s[1] <= m.start()]
            owner = max(before, key=lambda s: s[1])[2] if before else None
        got.setdefault(owner, set()).add(m.group(1).lower())
    out = []
    for i, e in enumerate(events):
        words = got.get(i, set())
        if e["type"] == "pass":
            if e["side"] not in words:
                out.append(f"side:missing:{i}")
            if words - {e["side"]}:
                out.append(f"side:opposite:{i}")
        elif words:
            out.append(f"side:stray:{i}")
    if got.get("stop") or got.get(None):
        out.append("side:stray")
    return out


def description_rejects(text, events, stop, names):
    """Why a description is unusable for these events; [] when it is fine."""
    if not text:
        return ["empty"]
    out = [f"drawing:{m.group(0).lower()}" for m in _DRAWING.finditer(text)]
    if re.search(r"\d", text):
        out.append("digits")
    n = len(text.split())
    if not (DESC_WORDS[0] <= n <= DESC_WORDS[1]):
        out.append(f"length:{n}")
    if _EGOCENTRIC.search(text):
        out.append("side:egocentric")
    spans, pos = [], 0
    for i, e in enumerate(events):
        for oid in e["objects"]:
            m = _find_name(text, names[oid], pos)
            if m is None:
                out.append(f"name:missing:{i}")
                continue
            spans.append((m.start(), m.end(), i))
            pos = m.end()
        if e["type"] == "between" and not re.search(r"\bbetween\b", text, re.I):
            out.append(f"between:missing:{i}")
        if e["type"] == "through" and not re.search(r"\bthrough\b", text, re.I):
            out.append(f"through:missing:{i}")
    if stop["object"]:
        m = _find_name(text, names[stop["object"]], pos)
        if m is None:
            out.append("stop:name")
        else:
            spans.append((m.start(), m.end(), "stop"))
    if not _STOP.search(text):
        out.append("stop:missing")
    if stop["type"] == "after_last" and not _AFTER.search(text):
        out.append("stop:after")
    if stop["type"] == "distance":
        number = DISTANCE_WORDS[stop["distance_m"]].rsplit(" ", 1)[0]
        if not re.search(rf"\b{number}\s+{_METRE}\b", text, re.I):
            out.append("stop:distance")
    return out + _side_rejects(text, events, spans)


def name_rejects(name, kind, need):
    """Why a Cosmos object name is unusable; [] when it is fine."""
    if not name:
        return ["name:empty"]
    out = []
    if len(name.split()) > 4:
        out.append("name:long")
    if head_noun(name) in GENERIC_NAMES or head_noun(name) is None:
        out.append("name:generic")
    if _DRAWING.search(name):
        out.append("name:drawing")
    if _SIDE.search(name):
        out.append("name:side")
    if re.search(r"\d", name):
        out.append("name:digits")
    if (need == "opening") != (kind == "opening"):
        out.append("kind:mismatch")
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_descriptions.py -v`
Expected: all passed. `"outlined region"` hits `_DRAWING` via `outlin\w*`; if `name:generic` also fires that is fine (the test checks membership).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py tests/test_descriptions.py
git commit -m "feat: d1 templates, description and name rules

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `events` stage and the geometry audit page — GATE 1

**Files:**
- Modify: `scripts/prepare_habitat_instr.py` (docstring, `event_row`, `stage_events`, subparser)
- Modify: `scripts/audit_instruction_labels.py` (mode `events`)
- Test: `tests/test_route_events.py` (event_row via importlib)

**Interfaces:**
- Consumes: `describe_route`, `collect_objects`, `REASONS` (Task 3); `template_description` (Task 4); `render_generation_jpeg`, `draw_reference_line` (existing).
- Produces:
  - `data/prepared_habitat_instr_v2/events_{train,eval}.jsonl` (one row per labelset record: `id, sample_dir, image, candidate_index, kind, pair_id, sample_id` + `describe_route` fields minus `objects`)
  - `data/prepared_habitat_instr_v2/objects_{train,eval}.jsonl`, `events_meta.json`
  - `event_row(rec) -> dict` (with `"objects"`), in `prepare_habitat_instr.py`
  - Audit page `outputs/instr_events_audit.html` (title "Route Events Audit"), marks db collection `marks`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_route_events.py`)

```python
import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "prepare_habitat_instr", Path(__file__).resolve().parent.parent / "scripts" / "prepare_habitat_instr.py")
prepare_instr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_instr)


def test_event_row_carries_record_fields(tmp_path):
    route = STOP_ROUTE
    d = write_sample(tmp_path / "s1", _stop_state(), [route])
    rec = {"id": "s1_L", "image": [str(d / "fpv_enhanced.png")],
           "habitat_meta": {"candidate_index": 0, "kind": "instructed", "pair_id": None, "sample_id": "s1"}}
    row = prepare_instr.event_row(rec)
    assert row["id"] == "s1_L" and row["reason"] is None and row["objects"]
    assert row["image"].endswith("fpv_enhanced.png") and row["sample_id"] == "s1"
    rec["habitat_meta"]["candidate_index"] = 9
    assert prepare_instr.event_row(rec)["reason"] == "bad_candidate"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_route_events.py::test_event_row_carries_record_fields -v`
Expected: FAIL — `AttributeError: module 'prepare_habitat_instr' has no attribute 'event_row'`.

- [ ] **Step 3: Implement the stage**

In `scripts/prepare_habitat_instr.py`:

1. Extend the module docstring (after the "Grounded landmarks" block):

```
Route descriptions (2026-09-30, docs/superpowers/specs/2026-09-30-route-descriptions-design.md):
events + stop phrase per labelset record (both routes of a pair), plus one object row per
frame object for Cosmos to name. CPU, 4 single-threaded workers:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run scripts/prepare_habitat_instr.py events --limit 200     # smoke run (prints timing)
    uv run scripts/prepare_habitat_instr.py events
```

2. Add the import `from rover_vlm.route_events import collect_objects, describe_route` and, after `stage_landmarks`:

```python
EVENT_WORKERS = 4


def event_row(rec):
    """One events row for a labelset record (rover_vlm.route_events.describe_route)."""
    hm = rec["habitat_meta"]
    sample_dir = Path(rec["image"][0]).parent
    try:
        d = describe_route(sample_dir, hm["candidate_index"])
    except Exception as e:  # noqa: BLE001 — one bad sample is a counted reason, not a dead stage
        d = {"reason": "error", "events": [], "stop": None, "objects": {},
             "diag": {"error": f"{type(e).__name__}: {e}"}}
    return {"id": rec["id"], "sample_dir": str(sample_dir), "image": rec["image"][0],
            "candidate_index": hm["candidate_index"], "kind": hm.get("kind"),
            "pair_id": hm.get("pair_id"), "sample_id": hm.get("sample_id"), **d}


def stage_events(args):
    import time
    d = args.out_dir
    meta = {}
    for name in SPLITS:
        recs = json.loads((d / f"labelset_{name}.json").read_text())
        if args.limit:
            recs = recs[: args.limit]
        t0 = time.time()
        with Pool(EVENT_WORKERS) as pool:
            rows = pool.map(event_row, recs, chunksize=16)
        rows, objects = collect_objects(rows)
        (d / f"events_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / f"objects_{name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in objects))
        ok = [r for r in rows if r["reason"] is None]
        pairs = Counter(r["pair_id"] for r in ok if r["pair_id"])
        meta[name] = {
            "records": len(rows), "described": len(ok),
            "reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
            "events_per_route": dict(sorted(Counter(len(r["events"]) for r in ok).items())),
            "event_types": dict(Counter(e["type"] for r in ok for e in r["events"])),
            "stop_types": dict(Counter(r["stop"]["type"] for r in ok)),
            "pairs_both_described": sum(1 for n in pairs.values() if n == 2),
            "objects": len(objects),
            "depth_fallback": round(sum(o["depth_fallback"] for o in objects) / max(1, len(objects)), 3),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: {json.dumps(meta[name])}", flush=True)
    (d / "events_meta.json").write_text(json.dumps(meta, indent=2))
```

3. Register the subparser in `main()` and the dispatch dict:

```python
    h = sub.add_parser("events", help="d1: route events + stop phrase per labelset record (CPU)")
    h.add_argument("--out-dir", type=Path, default=V2_OUT)
    h.add_argument("--limit", type=int, default=0, help="first N records per split (smoke run)")
```

and add `"events": stage_events` to the dispatch dict.

- [ ] **Step 4: Run the test, then a smoke run**

Run: `uv run pytest tests/test_route_events.py -v` → all passed.
Then (CPU, login node):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run scripts/prepare_habitat_instr.py events --limit 200
```

Expected: two JSON summary lines. Check: `described` ≥ 70% of `records`; `depth_fallback` < 0.3 (if higher, inspect a few frames' `calibrate_depth` residuals before going on); note `seconds` and extrapolate to 25,985 + 2,866 records (report it; if > 3 h, profile `visible_cells` and lower `MAX_VIS_SAMPLES` before the full run).

- [ ] **Step 5: Add the audit mode** in `scripts/audit_instruction_labels.py`

1. Docstring: add

```
--mode events (d1 geometry gate, 2026-09-30): 100 routes from events_train.jsonl (30 from
pairs), each drawn with its route, the straight line, numbered event outlines (orange) and the
stop object (green), plus the template sentence with "object N" for names. Question: is the
description TRUE of this route, and does it name what a person would? Gate >= 90% true.

    uv run scripts/audit_instruction_labels.py --mode events \
        --events data/prepared_habitat_instr_v2/events_train.jsonl \
        --objects data/prepared_habitat_instr_v2/objects_train.jsonl     # -> outputs/instr_events_audit.html
```

2. Add `"events"` to `VERDICTS`:

```python
    "events": [
        {"v": "ok", "label": "True", "help": "Every step and the stop are true, and it names what you would"},
        {"v": "object", "label": "Wrong object", "help": "A step names something the route does not pass that way"},
        {"v": "missing", "label": "Misses an obstacle", "help": "An obvious obstacle the route goes around is left out"},
        {"v": "stop", "label": "Wrong stop", "help": "The stop phrase does not match where the route ends"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
```

3. Add helpers after `load_landmarks_rows`:

```python
EVENT_COLOR, STOP_COLOR = (255, 170, 0), (60, 220, 60)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sample_events(rows, n_pair, n_single, seed):
    ok = [r for r in rows if r["reason"] is None]
    pairs = [r for r in ok if r["pair_id"]]
    single = [r for r in ok if not r["pair_id"]]
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n_pair, len(pairs))) + rng.sample(single, min(n_single, len(single)))
    rng.shuffle(picked)
    return picked, {"described_pair": len(pairs), "described_single": len(single), "rows": len(rows)}


def placeholder_names(row):
    """{object_id: "object N"} in drawing order (events, then the stop object)."""
    ids = [o for e in row["events"] for o in e["objects"]] + ([row["stop"]["object"]] if row["stop"]["object"] else [])
    return {oid: f"object {i + 1}" for i, oid in enumerate(dict.fromkeys(ids))}


def draw_numbered(img, objects, names, stop_id=None):
    """Outline each named object with its number; the stop object in green, events orange."""
    draw = ImageDraw.Draw(img)
    font = _font(max(16, IMG_PX // 30))
    for oid, label in names.items():
        colour = STOP_COLOR if oid == stop_id else EVENT_COLOR
        pts = [(u * IMG_PX, v * IMG_PX) for u, v in objects[oid]["polygon"]]
        if len(pts) >= 3:
            draw.polygon(pts, outline=colour, width=4)
            cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
            _draw_badge(draw, (cx, cy), label.split()[-1], colour, font, 14)


def events_item(rec, row, objects, text=None):
    names = placeholder_names(row)
    img = Image.open(io.BytesIO(render_generation_jpeg(rec))).resize((IMG_PX, IMG_PX), Image.LANCZOS)
    draw_reference_line(img, _answer_path(rec))
    draw_numbered(img, objects, names, stop_id=row["stop"]["object"])
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    m = rec["habitat_meta"]
    return {"id": rec["id"], "instruction": text or template_description(row["events"], row["stop"], names),
            "side": m["side"], "kind": m["kind"], "has_alt": bool(m.get("alt_path")), "scene": m["scene_id"],
            "target": "object", "reason": None,
            "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
```

4. Imports at the top: `from rover_vlm.habitat_choice import _draw_badge, _font` and `from rover_vlm.descriptions import template_description`.

5. In `main()`: add `"events"` to `--mode` choices, add arguments `--events` and `--objects` (`type=Path`), and before the `elif args.labels:` branch:

```python
    if args.mode == "events":
        if not (args.events and args.objects):
            raise SystemExit("--mode events needs --events events_train.jsonl --objects objects_train.jsonl")
        by_id = {r["id"]: r for r in json.loads((V2 / "labelset_train.json").read_text())}
        objects = {o["object_id"]: o for o in jsonl(args.objects)}
        rows, pop = sample_events(jsonl(args.events), 30, args.n - 30, args.seed)
        items = [events_item(by_id[r["id"]], r, objects) for r in rows]
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "events",
                   "verdicts": VERDICTS["events"], "source": str(args.events)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Route Events Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "instr_events_audit.html"
        out.write_text(html)
        print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
```

- [ ] **Step 6: Full run + page (CPU)**

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run scripts/prepare_habitat_instr.py events            # long: run it in tmux, not a session watcher
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
uv run scripts/audit_instruction_labels.py --mode events \
    --events data/prepared_habitat_instr_v2/events_train.jsonl \
    --objects data/prepared_habitat_instr_v2/objects_train.jsonl
uv run pytest -q
```

Expected: `events_meta.json` written; page ≤ 16 MB; full suite green.

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_habitat_instr.py scripts/audit_instruction_labels.py tests/test_route_events.py
git commit -m "feat: d1 events stage and route-events audit page

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: GATE 1 (human).** Publish `outputs/instr_events_audit.html` as an artifact with the same `db` capability as the earlier audit pages (load the `artifact-capabilities` skill first; the page stores marks in collection `marks`). Give the user the link and **stop until they have marked all 100**. Then read the marks (`ArtifactData list`, collection `marks`), tabulate by verdict and by pair/single, and look at every non-ok frame (render as in the 2026-09-30 review). If < 90% true: change only the constants the failures point to (`PASS_M`, `MIN_AREA`, `STOP_NEAR_M`, `STOP_FAR_M`, `APPROACH_COS`, `WALL_EXTENT_M`), add a synthetic test for each failure pattern first, re-run the stage, and re-audit on a **fresh seed**. Record the result in the memory file `instruction-conditioned-paths.md` and freeze the constants before Task 7's GPU job.

---

### Task 6: Cosmos prompts, reply parsers and the labelling logic

**Files:**
- Modify: `src/rover_vlm/descriptions.py`
- Modify: `src/rover_vlm/labelling_io.py` (`chat_text`)
- Test: `tests/test_descriptions.py`, `tests/test_labelling_io.py`

**Interfaces:**
- Consumes: Task 4 (`description_rejects`, `name_rejects`, `template_description`, `event_phrase`, `stop_text`); `instructions` (`STYLES`, `choose_style`, `_stable_index`, `normalize_name`, `_answer_block`, `TARGET_OUTLINE_COLOR`); `traces` (`SCENE_INTRO`, `parse_trace`, `answer_payload`).
- Produces:
  - `naming_image(image_path, polygon, crop_box, px=512) -> bytes` (JPEG, 2px × px)
  - `naming_prompt(need) -> str`, `parse_name_reply(raw) -> {"name", "kind"} | None`
  - `SHAPES`, `choose_shape(route_id) -> str`
  - `compose_prompt(events, stop, names, style, shape) -> str`, `parse_compose_reply(raw) -> str | None`
  - `label_object(ask, obj, seed) -> row` where `ask(image_bytes, prompt, seed) -> (raw, finish)`; row: `id, need, prompt_version, name, kind, passed, rejects, attempts, finish` (+ `raw` when rejected)
  - `label_route(ask_text, row, names, seed) -> row` where `ask_text(prompt, seed) -> (raw, finish)`; row: `id, pair_id, kind, prompt_version, description, instruction_source ("cosmos"|"template"|None), passed, rejects, cosmos_rejects, style, shape, attempts` (+ `raw` on fallback, `missing` when unnamed)
  - `labelling_io.chat_text(client, model, prompt, *, max_tokens, seed) -> (raw, finish)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_descriptions.py`:

```python
import io  # noqa: E402

from PIL import Image  # noqa: E402

from rover_vlm.descriptions import (  # noqa: E402
    choose_shape, compose_prompt, label_object, label_route, naming_image, naming_prompt,
    parse_compose_reply, parse_name_reply,
)


def _reply(obj):
    import json
    return f"<think>ok</think>\n<answer>\n{json.dumps(obj)}\n</answer>"


def test_parse_name_reply_normalises_kind_and_name():
    assert parse_name_reply(_reply({"name": " Grey  Sofa ", "kind": "Object"})) == {"name": "grey sofa", "kind": "object"}
    assert parse_name_reply(_reply({"name": "doorway", "kind": "doorway"}))["kind"] == "opening"
    assert parse_name_reply(_reply({"name": "white wall panel"}))["name"] == "white wall"
    assert parse_name_reply(_reply({"name": "lamp", "kind": "gizmo", "extra": 1}))["kind"] == "object"
    assert parse_name_reply(_reply({"kind": "object"})) is None
    assert parse_name_reply("no answer here") is None


def test_parse_compose_reply():
    assert parse_compose_reply(_reply({"description": " Go  left of the sofa. "})) == "Go left of the sofa."
    assert parse_compose_reply("<answer>Go left of the sofa and stop.</answer>") == "Go left of the sofa and stop."
    assert parse_compose_reply(_reply({"other": 1})) is None


def test_naming_image_is_two_panels(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (512, 512), (120, 120, 120)).save(p)
    img = Image.open(io.BytesIO(naming_image(p, [[0.2, 0.2], [0.6, 0.2], [0.6, 0.7]], [0.15, 0.15, 0.65, 0.75])))
    assert img.size == (1024, 512)


def test_prompts_mention_what_matters():
    assert "opening" in naming_prompt("opening") and "close-up" in naming_prompt("object")
    p = compose_prompt([PASS_L, THROUGH], NEAR, NAMES, "a polite request", "one sentence")
    assert "go left of the coffee table" in p and "go through the open doorway" in p
    assert "stop in front of the fridge" in p and "a polite request" in p
    assert choose_shape("x_L") == choose_shape("x_L")


def test_label_object_retries_once_then_rejects():
    calls = []

    def ask(img, prompt, seed):
        calls.append(seed)
        return _reply({"name": "outlined object", "kind": "object"}), "stop"
    obj = {"object_id": "f:b1", "kind_hint": "object", "image": None, "polygon": [], "crop_box": None}
    row = label_object(ask, obj, seed=3, image=b"jpeg")
    assert calls == [3, 4] and row["passed"] is False and row["name"] is None and row["rejects"]


def test_label_object_accepts_a_good_name():
    obj = {"object_id": "f:gap4", "kind_hint": "opening", "image": None, "polygon": [], "crop_box": None}
    row = label_object(lambda i, p, s: (_reply({"name": "open doorway", "kind": "opening"}), "stop"), obj, seed=0, image=b"x")
    assert row["passed"] and row["name"] == "open doorway" and row["attempts"] == 1


def test_label_route_cosmos_then_template_fallback():
    row = {"id": "f_L", "pair_id": None, "kind": "instructed", "events": [PASS_L], "stop": NEAR}
    good = "Please keep left of the coffee table and stop in front of the fridge."
    out = label_route(lambda p, s: (_reply({"description": good}), "stop"), row, NAMES, seed=0)
    assert out["passed"] and out["instruction_source"] == "cosmos" and out["description"] == good
    bad = "Go right of the coffee table and stop in front of the fridge."
    out = label_route(lambda p, s: (_reply({"description": bad}), "stop"), row, NAMES, seed=0)
    assert out["passed"] and out["instruction_source"] == "template" and "side:missing:0" in out["cosmos_rejects"]


def test_label_route_without_names_makes_no_call():
    row = {"id": "f_L", "pair_id": None, "kind": "instructed", "events": [PASS_L], "stop": NEAR}

    def boom(p, s):
        raise AssertionError("no call expected")
    out = label_route(boom, row, {"f:b5": "fridge"}, seed=0)
    assert out["passed"] is False and out["rejects"] == ["unnamed_object"] and out["missing"] == ["f:b1"]
```

Note: `label_object` takes an optional `image=` (bytes) so tests need no real frame; when `image` is None it renders `naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])`.

Append to `tests/test_labelling_io.py`:

```python
def test_chat_text_sends_a_plain_text_message():
    from types import SimpleNamespace
    from rover_vlm.labelling_io import chat_text
    seen = {}

    class Completions:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert chat_text(client, "m", "hello", max_tokens=5, seed=1) == ("hi", "stop")
    assert seen["messages"] == [{"role": "user", "content": "hello"}] and seen["seed"] == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_descriptions.py tests/test_labelling_io.py -v`
Expected: FAIL — `ImportError: cannot import name 'choose_shape'` / `'chat_text'`.

- [ ] **Step 3: Implement**

Append to `src/rover_vlm/labelling_io.py`:

```python
def chat_text(client, model, prompt, *, max_tokens, seed):
    """One text-only request -> (raw reply, finish_reason). Raises on transport errors."""
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, seed=seed,
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason
```

Append to `src/rover_vlm/descriptions.py` (add `import io` and PIL imports at the top; extend the instructions import with `STYLES, TARGET_OUTLINE_COLOR, _answer_block, _stable_index, choose_style, normalize_name`; add `from .traces import SCENE_INTRO, answer_payload, parse_trace`):

```python
PROMPT_VERSION = "d1"
KINDS = {"object": "object", "wall": "wall", "opening": "opening", "doorway": "opening", "door": "opening",
         "arch": "opening", "archway": "opening", "passage": "opening", "gap": "opening"}
SHAPES = ["one sentence", "two short sentences", "one sentence joined with 'then'",
          "a sentence that starts with the first thing to pass"]


def naming_image(image_path, polygon, crop_box, px=512):
    """One JPEG, two panels: the frame with the object outlined and the rest dimmed, and a
    close-up of the object's box (multi-image requests are untested on the server)."""
    img = Image.open(image_path).convert("RGB").resize((px, px), Image.LANCZOS)
    pts = [(u * px, v * px) for u, v in polygon]
    ctx = img.copy()
    if len(pts) >= 3:
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        ctx = Image.composite(img, Image.eval(img, lambda p: int(p * 0.45)), mask)
        ImageDraw.Draw(ctx).polygon(pts, outline=TARGET_OUTLINE_COLOR, width=4)
    x0, y0, x1, y1 = (c * px for c in (crop_box or [0, 0, 1, 1]))
    crop = img.crop((int(x0), int(y0), max(int(x1), int(x0) + 8), max(int(y1), int(y0) + 8)))
    crop.thumbnail((px, px))
    canvas = Image.new("RGB", (2 * px, px), (0, 0, 0))
    canvas.paste(ctx, (0, 0))
    canvas.paste(crop, (px + (px - crop.width) // 2, (px - crop.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def naming_prompt(need):
    what = ("The outlined region is an opening the rover drives through (a doorway, an arch, a gap "
            "between walls)." if need == "opening" else "The outlined region is something the rover drives past.")
    return f"""{SCENE_INTRO}

The picture has two panels. Left: the camera view with one region outlined and everything
else dimmed. Right: a close-up of that region. {what}

1. Name what is inside the outline in 1 to 4 words, the way a person would say it ("grey
   sofa", "stone column", "kitchen island", "open doorway"). Never use the words left or
   right, and never describe the picture itself (outline, panel, close-up, dimming).
2. Say what kind of thing it is: "object" (furniture, an appliance, a column), "wall" (part of
   a wall) or "opening" (a doorway, an arch, a gap between walls).

{_answer_block('{"name": "<name>", "kind": "object|wall|opening"}')}"""


def parse_name_reply(raw):
    """{"name", "kind"} from a naming reply, or None when there is no usable name."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is None or not isinstance(payload.get("name"), str) or not payload["name"].strip():
        return None
    kind = payload.get("kind")
    kind = KINDS.get(kind.strip().lower(), "object") if isinstance(kind, str) else "object"
    return {"name": normalize_name(" ".join(payload["name"].lower().split())), "kind": kind}


def choose_shape(route_id):
    return SHAPES[_stable_index("shape:" + route_id, len(SHAPES))]


def compose_prompt(events, stop, names, style, shape):
    steps = "\n".join(f"{i + 1}. {event_phrase(e, [names[o] for o in e['objects']])}" for i, e in enumerate(events))
    steps = steps or "1. go straight ahead"
    return f"""You are writing a route description for a small rover, the way a person standing behind
it would say it. The route, in order:
{steps}
Then: {stop_text(stop, names.get(stop['object']))}.

Write it as {style}, in {shape}. Keep every step in this order, keep each object's name and
the word left or right that goes with it, and keep where to stop. You may add small words
("then", "carefully", "keep going"), but add no other objects, sides or directions. Never say
"your left" or "your right". Write numbers as words. At most 45 words.

{_answer_block('{"description": "<the description>"}')}"""


def parse_compose_reply(raw):
    """The description string from a compose reply, or None (a bare sentence in <answer> is kept)."""
    _, answer = parse_trace(raw)
    payload = answer_payload(answer)
    if payload is not None:
        text = payload.get("description")
        return " ".join(text.split()) if isinstance(text, str) and text.strip() else None
    if answer and "{" not in answer and len(answer.split()) <= 60:
        return " ".join(answer.strip().strip('"').split()) or None
    return None


def label_object(ask, obj, seed, image=None):
    """Name one object: up to two Cosmos calls (seed, seed + 1). Never raises on a bad reply."""
    need = obj["kind_hint"]
    image = image if image is not None else naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])
    row = {"id": obj["object_id"], "need": need, "prompt_version": PROMPT_VERSION}
    rejects, raw, fin, reply = ["parse"], None, None, None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, fin = ask(image, naming_prompt(need), s)
        reply = parse_name_reply(raw)
        rejects = ["parse"] if reply is None else name_rejects(reply["name"], reply["kind"], need)
        if not rejects:
            return {**row, "name": reply["name"], "kind": reply["kind"], "passed": True, "rejects": [],
                    "attempts": attempt, "finish": fin}
    return {**row, "name": None, "kind": reply and reply["kind"], "passed": False, "rejects": rejects,
            "attempts": 2, "finish": fin, "raw": raw}


def route_objects(row):
    ids = [o for e in row["events"] for o in e["objects"]]
    return ids + ([row["stop"]["object"]] if row["stop"]["object"] else [])


def label_route(ask_text, row, names, seed):
    """Compose one route's description: two Cosmos attempts, then the template."""
    out = {"id": row["id"], "pair_id": row.get("pair_id"), "kind": row.get("kind"), "prompt_version": PROMPT_VERSION}
    missing = [o for o in route_objects(row) if not names.get(o)]
    if missing:
        return {**out, "passed": False, "description": None, "instruction_source": None,
                "rejects": ["unnamed_object"], "missing": missing}
    style, shape = choose_style(row["id"]), choose_shape(row["id"])
    prompt = compose_prompt(row["events"], row["stop"], names, style, shape)
    rejects, raw = [], None
    for attempt, s in enumerate((seed, seed + 1), 1):
        raw, _ = ask_text(prompt, s)
        text = parse_compose_reply(raw)
        rejects = description_rejects(text, row["events"], row["stop"], names)
        if not rejects:
            return {**out, "description": text, "instruction_source": "cosmos", "passed": True, "rejects": [],
                    "cosmos_rejects": [], "style": style, "shape": shape, "attempts": attempt}
    tpl = template_description(row["events"], row["stop"], names)
    trej = description_rejects(tpl, row["events"], row["stop"], names)
    return {**out, "description": tpl if not trej else None, "instruction_source": "template",
            "passed": not trej, "rejects": trej, "cosmos_rejects": rejects, "style": style, "shape": shape,
            "attempts": 2, "raw": raw}
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_descriptions.py tests/test_labelling_io.py -v` → all passed.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py src/rover_vlm/labelling_io.py tests/test_descriptions.py tests/test_labelling_io.py
git commit -m "feat: d1 naming and compose prompts, parsers, labelling logic

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `label_descriptions.py`, its sbatch and the label audit — GATE 2

**Files:**
- Create: `scripts/label_descriptions.py`
- Create: `slurm/label_descriptions.sbatch`
- Modify: `scripts/audit_instruction_labels.py` (mode `d1`)
- Test: `tests/test_descriptions.py` (`select_routes`, `summarize_d1`, `d1_gate_failures`)

**Interfaces:**
- Consumes: Task 6 (`label_object`, `label_route`, `route_objects`), `labelling_io` (`prepare_output`, `connect`, `resolve_endpoint`, `chat_with_image`, `chat_text`), Task 5 (`events_item`, `jsonl`, audit plumbing).
- Produces (in `descriptions.py`, so tests need no script import):
  - `select_routes(rows, limit=None) -> list` (described routes, first `limit`)
  - `summarize_d1(rows, stage) -> dict` (`n`, `errors`, `accept_rate`, `template_share` (compose), `rejects` counter)
  - `d1_gate_failures(summary, stage, min_accept=0.9, max_template=0.1) -> list[str]`
- Outputs: `outputs/instr_labels_v3/{names,desc}_{gate,eval,train}_d1.jsonl`; audit page `outputs/instr_d1_audit.html` (title "Route Description Audit").

- [ ] **Step 1: Write the failing tests** (append to `tests/test_descriptions.py`)

```python
from rover_vlm.descriptions import d1_gate_failures, select_routes, summarize_d1  # noqa: E402


def test_select_routes_and_gate():
    rows = [{"id": "a", "reason": None}, {"id": "b", "reason": "no_stop_anchor"}, {"id": "c", "reason": None}]
    assert [r["id"] for r in select_routes(rows)] == ["a", "c"]
    assert [r["id"] for r in select_routes(rows, limit=1)] == ["a"]
    comp = [{"id": str(i), "passed": True, "instruction_source": "cosmos", "rejects": []} for i in range(8)]
    comp += [{"id": "t", "passed": True, "instruction_source": "template", "rejects": []},
             {"id": "e", "error": "Timeout"}]
    s = summarize_d1(comp, "compose")
    assert s["n"] == 10 and s["errors"] == 1 and s["accept_rate"] == 1.0
    assert s["template_share"] == pytest.approx(1 / 9, abs=1e-3)
    assert d1_gate_failures(s, "compose") == ["template share 0.111 > 0.1"]
    names = [{"id": "o1", "passed": True, "rejects": []}, {"id": "o2", "passed": False, "rejects": ["name:long"]}]
    assert d1_gate_failures(summarize_d1(names, "name"), "name") == ["accept rate 0.500 < 0.9"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_descriptions.py::test_select_routes_and_gate -v`
Expected: FAIL — `ImportError: cannot import name 'd1_gate_failures'`.

- [ ] **Step 3: Implement the helpers** (append to `descriptions.py`, add `from collections import Counter`)

```python
def select_routes(rows, limit=None):
    ok = [r for r in rows if r.get("reason") is None]
    return ok[:limit] if limit else ok


def summarize_d1(rows, stage):
    ok = [r for r in rows if not r.get("error")]
    out = {"n": len(rows), "errors": len(rows) - len(ok),
           "accept_rate": round(sum(bool(r.get("passed")) for r in ok) / len(ok), 4) if ok else 0.0,
           "rejects": dict(Counter(x for r in ok for x in r.get("rejects", [])).most_common(15))}
    if stage == "compose":
        passed = [r for r in ok if r.get("passed")]
        out["template_share"] = round(sum(r.get("instruction_source") == "template" for r in passed)
                                      / len(passed), 4) if passed else 0.0
        out["cosmos_rejects"] = dict(Counter(x for r in ok for x in r.get("cosmos_rejects", [])).most_common(15))
    return out


def d1_gate_failures(summary, stage, min_accept=0.9, max_template=0.1):
    fails = []
    if summary["accept_rate"] < min_accept:
        fails.append(f"accept rate {summary['accept_rate']:.3f} < {min_accept}")
    if stage == "compose" and summary.get("template_share", 0) > max_template:
        fails.append(f"template share {summary['template_share']:.3f} > {max_template}")
    return fails
```

Run the test again → PASS.

- [ ] **Step 4: Write `scripts/label_descriptions.py`**

```python
"""Label route descriptions (prompt d1) with Cosmos3-Nano: name objects, then compose sentences.

Runs against the endpoint slurm/serve_cosmos3.sbatch (or slurm/label_descriptions.sbatch)
publishes. Inputs come from `prepare_habitat_instr.py events`.

    # stage 1: one call per frame object (both routes of a pair share it), on a composite image
    uv run scripts/label_descriptions.py --stage name \
        --events data/prepared_habitat_instr_v2/events_eval.jsonl \
        --objects data/prepared_habitat_instr_v2/objects_eval.jsonl \
        --out outputs/instr_labels_v3/names_eval_d1.jsonl --concurrency 8 --resume
    # stage 2: one text-only call per described route
    uv run scripts/label_descriptions.py --stage compose \
        --events data/prepared_habitat_instr_v2/events_eval.jsonl \
        --names outputs/instr_labels_v3/names_eval_d1.jsonl \
        --out outputs/instr_labels_v3/desc_eval_d1.jsonl --concurrency 8 --resume
    # gate: --limit 300 --gate on both stages (exit 2 when a threshold fails)

Each object/route gets up to two Cosmos attempts (seed, seed + 1) inside one row; compose then
falls back to the template sentence. Rows are flushed as they finish, so a killed run keeps
what it had; --resume retries error rows only.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rover_vlm.descriptions import (
    d1_gate_failures, label_object, label_route, route_objects, select_routes, summarize_d1,
)
from rover_vlm.labelling_io import chat_text, chat_with_image, connect, prepare_output, resolve_endpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_FILE = REPO_ROOT / "outputs" / "cosmos3_endpoint.txt"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["name", "compose"], required=True)
    p.add_argument("--events", type=Path, required=True, help="events_*.jsonl")
    p.add_argument("--objects", type=Path, help="objects_*.jsonl (stage name)")
    p.add_argument("--names", type=Path, help="names_*_d1.jsonl from stage name (stage compose)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="first N described routes (and their objects)")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gate", action="store_true", help="exit 2 unless accept >= 0.9 (and template share <= 0.1)")
    args = p.parse_args()

    routes = select_routes(jsonl(args.events), args.limit)
    if args.stage == "name":
        if not args.objects:
            raise SystemExit("--stage name needs --objects objects_*.jsonl")
        wanted = {o for r in routes for o in route_objects(r)}
        items = [o for o in jsonl(args.objects) if o["object_id"] in wanted]
        missing = wanted - {o["object_id"] for o in items}
        if missing:
            raise SystemExit(f"{len(missing)} route objects have no object row (e.g. {sorted(missing)[:3]})")
        key = "object_id"
    else:
        if not (args.names and args.names.exists()):
            raise SystemExit("--stage compose needs --names from a finished stage name run")
        names = {r["id"]: r["name"] for r in jsonl(args.names) if r.get("passed")}
        items, key = routes, "id"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, mode = prepare_output(args.out, resume=args.resume, overwrite=args.overwrite)
    todo = [it for it in items if it[key] not in done]
    print(f"stage {args.stage}: {len(items)} items, {len(todo)} to label, concurrency {args.concurrency}")

    if todo:
        base_url = resolve_endpoint(args.endpoint, ENDPOINT_FILE)
        client, model = connect(base_url, args.model)
        print(f"endpoint: {base_url}\nmodel:    {model}\n")

        def ask_image(image, prompt, seed):
            return chat_with_image(client, model, image, prompt, max_tokens=args.max_tokens, seed=seed)

        def ask_text(prompt, seed):
            return chat_text(client, model, prompt, max_tokens=args.max_tokens, seed=seed)

        def work(it):
            t0 = time.time()
            try:
                row = (label_object(ask_image, it, args.seed) if args.stage == "name"
                       else label_route(ask_text, it, names, args.seed))
            except Exception as e:  # noqa: BLE001 — one bad item must not kill the run
                return {"id": it[key], "error": f"{type(e).__name__}: {e}"}
            row["latency_s"] = round(time.time() - t0, 2)
            return row

        started = time.time()
        with args.out.open(mode) as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, it) for it in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                f.write(json.dumps(row) + "\n")
                f.flush()
                rate = i / max(1e-9, time.time() - started)
                status = (f"ERROR {row['error']}" if row.get("error") else "ok" if row["passed"]
                          else f"FAIL {row['rejects']}")
                shown = row.get("name") if args.stage == "name" else row.get("description")
                print(f"[{i}/{len(todo)}] {row['id']}  {rate * 60:.0f}/min  eta {(len(todo) - i) / rate / 60:.0f}m  "
                      f"{shown!r}  {status}", flush=True)

    rows = [r for r in jsonl(args.out) if r["id"] in {it[key] for it in items}]
    summary = summarize_d1(rows, args.stage)
    print("\nsummary:\n" + json.dumps(summary, indent=2))
    if summary["errors"]:
        print(f"{summary['errors']} request errors — re-run with --resume to retry them")
    if args.gate:
        fails = d1_gate_failures(summary, args.stage)
        print("GATE: " + ("PASS" if not fails else "FAIL — " + "; ".join(fails)))
        if fails:
            sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `slurm/label_descriptions.sbatch`** (modelled on `slurm/label_instructions.sbatch`, same serve library, mail and thread caps)

```bash
#!/bin/bash
# Label d1 route descriptions with Cosmos3-Nano, unattended: serve + both stages in one job.
#
#     sbatch slurm/label_descriptions.sbatch              # gate: 300 eval routes, then stop for the audit
#     STAGE=full sbatch slurm/label_descriptions.sbatch   # eval + train (resumes after a TIMEOUT)
#
# Needs data/prepared_habitat_instr_v2/{events,objects}_{eval,train}.jsonl
# (uv run scripts/prepare_habitat_instr.py events). Spec:
# docs/superpowers/specs/2026-09-30-route-descriptions-design.md. Safe to re-submit: every
# call passes --resume. Afterwards, on the login node:
#     uv run scripts/prepare_habitat_instr.py finalize-descriptions
#SBATCH --job-name=cosmos3-label-desc
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v3
source slurm/cosmos3_serve_lib.sh
D="${DATA:-data/prepared_habitat_instr_v2}"
OUT=outputs/instr_labels_v3
CONCURRENCY="${CONCURRENCY:-8}"
UV="${UV:-$HOME/.local/bin/uv}"
STAGE="${STAGE:-gate}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
for s in eval train; do
    for f in events objects; do
        [ -f "$D/${f}_$s.jsonl" ] || { echo "missing $D/${f}_$s.jsonl — run prepare_habitat_instr.py events"; exit 1; }
    done
done

cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy

label() {  # label <split> <tag> [extra args]
    local s=$1 tag=$2; shift 2
    echo; echo "=============== name: $tag ==============="
    "$UV" run scripts/label_descriptions.py --stage name --events "$D/events_$s.jsonl" \
        --objects "$D/objects_$s.jsonl" --out "$OUT/names_${tag}_d1.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume "$@"
    echo; echo "=============== compose: $tag ==============="
    "$UV" run scripts/label_descriptions.py --stage compose --events "$D/events_$s.jsonl" \
        --names "$OUT/names_${tag}_d1.jsonl" --out "$OUT/desc_${tag}_d1.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume "$@"
}

if [ "$STAGE" = "gate" ]; then
    # set -e: a failed gate (exit 2) ends the job; the EXIT trap stops the server
    label eval gate --limit 300 --gate
    echo "gate labelled — audit it, then: STAGE=full sbatch slurm/label_descriptions.sbatch"
    exit 0
fi
label eval eval
label train train
echo; echo "=============== done ==============="
wc -l "$OUT"/*_d1.jsonl
```

Add `"label_descriptions.sbatch"` to the `@pytest.mark.parametrize("name", [...])` list in `tests/test_slurm_thread_caps.py` (it enumerates files by name) and run `uv run pytest tests/test_slurm_thread_caps.py -v`.

- [ ] **Step 6: Add audit mode `d1`** in `scripts/audit_instruction_labels.py`

1. `VERDICTS["d1"]`:

```python
    "d1": [
        {"v": "ok", "label": "Correct", "help": "Names, sides and the stop are all right, and it reads naturally"},
        {"v": "name", "label": "Wrong name", "help": "An object is misnamed or is not the numbered thing"},
        {"v": "side", "label": "Wrong side", "help": "A side word does not match how the route passes"},
        {"v": "stop", "label": "Wrong stop", "help": "The stop phrase is wrong"},
        {"v": "unnatural", "label": "Unnatural", "help": "True, but nobody would say it like that"},
        {"v": "unsure", "label": "Can't tell", "help": "Too ambiguous from this frame"},
    ],
```

2. `--mode` choices gain `"d1"`; add `--desc` (`type=Path`). Branch (before `elif args.labels:`):

```python
    if args.mode == "d1":
        if not (args.events and args.objects and args.desc):
            raise SystemExit("--mode d1 needs --events, --objects and --desc desc_*_d1.jsonl")
        split = V2 / ("labelset_eval.json" if "eval" in args.events.name else "labelset_train.json")
        by_id = {r["id"]: r for r in json.loads(split.read_text())}
        ev = {r["id"]: r for r in jsonl(args.events)}
        objects = {o["object_id"]: o for o in jsonl(args.objects)}
        ok = [r for r in jsonl(args.desc) if r.get("passed")]
        rows = random.Random(args.seed).sample(ok, min(args.n, len(ok)))
        items = [events_item(by_id[r["id"]], ev[r["id"]], objects, text=r["description"]) for r in rows]
        for it, r in zip(items, rows):
            it["source"] = r["instruction_source"]
        pop = {"passed": len(ok), "template": sum(r["instruction_source"] == "template" for r in ok)}
        payload = {"items": items, "population": pop, "seed": args.seed, "mode": "d1",
                   "verdicts": VERDICTS["d1"], "source": str(args.desc)}
        html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload))
        html = html.replace("<title>Instruction Label Audit</title>", "<title>Route Description Audit</title>", 1)
        out = args.out or REPO_ROOT / "outputs" / "instr_d1_audit.html"
        out.write_text(html)
        print(f"{len(items)} items (population {pop}) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
```

(The numbered outlines stay on the page so a reviewer can check each name against its object.)

Docstring line:

```
--mode d1 (label gate): 100 passed descriptions from desc_*_d1.jsonl drawn with their numbered
objects. Gate >= 90% correct.
    uv run scripts/audit_instruction_labels.py --mode d1 \
        --events data/prepared_habitat_instr_v2/events_eval.jsonl \
        --objects data/prepared_habitat_instr_v2/objects_eval.jsonl \
        --desc outputs/instr_labels_v3/desc_gate_d1.jsonl
```

- [ ] **Step 7: Verify on CPU and commit**

```bash
uv run pytest -q
bash -n slurm/label_descriptions.sbatch
uv run scripts/label_descriptions.py --help
git add scripts/label_descriptions.py slurm/label_descriptions.sbatch scripts/audit_instruction_labels.py \
    src/rover_vlm/descriptions.py tests/test_descriptions.py tests/test_slurm_thread_caps.py
git commit -m "feat: d1 labelling script, sbatch and label audit page

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: GATE 2 (human + GPU).** Only after Gate 1 passed. Ask the user to run `sbatch slurm/label_descriptions.sbatch` (1 SLURM slot; check `squeue -u $USER` shows < 3 jobs) and paste the log. Read both summaries: name accept ≥ 0.9, compose accept ≥ 0.9, template share ≤ 0.1. If a machine gate fails, read the `raw` fields of 20 rejected rows, fix the prompt or parser test-first, and `--rescore`-style re-judge is **not** available here: re-run the gate with `--overwrite`. When the machine gate passes, build the page:

```bash
uv run scripts/audit_instruction_labels.py --mode d1 \
    --events data/prepared_habitat_instr_v2/events_eval.jsonl \
    --objects data/prepared_habitat_instr_v2/objects_eval.jsonl \
    --desc outputs/instr_labels_v3/desc_gate_d1.jsonl
```

publish it (db `marks`), and stop until the user has marked 100. Pass = ≥ 90% correct. Then the user runs `STAGE=full sbatch slurm/label_descriptions.sbatch` (re-submit after a 14 h TIMEOUT; it resumes). Record the gate numbers in memory.

---

### Task 8: The entry-only prompt

**Files:**
- Modify: `src/rover_vlm/habitat_data.py`
- Test: `tests/test_descriptions.py`

**Interfaces:**
- Produces: `HABITAT_PROMPT_DESCRIPTION`, `description_entry_prompt(start_wp, description) -> str`.

- [ ] **Step 1: Write the failing test** (append)

```python
from rover_vlm.habitat_data import description_entry_prompt, instruction_endpoints_prompt  # noqa: E402


def test_entry_prompt_has_the_entry_and_text_but_no_goal():
    p = description_entry_prompt([0.167, 1.0, 1], 'Go left of the "sofa" and stop by the fridge.')
    assert p.startswith("<image>\n") and "[0.167, 1.0]" in p and "where it is visible" in p
    assert "Go left of the 'sofa' and stop by the fridge." in p
    assert "goal" not in p.lower()
    q = instruction_endpoints_prompt([0.167, 1.0, 1], [0.5, 0.6, 1], "x")
    assert p.split("Coordinates are")[1].split("The first waypoint")[0] == q.split("Coordinates are")[1].split("The first waypoint")[0].replace(
        "Predict the traversable route between those two points", "Predict the route")
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_descriptions.py -k entry_prompt -v` → ImportError.

- [ ] **Step 3: Implement** (after `instruction_endpoints_prompt` in `habitat_data.py`)

```python
# Route descriptions (prompt d1, 2026-09-30): the entry point and a description that says
# where to stop; the goal point is NOT given. Wording kept parallel to the endpoints prompt.
HABITAT_PROMPT_DESCRIPTION = (
    "<image>\n"
    "You are a rover navigating an indoor environment. Its route enters the frame at "
    "[{sx}, {sy}], where it is {svis}. Drive it as this description says, and stop where it "
    'says to stop: "{desc}". Coordinates are normalized image coordinates in [0,1], x '
    "running right and y running down. Predict the route as a list of waypoints. Each "
    "waypoint is [x, y, v] where v is 1 if the point is on visible, unobstructed ground or "
    "0 if it is obstructed (hidden behind an obstacle). The first waypoint must be the entry "
    'point and the last must be where the rover stops. Answer as JSON: {{"path": [[x, y, v], ...]}}.'
)


def description_entry_prompt(start_wp, description):
    """HABITAT_PROMPT_DESCRIPTION for the entry waypoint and a route description (flattened,
    double quotes made single, exactly one trailing full stop — as instruction_endpoints_prompt)."""
    text = " ".join(str(description).split()).replace('"', "'").rstrip(".")
    return HABITAT_PROMPT_DESCRIPTION.format(sx=start_wp[0], sy=start_wp[1], svis=_vis_phrase(start_wp),
                                             desc=text + ".")
```

If the shared-wording assertion in the test fails, compare the two strings character by character and fix the new template (not the test) so only the intended phrases differ.

- [ ] **Step 4: Run** — `uv run pytest tests/test_descriptions.py -v` → passed.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/habitat_data.py tests/test_descriptions.py
git commit -m "feat: entry-only description prompt

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `finalize-descriptions` — train and eval files

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (finalize helpers)
- Modify: `scripts/prepare_habitat_instr.py` (stage, docstring)
- Test: `tests/test_descriptions.py`

**Interfaces:**
- Consumes: `description_entry_prompt` (Task 8), `instruction_endpoints_prompt`, `instructions._answer_path`, `_stable_index`, `assert_scene_disjoint`.
- Produces (in `descriptions.py`):
  - `ENTRY_SHARE = 70` (percent), `prompt_form(sample_id) -> "entry"|"goal"`
  - `desc_check(event_row, objects) -> dict` (`pose_xz, heading_xz, goal_xz, events: [{type, side, obstacles: [[[x,z],...], ...], gap_xz}]`)
  - `with_description(rec, text, source, form, check) -> rec` (deep copy; `habitat_meta` gains `instruction, instruction_source, prompt_form, desc_check, goal_in_prompt`)
  - `finalize_descriptions(label_recs, events_by_id, desc_by_id, objects) -> {"kept", "pairs", "stats"}`
  - `shuffled_texts(recs, seed=0) -> list[rec]` (entry form, each text from a record of another frame, `instruction_source="shuffled"`)
- Files: `data/prepared_habitat_instr_v3/{train_desc,eval_desc_entry,eval_desc_goal,eval_desc_pairs,eval_desc_shuffled}.json`, `meta_v3.json`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import copy  # noqa: E402
import json as _json  # noqa: E402

from rover_vlm.descriptions import (  # noqa: E402
    desc_check, finalize_descriptions, prompt_form, shuffled_texts, with_description,
)


def _rec(rid, sample, pair=None):
    path = [[0.2, 1.0, 1], [0.3, 0.8, 1], [0.5, 0.6, 1]]
    return {"id": rid, "image": ["x.png"],
            "conversations": [{"from": "human", "value": "old"}, {"from": "gpt", "value": _json.dumps({"path": path})}],
            "habitat_meta": {"sample_id": sample, "pair_id": pair, "scene_id": "s", "kind": "instructed"}}


def _ev(rid):
    return {"id": rid, "reason": None, "events": [PASS_L], "stop": NEAR, "goal_xz": [1.0, 2.0],
            "pose_xz": [0.0, 0.0], "heading_xz": [0.0, 1.0]}


OBJS = {"f:b1": {"cells_xz": [[0.1, 1.0], [0.2, 1.0]]}, "f:b5": {"cells_xz": [[1.0, 2.5]]}}


def test_prompt_form_is_stable_and_about_seventy_percent():
    forms = [prompt_form(f"frame{i}") for i in range(2000)]
    assert forms == [prompt_form(f"frame{i}") for i in range(2000)]
    assert 0.66 < forms.count("entry") / len(forms) < 0.74


def test_with_description_forms_and_check():
    chk = desc_check(_ev("a"), OBJS)
    assert chk["events"][0]["obstacles"] == [[[0.1, 1.0], [0.2, 1.0]]] and chk["goal_xz"] == [1.0, 2.0]
    rec = _rec("a", "f1")
    e = with_description(rec, "Go left of the table and stop.", "cosmos", "entry", chk)
    g = with_description(rec, "Go left of the table and stop.", "cosmos", "goal", chk)
    assert "goal" not in e["conversations"][0]["value"].lower() and "[0.5, 0.6]" in g["conversations"][0]["value"]
    assert e["habitat_meta"]["goal_in_prompt"] is False and g["habitat_meta"]["goal_in_prompt"] is True
    assert rec["conversations"][0]["value"] == "old"                      # not mutated


def test_finalize_keeps_survivors_and_only_full_pairs():
    recs = [_rec("p_L", "f1", "p"), _rec("p_R", "f1", "p"), _rec("q_L", "f2", "q"), _rec("q_R", "f2", "q"),
            _rec("s_L", "f3")]
    events = {r["id"]: _ev(r["id"]) for r in recs}
    events["q_R"] = dict(events["q_R"], reason="no_stop_anchor")
    desc = {r["id"]: {"id": r["id"], "passed": True, "description": "Go left of the table and stop.",
                      "instruction_source": "cosmos"} for r in recs}
    out = finalize_descriptions(recs, events, desc, OBJS)
    assert [r["id"] for r in out["kept"]] == ["p_L", "p_R", "q_L", "s_L"]
    assert [r["id"] for r in out["pairs"]] == ["p_L", "p_R"]
    forms = {r["id"]: r["habitat_meta"]["prompt_form"] for r in out["kept"]}
    assert forms["p_L"] == forms["p_R"]                                   # a pair shares its form
    assert out["stats"]["dropped_geometry"] == 1 and out["stats"]["pairs_kept"] == 1


def test_shuffled_texts_come_from_other_frames():
    recs = []
    for i in range(6):
        r = with_description(_rec(f"r{i}", f"f{i // 2}"), f"text {i}", "cosmos", "entry", desc_check(_ev("x"), OBJS))
        recs.append(r)
    out = shuffled_texts(recs)
    by = {r["id"]: r for r in recs}
    for r in out:
        src = next(o for o in recs if o["habitat_meta"]["instruction"] == r["habitat_meta"]["instruction"])
        assert src["habitat_meta"]["sample_id"] != by[r["id"]]["habitat_meta"]["sample_id"]
        assert r["habitat_meta"]["instruction_source"] == "shuffled"
        assert r["habitat_meta"]["desc_check"] == by[r["id"]]["habitat_meta"]["desc_check"]
```

- [ ] **Step 2: Run to verify they fail** — ImportError on `desc_check`.

- [ ] **Step 3: Implement** (append to `descriptions.py`; add `import copy, random` and imports `from .habitat_data import description_entry_prompt, instruction_endpoints_prompt` and `from .instructions import _answer_path`)

```python
ENTRY_SHARE = 70   # percent of frames whose records get the entry-only prompt (user, 2026-09-30)


def prompt_form(sample_id):
    """'entry' (no goal in the prompt) or 'goal', fixed per frame so a pair shares it."""
    return "entry" if _stable_index("form:" + sample_id, 100) < ENTRY_SHARE else "goal"


def desc_check(ev, objects):
    """What eval.description_follow needs, carried in the record (no file access at eval time)."""
    return {"pose_xz": ev["pose_xz"], "heading_xz": ev["heading_xz"], "goal_xz": ev["goal_xz"],
            "events": [{"type": e["type"], "side": e["side"],
                        "obstacles": [objects[o]["cells_xz"] for o in e["objects"]],
                        "gap_xz": e.get("gap_xz")} for e in ev["events"]]}


def with_description(rec, text, source, form, check):
    new = copy.deepcopy(rec)
    path = _answer_path(new)
    new["conversations"][0]["value"] = (description_entry_prompt(path[0], text) if form == "entry"
                                        else instruction_endpoints_prompt(path[0], path[-1], text))
    new["habitat_meta"].update({"instruction": text, "instruction_source": source, "prompt_form": form,
                                "desc_check": check, "goal_in_prompt": form == "goal"})
    return new


def finalize_descriptions(label_recs, events_by_id, desc_by_id, objects):
    kept, geo, unlabelled, rejected = [], 0, 0, 0
    for rec in label_recs:
        ev, row = events_by_id.get(rec["id"]), desc_by_id.get(rec["id"])
        if not ev or ev.get("reason") is not None:
            geo += 1
            continue
        if row is None:
            unlabelled += 1
            continue
        if not row.get("passed"):
            rejected += 1
            continue
        form = prompt_form(rec["habitat_meta"]["sample_id"])
        kept.append(with_description(rec, row["description"], row["instruction_source"], form, desc_check(ev, objects)))
    groups = {}
    for r in kept:
        if r["habitat_meta"].get("pair_id"):
            groups.setdefault(r["habitat_meta"]["pair_id"], []).append(r)
    pairs = [r for g in groups.values() if len(g) == 2 for r in g]
    return {"kept": kept, "pairs": pairs,
            "stats": {"records": len(label_recs), "kept": len(kept), "dropped_geometry": geo,
                      "unlabelled": unlabelled, "rejected": rejected, "pairs_kept": len(pairs) // 2,
                      "entry_form": sum(r["habitat_meta"]["prompt_form"] == "entry" for r in kept),
                      "template": sum(r["habitat_meta"]["instruction_source"] == "template" for r in kept)}}


def as_form(rec, form):
    hm = rec["habitat_meta"]
    return with_description(rec, hm["instruction"], hm["instruction_source"], form, hm["desc_check"])


def shuffled_texts(recs, seed=0):
    """Entry-form copies whose description comes from a record of another frame (the check
    and the answer stay the record's own): a text-reading model should get worse here."""
    order = list(recs)
    random.Random(seed).shuffle(order)
    n = len(order)
    for k in range(1, n):
        src = [order[(i + k) % n] for i in range(n)]
        if all(a["habitat_meta"]["sample_id"] != b["habitat_meta"]["sample_id"] for a, b in zip(order, src)):
            break
    else:
        raise ValueError("cannot shuffle: every record shares one frame")
    return [with_description(a, b["habitat_meta"]["instruction"], "shuffled", "entry", a["habitat_meta"]["desc_check"])
            for a, b in zip(order, src)]
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_descriptions.py -v` → passed.

- [ ] **Step 5: The stage** in `scripts/prepare_habitat_instr.py`

Docstring addition:

```
    # after slurm/label_descriptions.sbatch (STAGE=full)
    uv run scripts/prepare_habitat_instr.py finalize-descriptions
```

Code (imports `from rover_vlm.descriptions import as_form, finalize_descriptions, shuffled_texts`; constants `V3_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v3"`, `D1_LABELS = REPO_ROOT / "outputs" / "instr_labels_v3"`):

```python
def stage_finalize_descriptions(args):
    import random
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta, splits = {}, {}
    for name in SPLITS:
        recs = json.loads((args.events_dir / f"labelset_{name}.json").read_text())
        events = {r["id"]: r for r in map(json.loads, (args.events_dir / f"events_{name}.jsonl").open())}
        objects = {o["object_id"]: o for o in map(json.loads, (args.events_dir / f"objects_{name}.jsonl").open())}
        desc = _label_rows([args.labels / f"desc_{name}_d1.jsonl"])
        out = finalize_descriptions(recs, events, desc, objects)
        if out["stats"]["unlabelled"] > 0.01 * max(1, out["stats"]["records"] - out["stats"]["dropped_geometry"]) \
                and not args.allow_missing:
            raise SystemExit(f"{name}: {out['stats']['unlabelled']} described routes have no label row (> 1%) — "
                             "finish slurm/label_descriptions.sbatch or pass --allow-missing")
        splits[name] = out
        meta[name] = out["stats"]
        print(f"{name}: {json.dumps(out['stats'])}", flush=True)
    assert_scene_disjoint(splits["train"]["kept"], splits["eval"]["kept"])
    d = args.out_dir
    train = splits["train"]["kept"]
    random.Random(V2_SEED).shuffle(train)
    (d / "train_desc.json").write_text(json.dumps(train, indent=1))
    ev = splits["eval"]
    files = {"eval_desc_entry": [as_form(r, "entry") for r in ev["kept"]],
             "eval_desc_goal": [as_form(r, "goal") for r in ev["kept"]],
             "eval_desc_pairs": [as_form(r, "entry") for r in ev["pairs"]],
             "eval_desc_shuffled": shuffled_texts([as_form(r, "entry") for r in ev["kept"]], seed=V2_SEED)}
    for fname, rs in files.items():
        (d / f"{fname}.json").write_text(json.dumps(rs, indent=1))
        meta["eval"][fname] = len(rs)
    meta["train"]["train_desc"] = len(train)
    (d / "meta_v3.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
```

Subparser:

```python
    k = sub.add_parser("finalize-descriptions", help="d1: train/eval files from labelled descriptions (CPU)")
    k.add_argument("--events-dir", type=Path, default=V2_OUT)
    k.add_argument("--labels", type=Path, default=D1_LABELS)
    k.add_argument("--out-dir", type=Path, default=V3_OUT)
    k.add_argument("--allow-missing", action="store_true")
```

and `"finalize-descriptions": stage_finalize_descriptions` in the dispatch dict. (`V2_SEED` and `_label_rows` already exist in the script.)

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q
uv run scripts/prepare_habitat_instr.py finalize-descriptions --help
git add src/rover_vlm/descriptions.py scripts/prepare_habitat_instr.py tests/test_descriptions.py
git commit -m "feat: finalize-descriptions — 70/30 entry/goal forms, four eval sets

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Run the stage itself only after the full labelling job has finished (Task 7 Step 8).

---

### Task 10: "Description followed" metric

**Files:**
- Modify: `src/rover_vlm/eval.py`
- Modify: `scripts/evaluate.py` (after checking `git diff scripts/evaluate.py` — see Global Constraints)
- Test: `tests/test_description_eval.py`

**Interfaces:**
- Consumes: `landmarks.Camera`, `landmarks.CAM_HEIGHT_M`, `landmarks.densify`; records from `evaluate.py` (`parsed.path`, `gt.goal`, `metrics.goal_point_error`, `habitat_meta.desc_check`, `habitat_meta.pair_id`).
- Produces: `image_to_floor(path, pose_xz, heading_xz) -> np.ndarray (k, 2)`, `description_follow(pred_path, check) -> {"follows", "events", "stop"}`, `aggregate_description_metrics(records) -> dict | None`; `summary["description"]` in `metrics.json`. Constants `FOLLOW_NEAR_M = 2.0`, `STOP_TOL_M = 0.75`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_description_eval.py`:

```python
"""Description-followed metric: predicted image paths mapped back onto the frame's floor."""
import numpy as np
import pytest

from rover_vlm.eval import aggregate_description_metrics, description_follow, image_to_floor
from rover_vlm.landmarks import Camera

CAM = Camera([0.0, 0.0], [0.0, 1.0])          # at the origin facing +z; image-left is +x


def to_image(xz):
    uv, _ = CAM.uv(np.asarray(xz, float))
    return [[float(u), float(v), 1] for u, v in uv]


def check(events, goal=(0.0, 4.0)):
    return {"pose_xz": [0.0, 0.0], "heading_xz": [0.0, 1.0], "goal_xz": list(goal), "events": events}


LEFT_ROUTE = [(0.0, 0.8), (0.8, 2.0), (0.0, 4.0)]      # bows to +x = the rover's left
RIGHT_ROUTE = [(0.0, 0.8), (-0.8, 2.0), (0.0, 4.0)]
POST = [[[0.0, 2.0], [0.05, 2.0]]]


def test_image_to_floor_inverts_the_camera():
    xz = image_to_floor(to_image([(0.5, 2.0), (-1.0, 3.0)]), [0.0, 0.0], [0.0, 1.0])
    assert np.allclose(xz, [[0.5, 2.0], [-1.0, 3.0]], atol=1e-6)


def test_points_above_the_horizon_are_dropped():
    assert len(image_to_floor([[0.5, 0.4, 1], [0.5, 0.5, 1]], [0, 0], [0, 1])) == 0


def test_pass_left_followed_and_mirror_not():
    c = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    assert description_follow(to_image(LEFT_ROUTE), c) == {"follows": 1, "events": [1], "stop": 1}
    assert description_follow(to_image(RIGHT_ROUTE), c)["follows"] == 0


def test_between_needs_opposite_sides():
    c = check([{"type": "between", "side": None,
                "obstacles": [[[1.0, 2.0]], [[-1.0, 2.0]]], "gap_xz": None}])
    assert description_follow(to_image([(0.0, 0.8), (0.0, 2.0), (0.0, 4.0)]), c)["events"] == [1]
    assert description_follow(to_image([(0.0, 0.8), (1.6, 2.0), (0.0, 4.0)]), c)["events"] == [0]


def test_stop_tolerance():
    c = check([], goal=(0.0, 4.0))
    assert description_follow(to_image([(0.0, 0.8), (0.0, 4.5)]), c)["stop"] == 1
    assert description_follow(to_image([(0.0, 0.8), (0.0, 5.0)]), c)["stop"] == 0


def test_degenerate_predictions_count_as_not_followed():
    c = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    assert description_follow([[0.5, 0.3, 1]], c)["follows"] == 0
    recs = [{"id": "a", "parsed": None, "gt": {"goal": [0.5, 0.6, 1]}, "metrics": None,
             "habitat_meta": {"desc_check": c, "pair_id": None}}]
    agg = aggregate_description_metrics(recs)
    assert agg["n_records"] == 1 and agg["follow_rate"] == 0.0


def test_aggregate_pairs_and_constant_goal_baseline():
    c_l = check([{"type": "pass", "side": "left", "obstacles": POST, "gap_xz": None}])
    c_r = check([{"type": "pass", "side": "right", "obstacles": POST, "gap_xz": None}])
    rows = []
    for rid, c, pred, goal in (("p_L", c_l, LEFT_ROUTE, [0.5, 0.6]), ("p_R", c_r, LEFT_ROUTE, [0.4, 0.6])):
        rows.append({"id": rid, "parsed": {"path": to_image(pred)}, "gt": {"goal": goal + [1]},
                     "metrics": {"goal_point_error": 0.05}, "habitat_meta": {"desc_check": c, "pair_id": "p"}})
    agg = aggregate_description_metrics(rows)
    assert agg["follow_rate"] == 0.5 and agg["n_pairs"] == 1 and agg["pair_success"] == 0.0
    assert agg["constant_goal_endpoint_error_median"] == pytest.approx(0.05)
    assert agg["endpoint_error_median"] == pytest.approx(0.05)
    assert agg["event_rate_by_type"] == {"pass": 0.5}
    assert aggregate_description_metrics([{"id": "x", "habitat_meta": {}}]) is None
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_description_eval.py -v` → ImportError.

- [ ] **Step 3: Implement** (append to `src/rover_vlm/eval.py`; add `from .landmarks import CAM_HEIGHT_M, Camera, densify` at the top — `landmarks` imports nothing from `eval`, so there is no cycle)

```python
# --- route descriptions (prompt d1, 2026-09-30) ---------------------------------------
FOLLOW_NEAR_M = 2.0    # a prediction farther than this from an obstacle did not "pass" it
STOP_TOL_M = 0.75      # the predicted end within this of the true goal = stopped right
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


def description_follow(pred_path, check):
    """Did a predicted path do what its description says? Per event and the stop."""
    floor = image_to_floor(pred_path or [], check["pose_xz"], check["heading_xz"])
    n = len(check["events"])
    if len(floor) < 2:
        return {"follows": 0, "events": [0] * n, "stop": 0}
    res = []
    for e in check["events"]:
        if e["type"] == "pass":
            side, dist = _pass_side(floor, e["obstacles"][0])
            res.append(int(side == e["side"] and dist <= FOLLOW_NEAR_M))
        else:
            groups = e["obstacles"] if e["type"] == "between" else [[p] for p in e["gap_xz"]]
            a, b = (_pass_side(floor, g) for g in groups[:2])
            res.append(int(a[0] != b[0] and max(a[1], b[1]) <= FOLLOW_NEAR_M))
    stop = int(np.linalg.norm(floor[-1] - np.asarray(check["goal_xz"], float)) <= STOP_TOL_M)
    return {"follows": int(all(res) and stop), "events": res, "stop": stop}


def aggregate_description_metrics(records):
    """Summary for records carrying habitat_meta.desc_check, else None. Unparseable
    predictions count as not followed and stay in every denominator."""
    rows = [r for r in records if (r.get("habitat_meta") or {}).get("desc_check")]
    if not rows:
        return None
    follows, stops, by_type, pairs = {}, [], {}, {}
    for r in rows:
        chk = r["habitat_meta"]["desc_check"]
        path = (r.get("parsed") or {}).get("path")
        f = description_follow(path, chk) if path else {"follows": 0, "events": [0] * len(chk["events"]), "stop": 0}
        follows[r["id"]] = f["follows"]
        stops.append(f["stop"])
        for e, ok in zip(chk["events"], f["events"]):
            by_type.setdefault(e["type"], []).append(ok)
        if r["habitat_meta"].get("pair_id"):
            pairs.setdefault(r["habitat_meta"]["pair_id"], []).append(r["id"])
    full = [p for p in pairs.values() if len(p) == 2]
    goals = np.array([r["gt"]["goal"][:2] for r in rows if r.get("gt")], float)
    ep = [r["metrics"]["goal_point_error"] for r in rows if r.get("metrics")]
    return {
        "n_records": len(rows),
        "follow_rate": float(np.mean(list(follows.values()))),
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
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_description_eval.py -v` → passed; `uv run pytest -q` → green.

- [ ] **Step 5: Hook into `scripts/evaluate.py`.** First run `git diff scripts/evaluate.py`. If it shows the user's unrelated uncommitted hunk, **stop and ask the user** to commit it (it is open item 5 in NOTES.md) before editing. Then add `aggregate_description_metrics` to the `from rover_vlm.eval import (...)` list, and after the `wrong_side` block:

```python
        desc = aggregate_description_metrics(results)
        if desc is not None:
            summary["description"] = desc
```

and in the module docstring, after the instruction-set sentence:

```
Route-description sets (data/prepared_habitat_instr_v3/eval_desc_*.json) get a "description"
block: follow rate (every event and the stop), per-event-type rates, pair success, endpoint
error next to the constant-goal baseline.
```

- [ ] **Step 6: Commit**

```bash
git add src/rover_vlm/eval.py scripts/evaluate.py tests/test_description_eval.py
git commit -m "feat: description-followed metric with constant-goal baseline

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Eval sbatch and path videos

**Files:**
- Create: `slurm/eval_descriptions.sbatch`
- Create: `scripts/render_desc_videos.py`
- Test: `tests/test_render_desc_videos.py`

**Interfaces:**
- Consumes: `evaluate.py` outputs `<out>/<set>/<tag>/predictions.json` (rows with `id, parsed, gt, habitat_meta`), `overlay.draw_polyline`, `overlay.GT_GREY`.
- Produces: `pick_frames(rows, n) -> list[list[row]]` (pairs grouped, pairs first), `panel(image_path, rows, px=448) -> PIL.Image` (one tile per row, text under each), mp4 at `outputs/eval_desc/<set>/<tag>/paths.mp4`.

- [ ] **Step 1: Write the failing tests**

```python
"""render_desc_videos: frame picking and panel layout (ffmpeg itself is not exercised)."""
import importlib.util
from pathlib import Path

from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "render_desc_videos", Path(__file__).resolve().parent.parent / "scripts" / "render_desc_videos.py")
rdv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rdv)


def _row(rid, pair, img):
    return {"id": rid, "parsed": {"path": [[0.2, 1.0, 1], [0.5, 0.6, 1]]}, "gt": {"path": [[0.2, 1.0, 1], [0.4, 0.6, 1]]},
            "habitat_meta": {"pair_id": pair, "instruction": "Go left of the sofa and stop.", "image": str(img)}}


def test_pick_frames_groups_pairs_first(tmp_path):
    rows = [_row("s", None, "x"), _row("p_L", "p", "x"), _row("p_R", "p", "x"), _row("t", None, "x")]
    frames = rdv.pick_frames(rows, 2)
    assert [[r["id"] for r in f] for f in frames] == [["p_L", "p_R"], ["s"]]


def test_panel_has_one_tile_per_row(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (512, 512), (90, 90, 90)).save(p)
    one = rdv.panel(p, [_row("s", None, p)], px=200)
    two = rdv.panel(p, [_row("a", "p", p), _row("b", "p", p)], px=200)
    assert two.width == 2 * one.width and one.height == two.height and one.height > 200
```

- [ ] **Step 2: Run to verify they fail** — file not found.

- [ ] **Step 3: Implement `scripts/render_desc_videos.py`**

```python
"""Render predicted vs ground-truth paths on d1 eval frames as one mp4 per set and tag.

CPU only, login node, after slurm/eval_descriptions.sbatch. Each video frame shows one image
(both routes of a pair side by side): ground truth thick grey, prediction orange, the
description under each tile; every frame is held HOLD_S seconds.

    uv run scripts/render_desc_videos.py --set eval_desc_pairs --tag d1
    uv run scripts/render_desc_videos.py --set eval_desc_entry --tag d1 --n 40

Writes outputs/eval_desc/<set>/<tag>/paths.mp4.
"""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from rover_vlm.overlay import GT_GREY, draw_polyline

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED = (255, 140, 0)
HOLD_S = 3
FPS = 2
TEXT_H = 90


def pick_frames(rows, n):
    """Up to n frames: pairs first (both routes together), then single routes, input order."""
    pairs, singles, seen = {}, [], []
    for r in rows:
        pid = (r.get("habitat_meta") or {}).get("pair_id")
        if pid:
            if pid not in pairs:
                seen.append(pid)
            pairs.setdefault(pid, []).append(r)
        else:
            singles.append([r])
    frames = [pairs[p] for p in seen if len(pairs[p]) == 2] + singles
    return frames[:n]


def _tile(image_path, row, px, font):
    img = Image.open(image_path).convert("RGB").resize((px, px), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    draw_polyline(d, [(p[0], p[1]) for p in row["gt"]["path"]], GT_GREY, px, px, width=8)
    if (row.get("parsed") or {}).get("path"):
        draw_polyline(d, [(p[0], p[1]) for p in row["parsed"]["path"]], PRED, px, px, width=4)
    tile = Image.new("RGB", (px, px + TEXT_H), (20, 20, 20))
    tile.paste(img, (0, 0))
    text = "\n".join(textwrap.wrap(row["habitat_meta"].get("instruction") or "", width=max(20, px // 8))[:5])
    ImageDraw.Draw(tile).multiline_text((6, px + 4), text, fill=(235, 235, 235), font=font, spacing=2)
    return tile


def panel(image_path, rows, px=448):
    font = ImageFont.load_default()
    tiles = [_tile(image_path, r, px, font) for r in rows]
    out = Image.new("RGB", (px * len(tiles), tiles[0].height), (0, 0, 0))
    for i, t in enumerate(tiles):
        out.paste(t, (i * px, 0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "outputs" / "eval_desc")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "prepared_habitat_instr_v3")
    args = p.parse_args()
    run = args.eval_dir / args.set / args.tag
    rows = json.loads((run / "predictions.json").read_text())
    images = {r["id"]: r["image"][0] for r in json.loads((args.data_dir / f"{args.set}.json").read_text())}
    frames = pick_frames(rows, args.n)
    if not frames:
        raise SystemExit(f"no predictions in {run}")
    width = max(len(f) for f in frames) * 448
    height = 448 + TEXT_H
    out = run / "paths.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        img = panel(images[f[0]["id"]], f)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(img, (0, 0))
        for _ in range(HOLD_S * FPS):
            proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
```

(The test builds rows with `habitat_meta.image`; `panel` takes the image path explicitly, so the tests pass the path directly — keep `panel(image_path, rows, px)` as the signature.)

- [ ] **Step 4: Write `slurm/eval_descriptions.sbatch`** (same shape as `slurm/eval_instruction_v2.sbatch`)

```bash
#!/bin/bash
# d1 eval cells, skipping finished ones (safe to re-submit):
#     sbatch slurm/eval_descriptions.sbatch                         # adapter outputs/runs_desc/d1/adapter
#     ADAPTER=<dir> TAG=<tag> sbatch slurm/eval_descriptions.sbatch
# metrics.json gains a "description" block. Then, on the login node:
#     uv run scripts/render_desc_videos.py --set eval_desc_pairs --tag d1
#SBATCH --job-name=qwen-desc-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
D=data/prepared_habitat_instr_v3
OUT=outputs/eval_desc
TAG="${TAG:-d1}"
ADAPTER="${ADAPTER:-outputs/runs_desc/d1/adapter}"
run() {  # run <set>
    local set=$1
    if [ -f "$OUT/$set/$TAG/metrics.json" ]; then echo "skip $TAG/$set (done)"; return; fi
    if [ ! -f "$D/$set.json" ]; then echo "SKIP $TAG/$set: no $D/$set.json"; return; fi
    if [ ! -d "$ADAPTER" ]; then echo "SKIP $TAG/$set: no adapter at $ADAPTER"; return; fi
    echo "=== $TAG on $set ==="
    $UV run scripts/evaluate.py --task habitat --tag "$TAG" --eval-file "$D/$set.json" \
        --out-dir "$OUT/$set" --adapter "$ADAPTER"
}
for s in eval_desc_entry eval_desc_goal eval_desc_pairs eval_desc_shuffled; do run "$s"; done
echo "all cells done: uv run scripts/render_desc_videos.py --set eval_desc_pairs --tag $TAG"
```

- [ ] **Step 5: Verify and commit**

Add `"eval_descriptions.sbatch"` to the parametrize list in `tests/test_slurm_thread_caps.py`, then:

```bash
uv run pytest tests/test_render_desc_videos.py tests/test_slurm_thread_caps.py -v
bash -n slurm/eval_descriptions.sbatch
git add scripts/render_desc_videos.py slurm/eval_descriptions.sbatch tests/test_render_desc_videos.py tests/test_slurm_thread_caps.py
git commit -m "feat: d1 eval sbatch and path videos

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: README and handoff notes

**Files:**
- Modify: `README.md` (new section "Route descriptions (prompt d1)" after "Grounded landmarks")
- Modify: `NOTES.md` (handoff, ≤ 40 lines — the user's file; rewrite only its "where we are" and "next" parts)

- [ ] **Step 1:** `git diff README.md` — if the user's unrelated framing-row hunk is still uncommitted, ask before editing (Global Constraints).
- [ ] **Step 2:** Add the README section: why (the 2026-09-30 review in four groups, link to the spec), the pipeline commands in order (events → geometry audit → `sbatch slurm/label_descriptions.sbatch` → label audit → `STAGE=full` → `finalize-descriptions` → train with `slurm/train_habitat.sbatch data/prepared_habitat_instr_v3/train_desc.json …` → `sbatch slurm/eval_descriptions.sbatch` → `render_desc_videos.py`), where outputs land (`data/prepared_habitat_instr_v2/{events,objects}_*.jsonl`, `outputs/instr_labels_v3/`, `data/prepared_habitat_instr_v3/`, `outputs/eval_desc/`), and how to read the metrics (follow rate, pair success, endpoint error vs the constant-goal baseline, the shuffled control).
- [ ] **Step 3:** Update `NOTES.md` with the current gate state and next step.
- [ ] **Step 4:** `uv run pytest -q` → green. Commit README only (`git add README.md`; NOTES.md is untracked and stays so unless the user asks).

```bash
git add README.md
git commit -m "docs: README route descriptions (prompt d1)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (plan author)

- Spec coverage: geometry (Tasks 1–3), group B depth trim (Task 2), group C near-camera rule (Task 1), text rules/templates (Task 4), geometry gate (Task 5), both Cosmos stages + composite image + chat_text (Task 6), labelling script/sbatch/label gate (Task 7), entry prompt (Task 8), 70/30 finalize + four eval sets + shuffled control (Task 9), follow metric + pair success + constant-goal baseline (Task 10), videos + eval sbatch (Task 11), README (Task 12). Out-of-scope items (training runs, new rendering, HM3D checks, group-D drop) have no task, by design.
- Synthetic fixtures were checked by hand against the thresholds (pass ≤ 0.7 m, outline ≥ 1% of the image, stop radii); if a fixture test fails, re-check the fixture geometry before touching a constant.
