# Habitat Rover Path + Visibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen3.5-2B (LoRA) on the Habitat dataset to predict the traversable path to a goal as `[x, y, v]` waypoints plus a goal `[x, y, v]`, where `v` marks visible/obstructed, and evaluate base-vs-LoRA across training-set sizes with visibility metrics.

**Architecture:** A new pure-function module (`habitat_data.py`) converts each Habitat sample dir into a conversation record (select correct path → normalize → clip-to-frame → resample-preserving-transitions → JSON answer). A prepare script writes nested train/eval splits. Training reuses the existing stack unchanged (absolute image paths make `image_root` a no-op). Evaluation gains a `habitat` task mode with a new answer parser and position + visibility metrics.

**Tech Stack:** Python 3.12, uv, numpy, Pillow, matplotlib, pytest (new dev dep). transformers/peft for the (unchanged) training/eval GPU scripts.

## Global Constraints

- Environment: **uv** (managed Python 3.12). Run everything via `~/.local/bin/uv run ...`. Tests: `uv run pytest`.
- No local GPU. Pure-function code + data prep + tests run on the login node; training/eval run on SLURM (user launches). Set `HF_HUB_OFFLINE=1` for GPU jobs.
- Dataset root (NFS, read-only, shared with thor): `/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/dataset`. **Never copy images into the repo.**
- Prepared data → gitignored `data/prepared_habitat/`. Outputs → gitignored `outputs/`.
- Coordinates normalized to [0,1] by `fpv_paths["image_size"]` (512), 3 decimals. Visibility `v`: **1 = visible, 0 = obstructed** (i.e. `v = 0 if run/goal "hidden" else 1`).
- Prepared records store the **absolute** path to `fpv_enhanced.png` in `image`, so `TrajectoryDataset`/`evaluate.py` resolve it regardless of `image_root`.
- Keep the ShareRobot code paths working; add Habitat alongside, don't replace.
- Commit after every task (this repo already has history; commit to `main`).

---

## File Structure

- Create `src/rover_vlm/habitat_data.py` — load sample, select correct path, normalize, clip, resample+transitions, format answer/prompt, build record. Pure functions (no argparse).
- Create `scripts/prepare_habitat.py` — CLI: dataset root → `data/prepared_habitat/{eval,train_*,meta}.json`.
- Create `scripts/inspect_habitat.py` — overlay visibility-colored path + goal on `fpv_enhanced.png`.
- Create `tests/test_habitat_data.py`, `tests/test_habitat_eval.py`.
- Modify `src/rover_vlm/eval.py` — add `parse_path_answer`, habitat position+visibility metrics, `aggregate_habitat_metrics`.
- Modify `scripts/evaluate.py` — add `--task {sharerobot,habitat}` selecting prompt-gt parsing + metrics.
- Modify `scripts/compare_evals.py` — visibility columns + habitat tag/meta support.
- Create `slurm/train_habitat.sbatch`, `slurm/eval_habitat.sbatch` (thin wrappers over the existing scripts).
- (Deferred) `scripts/visualize_predictions.py` habitat adaptation — not in this plan; `inspect_habitat.py` covers qualitative needs.
- Modify `README.md`, `pyproject.toml` (add pytest dev dep).

---

## Task 1: Test harness + pytest

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Add pytest as a dev dependency**

Edit `pyproject.toml` — after the `[build-system]` block add:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Sync**

Run: `~/.local/bin/uv sync`
Expected: installs pytest; no errors.

- [ ] **Step 3: Write a smoke test**

Create `tests/test_smoke.py`:

```python
def test_package_imports():
    import rover_vlm  # noqa: F401
```

- [ ] **Step 4: Run it**

Run: `~/.local/bin/uv run pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/test_smoke.py
git commit -m "test: add pytest harness"
```

---

## Task 2: Normalize + clip-to-frame geometry

**Files:**
- Create: `src/rover_vlm/habitat_data.py`
- Create: `tests/test_habitat_data.py`

**Interfaces:**
- Produces:
  - `normalize_points(raw: list[tuple[float,float,bool]], w: float, h: float) -> list[tuple[float,float,bool]]`
  - `clip_polyline_unit(points: list[tuple[float,float,bool]]) -> list[tuple[float,float,bool]]`
  (points are `(x, y, hidden)`, hidden is bool; clip preserves order + hidden, inserts boundary crossings, drops out-of-frame points.)

- [ ] **Step 1: Write failing tests**

Create `tests/test_habitat_data.py`:

```python
from rover_vlm.habitat_data import normalize_points, clip_polyline_unit


def test_normalize_divides_by_size():
    out = normalize_points([(256.0, 128.0, False), (512.0, 512.0, True)], 512, 512)
    assert out == [(0.5, 0.25, False), (1.0, 1.0, True)]


def test_clip_keeps_fully_inside():
    pts = [(0.2, 0.9, False), (0.3, 0.5, False), (0.4, 0.2, False)]
    assert clip_polyline_unit(pts) == pts


def test_clip_inserts_bottom_boundary():
    # path starts below the frame (y > 1) then enters; expect a point at y == 1
    pts = [(0.5, 1.6, False), (0.5, 0.4, False)]
    out = clip_polyline_unit(pts)
    assert all(0.0 <= y <= 1.0 for _, y, _ in out)
    assert any(abs(y - 1.0) < 1e-6 for _, y, _ in out)
    assert out[-1] == (0.5, 0.4, False)


def test_clip_preserves_visibility_transition():
    # visible run then hidden run, all in-frame
    pts = [(0.5, 0.9, False), (0.5, 0.6, False), (0.5, 0.6, True), (0.5, 0.3, True)]
    out = clip_polyline_unit(pts)
    flags = [h for _, _, h in out]
    assert False in flags and True in flags
    # transition index exists
    assert any(flags[i] != flags[i - 1] for i in range(1, len(flags)))
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.local/bin/uv run pytest tests/test_habitat_data.py -v`
Expected: FAIL (ModuleNotFoundError: rover_vlm.habitat_data).

- [ ] **Step 3: Implement**

Create `src/rover_vlm/habitat_data.py`:

```python
"""Habitat rover-navigation data: select correct path, normalize, clip, resample,
format the path+visibility conversation record.

Source layout: <dataset_root>/<scene>/samples/<scene>_cNNN/ with fpv_enhanced.png,
fpv_paths.json, meta.json. See docs/superpowers/specs/2026-07-17-habitat-path-visibility-design.md.

Coordinates normalize to [0,1] by fpv_paths["image_size"]; visibility v is
1 = visible, 0 = obstructed (v = 0 if the run/goal is "hidden").
"""

import json
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/dataset")
IMAGE_NAME = "fpv_enhanced.png"
MAX_WAYPOINTS = 10
HARD_CAP = 12

HABITAT_PROMPT = (
    "<image>\n"
    "You are a rover navigating an indoor environment. The goal is located straight "
    "ahead. Predict the traversable path to the goal as a list of waypoints. Each "
    "waypoint is [x, y, v] where x and y are normalized image coordinates in [0,1] and "
    "v is 1 if the point is on visible, unobstructed ground or 0 if it is obstructed "
    "(hidden behind an obstacle). Then give the goal as [x, y, v]. Answer as JSON: "
    '{"path": [[x, y, v], ...], "goal": [x, y, v]}.'
)


def normalize_points(raw, w, h):
    """(u, v, hidden) pixel points -> (x, y, hidden) normalized by image size."""
    return [(u / w, v / h, hidden) for u, v, hidden in raw]


def _liang_barsky(x0, y0, x1, y1):
    """Clip segment (x0,y0)->(x1,y1) to the unit square. Returns (t0, t1) or None."""
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - 0.0, 1.0 - x0, y0 - 0.0, 1.0 - y0]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None  # parallel and outside
        else:
            t = qi / pi
            if pi < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return t0, t1


def clip_polyline_unit(points):
    """Clip an ordered (x,y,hidden) polyline to [0,1]^2.

    Keeps in-frame vertices, inserts boundary-crossing points (inheriting the
    segment's hidden flag), preserves order and visibility transitions.
    """
    out = []

    def push(pt):
        if not out or abs(out[-1][0] - pt[0]) > 1e-9 or abs(out[-1][1] - pt[1]) > 1e-9 or out[-1][2] != pt[2]:
            out.append(pt)

    for i in range(len(points) - 1):
        x0, y0, h = points[i]
        x1, y1, _ = points[i + 1]
        seg = _liang_barsky(x0, y0, x1, y1)
        if seg is None:
            continue
        t0, t1 = seg
        a = (round(x0 + t0 * (x1 - x0), 6), round(y0 + t0 * (y1 - y0), 6), h)
        b = (round(x0 + t1 * (x1 - x0), 6), round(y0 + t1 * (y1 - y0), 6), h)
        push(a)
        push(b)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `~/.local/bin/uv run pytest tests/test_habitat_data.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/habitat_data.py tests/test_habitat_data.py
git commit -m "feat: habitat normalize + clip-to-frame geometry"
```

---

## Task 3: Resample preserving transitions

**Files:**
- Modify: `src/rover_vlm/habitat_data.py`
- Modify: `tests/test_habitat_data.py`

**Interfaces:**
- Consumes: `clip_polyline_unit` output `list[(x,y,hidden)]`.
- Produces: `resample_with_transitions(clipped, target=MAX_WAYPOINTS, cap=HARD_CAP) -> list[tuple[float,float,int]]` — `[x, y, v]` triples, `v in {0,1}`, 3-decimal coords, first/last + every transition preserved, length ≤ cap.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_habitat_data.py`:

```python
from rover_vlm.habitat_data import resample_with_transitions


def test_resample_short_all_visible():
    clipped = [(0.5, 0.9, False), (0.5, 0.6, False), (0.5, 0.3, False)]
    out = resample_with_transitions(clipped, target=10, cap=12)
    assert out == [(0.5, 0.9, 1), (0.5, 0.6, 1), (0.5, 0.3, 1)]


def test_resample_preserves_transition_flags():
    clipped = [(0.5, 0.9, False), (0.5, 0.6, False), (0.5, 0.6, True), (0.5, 0.3, True)]
    out = resample_with_transitions(clipped, target=4, cap=12)
    flags = [v for _, _, v in out]
    assert 1 in flags and 0 in flags
    assert out[0][2] == 1 and out[-1][2] == 0  # starts visible, ends obstructed


def test_resample_respects_cap():
    clipped = [(round(0.1 + 0.01 * i, 3), round(0.9 - 0.01 * i, 3), False) for i in range(60)]
    out = resample_with_transitions(clipped, target=10, cap=12)
    assert len(out) <= 12
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.local/bin/uv run pytest tests/test_habitat_data.py -k resample -v`
Expected: FAIL (resample_with_transitions not defined).

- [ ] **Step 3: Implement**

Append to `src/rover_vlm/habitat_data.py`:

```python
def resample_with_transitions(clipped, target=MAX_WAYPOINTS, cap=HARD_CAP):
    """Reduce a clipped (x,y,hidden) polyline to <= cap [x,y,v] waypoints.

    Arc-length uniform sampling toward `target`, but always keep the first and last
    point and both sides of every visible<->obstructed transition. Transitions take
    priority over uniform fill when the cap is tight. v = 1 visible, 0 obstructed.
    """
    n = len(clipped)
    if n == 0:
        return []
    xy = np.array([(x, y) for x, y, _ in clipped], dtype=float)
    hid = [h for _, _, h in clipped]
    if n == 1:
        return [(round(xy[0, 0], 3), round(xy[0, 1], 3), 0 if hid[0] else 1)]

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    forced = {0, n - 1}
    for i in range(1, n):
        if hid[i] != hid[i - 1]:
            forced.add(i - 1)
            forced.add(i)

    if total > 0:
        targets = np.linspace(0.0, total, target)
        uni = {int(np.argmin(np.abs(cum - t))) for t in targets}
    else:
        uni = set(forced)

    keep = set(forced) | uni
    if len(keep) > cap:
        extra = sorted(keep - forced)
        while len(forced) + len(extra) > cap and extra:
            extra.pop(len(extra) // 2)  # thin the uniformly-spaced extras from the middle
        keep = forced | set(extra)

    return [
        (round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3), 0 if hid[i] else 1)
        for i in sorted(keep)
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `~/.local/bin/uv run pytest tests/test_habitat_data.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/habitat_data.py tests/test_habitat_data.py
git commit -m "feat: resample habitat path preserving visibility transitions"
```

---

## Task 4: Load sample + build conversation record

**Files:**
- Modify: `src/rover_vlm/habitat_data.py`
- Modify: `tests/test_habitat_data.py`

**Interfaces:**
- Consumes: `normalize_points`, `clip_polyline_unit`, `resample_with_transitions`, `HABITAT_PROMPT`.
- Produces:
  - `select_correct_path(fpv_paths: dict, meta: dict) -> list[tuple[float,float,bool]] | None`
  - `format_answer(path_wps: list[tuple[float,float,int]], goal_wp: tuple[float,float,int]) -> str`
  - `build_record(sample_dir: Path) -> dict | None` — returns `{"id","image":[abs_path],"conversations":[human,gpt]}` or None if the sample is filtered/invalid.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_habitat_data.py`:

```python
import json
from rover_vlm.habitat_data import select_correct_path, format_answer, build_record, DATASET_ROOT


def test_select_correct_path_uses_label_and_concats_runs():
    fpv = {"candidates": [
        {"runs": [{"hidden": False, "uv": [[1.0, 2.0], [3.0, 4.0]]}]},
        {"runs": [{"hidden": False, "uv": [[10.0, 20.0]]}, {"hidden": True, "uv": [[30.0, 40.0]]}]},
    ]}
    meta = {"label": 1, "candidates": [{}, {}]}
    assert select_correct_path(fpv, meta) == [(10.0, 20.0, False), (30.0, 40.0, True)]


def test_select_correct_path_bad_label_returns_none():
    fpv = {"candidates": [{"runs": []}]}
    meta = {"label": 5, "candidates": [{}]}
    assert select_correct_path(fpv, meta) is None


def test_format_answer_is_parseable_json():
    s = format_answer([(0.4, 0.8, 1), (0.4, 0.5, 0)], (0.5, 0.58, 0))
    obj = json.loads(s)
    assert obj == {"path": [[0.4, 0.8, 1], [0.4, 0.5, 0]], "goal": [0.5, 0.58, 0]}


def test_build_record_on_live_sample():
    import pytest
    if not DATASET_ROOT.exists():
        pytest.skip("dataset not mounted")
    sample = next(DATASET_ROOT.glob("*/samples/*/"))
    rec = build_record(sample)
    if rec is None:
        pytest.skip("first sample filtered out; covered by unit tests")
    assert rec["image"][0].endswith("fpv_enhanced.png")
    obj = json.loads(rec["conversations"][1]["value"])
    assert "path" in obj and "goal" in obj
    for x, y, v in obj["path"] + [obj["goal"]]:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and v in (0, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.local/bin/uv run pytest tests/test_habitat_data.py -k "select_correct or format_answer or build_record" -v`
Expected: FAIL (names not defined).

- [ ] **Step 3: Implement**

Append to `src/rover_vlm/habitat_data.py`:

```python
def select_correct_path(fpv_paths, meta):
    """Concatenate the runs of the correct candidate (meta['label']) into
    (u, v, hidden) points. None if the label is missing/misaligned."""
    label = meta.get("label")
    cands = fpv_paths.get("candidates", [])
    if label is None or label >= len(cands) or len(cands) != len(meta.get("candidates", [])):
        return None
    runs = cands[label].get("runs", [])
    return [(float(u), float(v), bool(run["hidden"])) for run in runs for (u, v) in run["uv"]]


def _clamp01(v):
    return round(min(max(v, 0.0), 1.0), 3)


def format_answer(path_wps, goal_wp):
    obj = {
        "path": [[x, y, v] for x, y, v in path_wps],
        "goal": [goal_wp[0], goal_wp[1], goal_wp[2]],
    }
    return json.dumps(obj, separators=(",", ":"))


def build_record(sample_dir):
    """Habitat sample dir -> conversation record, or None if filtered/invalid."""
    sample_dir = Path(sample_dir)
    fpv = json.loads((sample_dir / "fpv_paths.json").read_text())
    meta = json.loads((sample_dir / "meta.json").read_text())
    if not meta.get("correct_path_in_fov"):
        return None
    raw = select_correct_path(fpv, meta)
    if not raw or len(raw) < 2:
        return None
    w, h = fpv["image_size"]
    clipped = clip_polyline_unit(normalize_points(raw, w, h))
    if len(clipped) < 2:
        return None
    path_wps = resample_with_transitions(clipped)
    g = fpv["goal"]
    goal_wp = (_clamp01(g["uv"][0] / w), _clamp01(g["uv"][1] / h), 0 if g["hidden"] else 1)
    answer = format_answer(path_wps, goal_wp)
    image_abs = str(sample_dir / IMAGE_NAME)
    return {
        "id": sample_dir.name,
        "image": [image_abs],
        "conversations": [
            {"from": "human", "value": HABITAT_PROMPT},
            {"from": "gpt", "value": answer},
        ],
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `~/.local/bin/uv run pytest tests/test_habitat_data.py -v`
Expected: all pass (the live test may `skip` — acceptable).

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/habitat_data.py tests/test_habitat_data.py
git commit -m "feat: build habitat conversation record from a sample dir"
```

---

## Task 5: prepare_habitat.py (splits + write)

**Files:**
- Create: `scripts/prepare_habitat.py`

**Interfaces:**
- Consumes: `build_record`, `DATASET_ROOT` from `rover_vlm.habitat_data`; `make_splits` from `rover_vlm.data`.
- Produces: `data/prepared_habitat/{eval,train_500,train_1000,train_2000,train_full,meta}.json`.

- [ ] **Step 1: Implement the script**

Create `scripts/prepare_habitat.py`:

```python
"""Convert the Habitat dataset into conversation-format splits for training/eval.

Login node (CPU-only):
    uv run scripts/prepare_habitat.py
    uv run scripts/prepare_habitat.py --limit 200   # quick subset for testing

Reads every <root>/*/samples/*/ dir, keeps correct-path-in-FOV samples, and writes
fixed-seed nested splits to data/prepared_habitat/. Images are referenced by absolute
NFS path (never copied).
"""

import argparse
import json
from pathlib import Path

from rover_vlm.data import make_splits
from rover_vlm.habitat_data import DATASET_ROOT, build_record

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "prepared_habitat"
EVAL_SIZE = 500
TRAIN_SIZES = [500, 1000, 2000]
SEED = 42


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="cap sample dirs scanned (testing)")
    args = p.parse_args()

    sample_dirs = sorted(args.dataset_root.glob("*/samples/*/"))
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]
    print(f"scanning {len(sample_dirs)} sample dirs under {args.dataset_root}")

    records, skipped = [], 0
    for i, d in enumerate(sample_dirs):
        try:
            rec = build_record(d)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {d.name}: {e}")
            rec = None
        if rec is None:
            skipped += 1
        else:
            records.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(sample_dirs)} ({len(records)} kept, {skipped} skipped)")

    print(f"kept {len(records)} records, skipped {skipped}")
    if len(records) <= EVAL_SIZE:
        raise SystemExit(f"only {len(records)} records — need > {EVAL_SIZE} for an eval split")

    splits = make_splits(records, EVAL_SIZE, TRAIN_SIZES, SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        (args.out_dir / f"{name}.json").write_text(json.dumps(recs, indent=1))
        print(f"  {name}: {len(recs)}")

    # eval disjoint from every train subset
    eval_ids = {r["id"] for r in splits["eval"]}
    for name, recs in splits.items():
        if name.startswith("train_"):
            assert not (eval_ids & {r["id"] for r in recs}), f"{name} overlaps eval!"

    meta = {"splits": {k: len(v) for k, v in splits.items()}, "kept": len(records), "skipped": skipped}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote splits -> {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on a small subset**

Run: `~/.local/bin/uv run scripts/prepare_habitat.py --limit 1200 --out-dir /tmp/prep_test`
Expected: prints kept/skipped, writes `eval.json` + `train_500.json` (train_1000/2000 may be absent if the pool is small) + `meta.json`; the disjointness assert passes (no AssertionError).

- [ ] **Step 3: Spot-check a record**

Run:
```bash
~/.local/bin/uv run python -c "
import json; r=json.load(open('/tmp/prep_test/eval.json'))[0]
print(r['id']); print(r['image'][0]); print(r['conversations'][1]['value'][:200])
obj=json.loads(r['conversations'][1]['value']); print('path len', len(obj['path']), 'goal', obj['goal'])
"
```
Expected: absolute `.../fpv_enhanced.png` path; a parseable `{"path":[...],"goal":[...]}`; path length ≤ 12; goal is `[x,y,v]`.

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_habitat.py
git commit -m "feat: prepare_habitat.py — nested Habitat splits"
```

---

## Task 6: inspect_habitat.py (visibility overlay)

**Files:**
- Create: `scripts/inspect_habitat.py`

**Interfaces:**
- Consumes: `build_record`, `DATASET_ROOT` (for image resolution the record stores the absolute path).

- [ ] **Step 1: Implement the script**

Create `scripts/inspect_habitat.py`:

```python
"""Overlay the prepared path + goal (colored by visibility) on fpv_enhanced.png.

Login node (CPU-only), run after prepare_habitat.py or directly on the dataset:
    uv run scripts/inspect_habitat.py --num 16

Writes outputs/inspection_habitat/overlays/*.png. Visible waypoints = green,
obstructed = red; goal ring green (visible) / red (obstructed). Sanity-checks the
coordinate + visibility convention before training.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from rover_vlm.habitat_data import DATASET_ROOT, build_record

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "inspection_habitat" / "overlays"
VIS, OBS = "#2a78d6", "#d64a2a"  # visible=blue, obstructed=red (green reserved for goal)


def draw(rec: dict, out_path: Path) -> None:
    obj = json.loads(rec["conversations"][1]["value"])
    img = Image.open(rec["image"][0]).convert("RGB")
    W, H = img.size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    pts = obj["path"]
    xs = [p[0] * W for p in pts]
    ys = [p[1] * H for p in pts]
    ax.plot(xs, ys, "-", color="#dddddd", linewidth=1.5, zorder=1)
    for (x, y, v) in pts:
        ax.plot(x * W, y * H, "o", color=(VIS if v == 1 else OBS), markersize=7, zorder=2)
    gx, gy, gv = obj["goal"]
    ax.plot(gx * W, gy * H, "*", color=("#00a000" if gv == 1 else "#d00000"),
            markersize=22, markeredgecolor="white", zorder=3)
    ax.set_title(f"{rec['id']}  (goal {'visible' if gv == 1 else 'obstructed'})", fontsize=9)
    ax.axis("off")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dirs = sorted(args.dataset_root.glob("*/samples/*/"))
    random.Random(args.seed).shuffle(dirs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for d in dirs:
        if drawn >= args.num:
            break
        rec = build_record(d)
        if rec is None:
            continue
        draw(rec, OUT_DIR / f"{rec['id']}.png")
        drawn += 1
    print(f"wrote {drawn} overlays -> {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `~/.local/bin/uv run scripts/inspect_habitat.py --num 12`
Expected: `wrote 12 overlays -> .../outputs/inspection_habitat/overlays`.

- [ ] **Step 3: Eyeball (manual)**

Open several overlays. Verify: the path lies on the floor toward the doorway/goal; blue (visible) points sit on observed floor and red (obstructed) points continue past the occluder; the goal star color matches the title's visible/obstructed. Check one single-transition and one multi-transition sample.

- [ ] **Step 4: Commit**

```bash
git add scripts/inspect_habitat.py
git commit -m "feat: inspect_habitat.py — visibility-colored path overlays"
```

---

## Task 7: Habitat answer parser + metrics

**Files:**
- Modify: `src/rover_vlm/eval.py`
- Create: `tests/test_habitat_eval.py`

**Interfaces:**
- Consumes: existing `resample_polyline`, `frechet_distance` from `rover_vlm.eval`.
- Produces:
  - `parse_path_answer(text: str) -> dict | None` → `{"path": [[x,y,v],...], "goal": [x,y,v]}` or None.
  - `habitat_metrics(pred: dict, gt: dict, n_resample: int = 10) -> dict` → per-sample
    `{mean_point_error, frechet, path_visibility_acc, goal_point_error, goal_visibility_correct}`.
  - `aggregate_habitat_metrics(records: list[dict]) -> dict` → parse/valid rates + mean/median of each.

- [ ] **Step 1: Write failing tests**

Create `tests/test_habitat_eval.py`:

```python
from rover_vlm.eval import parse_path_answer, habitat_metrics, aggregate_habitat_metrics


def test_parse_clean_object():
    out = parse_path_answer('{"path":[[0.4,0.8,1],[0.4,0.5,0]],"goal":[0.5,0.6,0]}')
    assert out["path"] == [[0.4, 0.8, 1], [0.4, 0.5, 0]]
    assert out["goal"] == [0.5, 0.6, 0]


def test_parse_with_surrounding_text():
    out = parse_path_answer('Here: {"path": [[0.1,0.2,1],[0.3,0.4,1]], "goal": [0.5,0.6,1]} done')
    assert out is not None and len(out["path"]) == 2 and out["goal"][2] == 1


def test_parse_garbage_returns_none():
    assert parse_path_answer("no coordinates here") is None


def test_habitat_metrics_perfect_match():
    gt = {"path": [[0.5, 0.9, 1], [0.5, 0.5, 0]], "goal": [0.5, 0.3, 0]}
    m = habitat_metrics(gt, gt)
    assert m["mean_point_error"] < 1e-9
    assert m["path_visibility_acc"] == 1.0
    assert m["goal_point_error"] < 1e-9
    assert m["goal_visibility_correct"] == 1


def test_habitat_metrics_wrong_goal_visibility():
    gt = {"path": [[0.5, 0.9, 1], [0.5, 0.5, 1]], "goal": [0.5, 0.3, 1]}
    pred = {"path": [[0.5, 0.9, 1], [0.5, 0.5, 1]], "goal": [0.5, 0.3, 0]}
    m = habitat_metrics(pred, gt)
    assert m["goal_visibility_correct"] == 0


def test_aggregate_reports_rates():
    recs = [
        {"parsed": {"path": [[0.5, 0.9, 1]], "goal": [0.5, 0.3, 1]},
         "metrics": {"mean_point_error": 0.1, "frechet": 0.2, "path_visibility_acc": 1.0,
                     "goal_point_error": 0.05, "goal_visibility_correct": 1}},
        {"parsed": None, "metrics": None},
    ]
    agg = aggregate_habitat_metrics(recs)
    assert agg["parse_rate"] == 0.5
    assert agg["goal_visibility_accuracy"] == 1.0  # over parsed only
```

- [ ] **Step 2: Run to verify failure**

Run: `~/.local/bin/uv run pytest tests/test_habitat_eval.py -v`
Expected: FAIL (names not defined).

- [ ] **Step 3: Implement**

Append to `src/rover_vlm/eval.py` (it already imports `re`, `numpy as np`, and defines `resample_polyline`, `frechet_distance`):

```python
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
```

Note: the `pred["path"] or ...` guards keep the metric defined when a prediction has an empty path but a goal.

- [ ] **Step 4: Run to verify pass**

Run: `~/.local/bin/uv run pytest tests/test_habitat_eval.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/eval.py tests/test_habitat_eval.py
git commit -m "feat: habitat answer parser + position/visibility metrics"
```

---

## Task 8: evaluate.py habitat mode

**Files:**
- Modify: `scripts/evaluate.py`

**Interfaces:**
- Consumes: `parse_path_answer`, `habitat_metrics`, `aggregate_habitat_metrics` (habitat); existing `parse_waypoints`/`trajectory_metrics`/`aggregate_metrics` (sharerobot).

- [ ] **Step 1: Add a `--task` flag and default eval file**

In `scripts/evaluate.py`, modify `parse_args()` to add:

```python
    p.add_argument("--task", choices=["sharerobot", "habitat"], default="sharerobot")
```

- [ ] **Step 2: Import habitat eval helpers**

Change the `from rover_vlm.eval import (...)` block to also import the habitat functions:

```python
from rover_vlm.eval import (
    aggregate_metrics,
    aggregate_habitat_metrics,
    habitat_metrics,
    normalize_prediction,
    parse_path_answer,
    parse_waypoints,
    trajectory_metrics,
)
```

- [ ] **Step 3: Branch gt-parsing, prediction-parsing, and metrics on task**

In `main()`, replace the per-batch gt build and the scoring loop with task-aware logic. Where the code currently does `gts.append(json.loads(rec["conversations"][1]["value"]))`, keep it (both tasks store JSON — a list for sharerobot, an object for habitat). Replace the scoring block:

```python
        for rec, gt, text in zip(chunk, gts, decoded):
            if args.task == "habitat":
                parsed = parse_path_answer(text)
                metrics = habitat_metrics(parsed, gt) if parsed else None
                rescaled = False
            else:
                parsed = parse_waypoints(text)
                if parsed:
                    scored, rescaled = normalize_prediction(parsed)
                    metrics = trajectory_metrics(scored, gt)
                else:
                    metrics, rescaled = None, False
            results.append({
                "id": rec["id"], "generated": text, "parsed": parsed,
                "rescaled": rescaled, "gt": gt, "metrics": metrics,
            })
```

And replace the final aggregate line:

```python
    summary = aggregate_habitat_metrics(results) if args.task == "habitat" else aggregate_metrics(results)
```

- [ ] **Step 4: Point the image root at absolute paths**

Habitat records store absolute image paths, so `IMAGE_ROOT / rec["image"][0]` already resolves correctly (an absolute right-hand operand wins). No change needed — verify by reading the existing `Image.open(IMAGE_ROOT / rec["image"][0])` line and confirming records carry absolute paths.

- [ ] **Step 4b: Add `--out-dir` so habitat evals don't collide with ShareRobot**

Add to `parse_args()`:

```python
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eval")
```

And change the output path line in `main()` from `out_dir = REPO_ROOT / "outputs" / "eval" / args.tag` to:

```python
    out_dir = args.out_dir / args.tag
```

- [ ] **Step 5: CPU sanity check (no GPU needed for parsing path)**

Run:
```bash
~/.local/bin/uv run python -c "
from rover_vlm.eval import parse_path_answer, habitat_metrics
gt={'path':[[0.5,0.9,1],[0.5,0.5,0]],'goal':[0.5,0.3,0]}
p=parse_path_answer('{\"path\":[[0.5,0.9,1],[0.5,0.5,0]],\"goal\":[0.5,0.3,0]}')
print(habitat_metrics(p,gt))
"
```
Expected: a dict with near-zero errors, `path_visibility_acc` 1.0, `goal_visibility_correct` 1.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: evaluate.py habitat task mode (path+visibility)"
```

---

## Task 9: Visibility-aware comparison table

**Files:**
- Modify: `scripts/compare_evals.py`

**Interfaces:**
- Consumes: habitat `metrics.json` keys (`path_visibility_acc_mean`, `goal_visibility_accuracy`, `goal_point_error_mean`, etc.).

**Note:** Adapting the interactive `visualize_predictions.py` explorer to habitat
(configurable tags + the `{"path","goal"}` data shape + visibility coloring) is a
deferred follow-up — `inspect_habitat.py` (Task 6) covers qualitative visibility
inspection for this phase.

- [ ] **Step 1: Add habitat columns to the comparison table**

In `scripts/compare_evals.py`, extend `TABLE_COLUMNS` so habitat metrics render when present (missing keys already show `—` via the existing `f"{m[k]:.3f}" if k in m else "—"`):

```python
TABLE_COLUMNS = [
    ("parse_rate", "Parse rate"),
    ("in_range_rate", "In-range rate"),
    ("rescaled_rate", "Rescaled (0-1000)"),
    ("mean_point_error_mean", "Mean point err"),
    ("mean_point_error_median", "Median point err"),
    ("endpoint_error_mean", "Endpoint err"),
    ("frechet_mean", "Fréchet"),
    ("path_visibility_acc_mean", "Vis. acc"),
    ("goal_point_error_mean", "Goal err"),
    ("goal_visibility_accuracy", "Goal vis. acc"),
]
```

- [ ] **Step 1b: Teach `train_size` + base-detection about habitat tags**

Habitat eval tags are `habitat_base` / `habitat_train_<size>` and their `full` size lives in `data/prepared_habitat/meta.json`. Add a `--meta` arg and generalize.

In `parse_args`/`main`, add:
```python
    parser.add_argument("--meta", type=Path, default=REPO_ROOT / "data/prepared/meta.json")
```
Change `train_size` to take the meta path and accept both prefixes:
```python
def train_size(tag: str, meta_path: Path) -> int | None:
    for prefix in ("lora_train_", "habitat_train_"):
        if tag.startswith(prefix):
            suffix = tag.removeprefix(prefix)
            if suffix == "full":
                return json.loads(meta_path.read_text())["splits"]["train_full"]
            return int(suffix) if suffix.isdigit() else None
    return None
```
In `load_runs`, treat any tag ending in `base` as the baseline and pass the meta path:
```python
        if m["tag"].endswith("base"):
            base = m
        else:
            size = train_size(m["tag"], meta_path)
```
(Thread `meta_path` from `args.meta` into `load_runs`.)

- [ ] **Step 2: Verify compare still runs on existing sharerobot evals**

Run: `~/.local/bin/uv run scripts/compare_evals.py`
Expected: prints the ShareRobot table (new habitat columns show `—`), writes `comparison.md` — no error. (ShareRobot tag `base` still matches `endswith("base")`; `lora_train_*` still map via the first prefix.)

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_evals.py
git commit -m "feat: visibility-aware comparison table + habitat tag/meta support"
```

---

## Task 10: SLURM wrappers, full prepare, smoke, README

**Files:**
- Create: `slurm/train_habitat.sbatch`, `slurm/eval_habitat.sbatch`
- Modify: `README.md`

- [ ] **Step 1: Train sbatch**

Create `slurm/train_habitat.sbatch` (mirrors `slurm/train.sbatch`, defaults to the habitat train file):

```bash
#!/bin/bash
# LoRA fine-tune on Habitat path+visibility data.
#   sbatch slurm/train_habitat.sbatch [train-file] [output-dir]
#SBATCH --job-name=qwen-habitat
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

TRAIN_FILE="${1:-data/prepared_habitat/train_full.json}"
stem=$(basename "$TRAIN_FILE" .json)
OUTPUT_DIR="${2:-outputs/runs/habitat_${stem}}"
shift $(( $# >= 2 ? 2 : $# )) || true

$UV run scripts/train.py --train-file "$TRAIN_FILE" --output-dir "$OUTPUT_DIR" "$@"
```

- [ ] **Step 2: Eval sbatch**

Create `slurm/eval_habitat.sbatch`:

```bash
#!/bin/bash
# Evaluate base or base+adapter on the Habitat eval split.
#   sbatch slurm/eval_habitat.sbatch base
#   sbatch slurm/eval_habitat.sbatch habitat_train_full outputs/runs/habitat_train_full/adapter
#SBATCH --job-name=qwen-habitat-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

TAG="${1:-habitat_base}"
ADAPTER="${2:-}"
EVAL_FILE=data/prepared_habitat/eval.json
OUT_DIR=outputs/eval_habitat   # separate dir so ShareRobot evals stay intact
if [ -n "$ADAPTER" ]; then
    $UV run scripts/evaluate.py --task habitat --tag "$TAG" --adapter "$ADAPTER" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR"
else
    $UV run scripts/evaluate.py --task habitat --tag "$TAG" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR"
fi
```

Tags: `habitat_base` for the zero-shot baseline, `habitat_train_<size>` for adapters.

- [ ] **Step 3: Run the full prepare**

Run: `~/.local/bin/uv run scripts/prepare_habitat.py`
Expected: scans ~4,398 dirs, keeps ~4,100 (skips ~6% not-in-FOV + any invalid), writes `data/prepared_habitat/{eval,train_500,train_1000,train_2000,train_full,meta}.json`; disjointness assert passes. Check `meta.json` split counts.

- [ ] **Step 4: CPU smoke train on habitat data**

Run: `~/.local/bin/uv run scripts/train.py --smoke --train-file data/prepared_habitat/train_500.json`
Expected: `== SMOKE TEST PASSED ==` (loss finite, adapter reload OK). This confirms the absolute-image-path + new-answer records flow through training unchanged.

- [ ] **Step 5: Update README**

Add a "Habitat phase" subsection to `README.md` after the ShareRobot workflow, documenting:
- `uv run scripts/prepare_habitat.py` → `data/prepared_habitat/`
- `uv run scripts/inspect_habitat.py` → overlays
- `sbatch slurm/train_habitat.sbatch [train-file]` and the size loop
- `sbatch slurm/eval_habitat.sbatch <tag> [adapter]` (tags `habitat_base` / `habitat_train_<size>`, results in `outputs/eval_habitat/`), then
  `uv run scripts/compare_evals.py --eval-dir outputs/eval_habitat --meta data/prepared_habitat/meta.json`
- note: images referenced by absolute NFS path; visibility `v` = 1 visible / 0 obstructed.

- [ ] **Step 6: Run the full test suite**

Run: `~/.local/bin/uv run pytest -v`
Expected: all tests pass (live-data tests may skip).

- [ ] **Step 7: Commit**

```bash
git add slurm/train_habitat.sbatch slurm/eval_habitat.sbatch README.md
git commit -m "feat: habitat SLURM wrappers + README; full prepare verified"
```

---

## After the plan — cluster runs (user-launched, not part of task commits)

1. `sbatch slurm/eval_habitat.sbatch habitat_base` — zero-shot baseline → `outputs/eval_habitat/habitat_base/`.
2. For each size in 500/1000/2000/full: `sbatch slurm/train_habitat.sbatch data/prepared_habitat/train_<size>.json outputs/runs/habitat_train_<size>`, then `sbatch slurm/eval_habitat.sbatch habitat_train_<size> outputs/runs/habitat_train_<size>/adapter`.
3. `uv run scripts/compare_evals.py --eval-dir outputs/eval_habitat --meta data/prepared_habitat/meta.json` for the scaling table + visibility accuracy. Use `scripts/inspect_habitat.py` for qualitative per-sample visibility overlays (the interactive explorer's habitat adaptation is a deferred follow-up).
   (Optionally add a single-job `slurm/run_all_habitat.sbatch` mirroring `run_all.sbatch` once eval dirs/tags are settled.)

## Notes / risks (from the spec)
- Visibility accuracy aligns pred↔GT by arc-length resample + nearest-original-waypoint flag. Revisit if noisy.
- Multi-transition paths honor the ~12 cap with transitions prioritized; `prepare_habitat.py` logs skips, no silent truncation.
- Goal x is ~0.5 for all samples — its informative signal is visibility + vertical position.
