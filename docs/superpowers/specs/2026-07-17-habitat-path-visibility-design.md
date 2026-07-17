# Design: Habitat rover path + visibility fine-tuning

**Date:** 2026-07-17
**Status:** approved design, pre-implementation
**Supersedes phase:** ShareRobot feasibility (done, commit `6d36ab3`)

## Context

The ShareRobot feasibility phase confirmed the LoRA fine-tuning loop works and
scales cleanly (base unusable → LoRA 100% valid, monotonic error reduction). The
results were good enough that we skip the intermediate reasoning-trace step for now
and move straight to the **real target dataset**: Habitat-generated rover navigation
samples. Reasoning traces (Cosmos Reason 3) remain deferred — see the
`reasoning-trace-think-block` memory.

New task vs. ShareRobot: instead of a manipulation trajectory, the model predicts a
**traversable navigation path to a goal**, and — new requirement — labels **each
waypoint and the goal as visible or obstructed** (hidden behind an obstacle).

## Goal

Fine-tune Qwen3.5-2B (LoRA) to take a single forward-facing `fpv_enhanced.png` and
output the correct traversable path to the goal as a list of `[x, y, v]` waypoints
(v = visibility) plus the goal as `[x, y, v]`, then evaluate base vs. LoRA across
training-set sizes — mirroring the ShareRobot scaling study, extended with
visibility metrics.

### Non-goals
- Reasoning/`<think>` traces (deferred).
- Using BEV (`bev_*`), depth, or `state.npz` as model input — FPV RGB only.
- Habitat scene generation / A* (already done upstream; we only consume the output).
- Predicting the *incorrect* candidate paths or a choice among candidates — we train
  only on the single correct path per sample.

## Source dataset

Path: `/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/dataset`
(NFS, shared with thor; **not** copied into the repo). Layout:
`<scene>/samples/<scene>_cNNN/` — one navigation decision per sample dir. 4,398
sample dirs across ~376 HM3D scenes (5.9 GB).

Per-sample files we use:
- `fpv_enhanced.png` — photorealistic first-person RGB (1024×1024). **Model input.**
- `fpv_paths.json` — FPV-projected paths:
  - `image_size`: `[512, 512]` — the coordinate frame for all `uv` values
    (note: half the resolution of `fpv_enhanced.png`; normalize by this, not by pixels).
  - `goal`: `{ "uv": [u, v], "hidden": bool }` — goal projected into the image;
    `u` is always 256 (center); `hidden` ⇒ obstructed.
  - `candidates`: list aligned by index with `meta["candidates"]`. Each has
    `runs`: an ordered list of `{ "hidden": bool, "uv": [[u,v], ...] }`. Concatenating
    a candidate's runs gives its full ordered polyline (near→far), each point carrying
    its run's `hidden` flag. Visibility can alternate multiple times.
- `meta.json` — `label` (int, index of the correct candidate), `accepted` (list; may
  contain several — we use `label`), `correct_path_in_fov` (bool), plus per-candidate
  world-space info we do **not** need for training.

Verified over a 500-sample scan: candidate counts align `fpv_paths` == `meta` 100%;
the correct candidate always `reaches_goal`; goal `u` always 256; goal visible 23% /
obstructed 77%; `correct_path_in_fov` true for 94%.

## Sample selection

- One training sample per `samples/<scene>_cNNN/`.
- **Filter:** keep only `meta["correct_path_in_fov"] == True` (~4,130 samples).
- Correct path = `fpv_paths["candidates"][meta["label"]]`. Assert `label` is a valid
  index and the fpv/meta candidate counts match; skip + log any sample that fails.

## Geometry pipeline

Per correct path (ordered near→far, i.e. from the rover outward to the goal):

1. **Flatten runs → points** `[(u, v, hidden), ...]`, preserving order.
2. **Normalize** by `image_size`: `x = u / W`, `y = v / H` (W = H = 512). Coordinates
   may fall outside `[0, 1]` (path starts below the frame; can exit the sides).
3. **Clip to the unit square.** Walk consecutive point pairs; keep in-frame points; at
   any segment that crosses the `[0,1]²` boundary, insert the intersection point,
   assigning it the segment's `hidden` flag. Drop out-of-frame points. Result: the
   in-frame portion of the path, visibility-labeled. (~21% of raw points are dropped
   as off-frame; median in-frame fraction 0.79.)
4. **Resample to waypoints** with a target budget of **10**, arc-length spaced, but:
   - **force-keep every visibility transition** (both endpoints of each
     visible↔obstructed boundary that lies in-frame),
   - always keep the first in-frame point (path entry),
   - **hard cap ≈ 12** total path waypoints (transitions take priority over uniform
     spacing when the budget is tight).
   Each resampled waypoint's `v` is the `hidden` flag of the segment it lies on
   (1 = visible, 0 = obstructed).
5. **Goal** is appended separately (not part of the path list): `[x, y, v]` from
   `goal.uv` / `goal.hidden`. The goal is always in-frame (center, near horizon).
6. Round coordinates to 3 decimals.

## Output format & prompt

Conversation format identical in spirit to ShareRobot (`<image>` + human/gpt turns),
so the existing training code consumes it unchanged.

**Answer (gpt turn)** — compact numeric triples in one JSON object:
```json
{"path": [[0.42,0.83,1],[0.40,0.71,1],[0.39,0.55,0],[0.44,0.40,0]], "goal": [0.50,0.58,0]}
```
- `path`: ordered `[x, y, v]` waypoints, near→far. `v`: 1 = visible, 0 = obstructed.
- `goal`: single `[x, y, v]`.

**Prompt (human turn)** — fixed (the goal is always straight ahead, no per-sample
task string):
> `<image>`
> You are a rover navigating an indoor environment. The goal is located straight
> ahead. Predict the traversable path to the goal as a list of waypoints. Each
> waypoint is `[x, y, v]` where x and y are normalized image coordinates in [0,1] and
> v is 1 if the point is on visible, unobstructed ground or 0 if it is obstructed
> (hidden behind an obstacle). Then give the goal as `[x, y, v]`. Answer as JSON:
> `{"path": [[x, y, v], ...], "goal": [x, y, v]}`.

## Splits

Fixed-seed held-out eval split (500 samples) + nested training subsets
(500 ⊂ 1K ⊂ 2K ⊂ full ≈ 4,130 minus eval) so the scaling comparison isn't confounded
by subset composition — same discipline as ShareRobot. Assert eval disjoint from all
train subsets. Emit `eval.json`, `train_{500,1000,2000}.json`, `train_full.json`,
`meta.json` into `data/prepared_habitat/`.

## Training

Reuse `scripts/train.py` and the existing SLURM sbatch **unchanged** — they are
format-agnostic (train on whatever answer string the conversation carries). Only the
`--train-file` / `--output-dir` change (point at `data/prepared_habitat/`). Same LoRA
config as ShareRobot (r=16, α=32, lr 2e-4, eff. batch 16, 2 epochs, LM tower only) as
the starting point.

## Eval & metrics

Extend `src/rover_vlm/eval.py` with a parser + metrics for the new answer:

- **Parsing:** read the `{"path": [...], "goal": [...]}` object; tolerate minor
  format drift (regex-fallback for the triples). Split each triple into position
  `(x, y)` and visibility `v`.
- **Metrics (over parseable predictions):**
  - format/parse rate; in-range rate (as now).
  - **path position error:** resampled mean point error + discrete Fréchet on the
    `(x, y)` polyline vs. ground truth (reuse existing `resample_polyline` /
    `frechet_distance`).
  - **path visibility accuracy:** per-waypoint `v` correctness. Because predicted and
    GT waypoint counts differ, compare by nearest-GT-point assignment along arc length
    (define once; document the matching rule).
  - **goal position error** and **goal visibility accuracy** (binary
    visible/obstructed classification — the "green vs obstructed" call).
- **Reporting:** `scripts/compare_evals.py` gains the visibility columns; the
  interactive explorer (`scripts/visualize_predictions.py`) is extended to color
  waypoints by visible/obstructed and mark the goal green (visible) / red
  (obstructed), for both GT and prediction.

## Module / file layout

New:
- `src/rover_vlm/habitat_data.py` — load sample, select correct path, normalize, clip,
  resample+transitions, format answer/prompt, build splits. Mirrors the role of
  `data.py` for ShareRobot; kept separate rather than overloading `data.py`.
- `scripts/prepare_habitat.py` — CLI: dataset root → `data/prepared_habitat/`.
- `scripts/inspect_habitat.py` — overlay the clipped, visibility-colored path + goal
  on `fpv_enhanced.png` for N random samples → `outputs/inspection_habitat/`;
  validates the coordinate/visibility convention before any training.

Changed:
- `src/rover_vlm/eval.py` — add the object/triple parser + visibility/goal metrics
  (keep the ShareRobot parser working; branch on format or add a new function).
- `scripts/evaluate.py` — swap the prompt/eval-file defaults for Habitat; wire the new
  metrics.
- `scripts/compare_evals.py`, `scripts/visualize_predictions.py` — visibility-aware
  columns / overlays.
- `README.md` — a Habitat phase section.

Unchanged: `scripts/train.py`, `slurm/*.sbatch` (only args differ),
`scripts/download.py`.

Prepared data + all outputs stay in the gitignored `data/` and `outputs/`. Images are
read directly from the NFS source path — never copied into the repo.

## Verification

1. `scripts/inspect_habitat.py` on ~16 samples: the colored path hugs the floor toward
   the doorway/goal, visible segments on observed floor, obstructed segments beyond the
   occluder, goal marker green/red matches `goal.hidden`. Eyeball a mix of
   single-transition and multi-transition samples.
2. `scripts/prepare_habitat.py`: spot-check a prepared record — coords in [0,1], triples
   well-formed, answer parses back, eval split disjoint from every train subset,
   waypoint counts ≤ cap, transitions preserved.
3. Smoke train (`scripts/train.py --smoke`) on a Habitat subset: pipeline runs,
   adapter reloads.
4. Full loop on the cluster: base eval + LoRA scaling runs; `compare_evals` +
   explorer show the position error dropping and visibility accuracy rising with data,
   base vs. LoRA — the deliverable analogous to the ShareRobot scaling curve.

## Risks / open questions

- **Visibility-accuracy matching** (pred↔GT waypoint alignment) needs a single clear
  rule; nearest-point-along-arc-length is the plan, revisit if it proves noisy.
- **Multi-transition paths** (up to 6 flips) under a ~12 cap: transitions are
  prioritized; if a path genuinely needs more, log and clip (no silent truncation).
- **Goal x is constant (0.5):** the goal's informative signal is its visibility and
  vertical position; that's fine, but note the model gets an easy x — acceptable.
- **`fpv_enhanced` domain gap:** the enhanced (diffusion-upscaled) image differs from
  what the deployed rover sees; out of scope to address now, just noted.
