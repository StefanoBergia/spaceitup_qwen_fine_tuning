# The pinned endpoint: why the Habitat models cannot draw a curve (2026-09-07)

Found while analysing SLURM job **90936**, the benchmark-clip eval re-run under the new
`--goal-from-path` label rule (632 TUM frames, 12 clips, `plain_2b` + `traced_2b`).

## The observation

Every prediction on the real clips ends at exactly **x = 0.500** — min = max = 0.500 across
all 632 samples, both models — while real ground-truth endpoints have sd 0.129 and sit a mean
0.11 off centre. `shape_correlation` agrees the answers are uninformed by the picture: plain
−0.017, traced +0.132 overall, and on the `curve` clips specifically plain is **−0.114**,
i.e. bending *against* the true route.

The models therefore fail the one thing the curve clips exist to test, and both lose to the
image-blind floors from `trivial_baselines`:

| | mean point error | median |
|---|---|---|
| set-mean constant path | **0.169** | **0.164** |
| straight line up the middle | **0.185** | **0.174** |
| `plain_2b` | 0.254 | 0.215 |
| `traced_2b` | 0.253 | 0.221 |

Per-sample, a fixed centre line beats both models on 322/632 frames (51%) and on 8 of 12 clips.

## The causal chain

1. **The generator yaws the camera onto the goal.** Raw
   `/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/.../fpv_paths.json`
   carries `goal.uv = [256.0, …]` on a 512-wide render — the goal is on the optical axis by
   construction, not by chance.
2. **So every label is pinned.** All **9,140** prepared Habitat samples
   (`data/prepared_habitat_v2/`: 8,140 train + 1,000 eval) have `goal[0] == 0.5` *and*
   `path[-1][0] == 0.5`, sd **0.00000**. Note the path *x* itself varies fine (sd 0.235) — only
   the terminus is fixed, so the task as posed is "bend however you like, then funnel into the
   frame centre".
3. **LoRA learns it perfectly.** The base model predicts `path_end_x` with sd **0.2951**
   (34% at 0.5); after fine-tuning on Habitat it is sd **0.0000** (100% at 0.5). The variation
   was there and fine-tuning destroyed it. This is learned, not a decoding artifact — it is
   already near-total at the smallest subset and complete by 2K samples: `train_500` 0.0197,
   `train_1000` 0.0265, `train_2000` 0.0000, `train_full` 0.0001.
4. **Real routes are then unreachable.** A frame whose true route ends off-axis cannot be
   answered correctly by a model whose output distribution has no mass off x = 0.5.

The prompt itself says "The goal is located straight ahead", which is *true* in Habitat and
false on real frames — the same contradiction the `--goal-from-path` rule fixed on the label
side. Fixing the label made the numbers look better (median point error 0.362 → 0.215) without
changing the verdict, because the model side of the contradiction is still there.

## How to check it

`endpoint_spread` in `src/rover_vlm/eval.py`, alongside `trivial_baselines` and
`shape_correlation`. It reports `pred_end_x_sd` / `pred_end_x_mode_frac` against the same two
numbers for the ground truth; near-zero prediction spread against a spread ground truth is the
collapse. `scripts/visualize_real_eval.py` surfaces it as the "Endpoint x spread" column and an
`endpoint pinned` transfer chip.

| predictions | `pred_end_x_sd` | mode frac |
|---|---|---|
| `outputs/eval_real/real_clips/{plain,traced}_2b` | 0.0000 | 1.000 |
| `outputs/eval_habitat/habitat_base` | 0.2951 | 0.344 |
| `outputs/eval_habitat_v2/habitat_train_full` | 0.0000 | 1.000 |

## What would fix it

Not decided — scoped only.

- **Re-render with camera-yaw jitter.** The correct fix, but the generator source is not in this
  repo and not under `/nfs/projects/spaceitup/rover_navigation/` (only rendered data is there),
  so it needs whoever owns it.
- **Horizontal shift/crop augmentation at prepare time.** Shift the existing 512×512 renders and
  re-project every coordinate, using data already on disk. The shift must be bounded per sample
  by that sample's own path headroom: only 59% of samples keep their whole path inside the
  central 80% of the frame, and 35% inside the central 60%, so a global shift would clip paths.
