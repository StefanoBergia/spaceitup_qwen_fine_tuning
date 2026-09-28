# Grounded landmarks for the v2 instruction dataset — design

Date: 2026-09-28. Branch: `feat/instruction-paths`. Supersedes the landmark part of
`2026-09-25-instruction-general-model-design.md` (prompt i3); the rest of v2 (plain copies,
labelsets, finalize-general, eval files) is unchanged.

## Why

A 100-label human audit of the v2 training labels (prompt i3, 2026-09-28; marks in the
artifact https://claude.ai/artifact/6d5JYwW2QvQQmJ1ZwMW3wC, db collection `marks`) found
~20% wrong, population-weighted:

| record type | wrong |
|---|---|
| pair, planner's chosen side | 1 / 16 |
| pair, **other side** (`kind == "opposite"`) | **15 / 20** |
| single-route frame | 11 / 64 |

Object *naming* was good (6 landmark errors). The dominant error: the named object is real
but is **not the one the route goes around**, plus routes that bend both ways. Choosing *which*
object a drawn route circumvents is the geometric judgment VLMs failed at in v1 (Cosmos
verifier at chance). A second cause: **93% of v2 records are `doorway` scenarios** (the
straight entry→goal line hits a wall; the route reaches the goal room through an opening),
while the i3 prompt treats every case as "name the thing the route passes around".

## Goal and success criteria

The target model must follow instructions that **name things** ("go left around the sofa",
"take the doorway on the right"), so each instruction's referent must be the right one.

- Fresh human audit of the new labels: **< 5% wrong overall and < 5% wrong on pair records**
  (both sides).
- On scenes with HM3D semantics: report the rate at which the instruction's name agrees
  with the HM3D category of the object under the target (no fixed threshold; mismatches
  are listed for review).
- Every record that gets no instruction is counted under a reason code; nothing is dropped
  silently. Dropped records keep their plain round-4 copy in training.

Out of scope: training and evaluating the model (after the dataset passes), changes to the
Habitat generator repo, new scenarios.

## Conventions (unchanged)

- The side word is the route's side: "left" = the route bows left of the straight
  entry→goal line = passes the obstacle on its left, or uses the opening on the left.
  The side is taken from the candidate metadata, never from a VLM.
- Egocentric "on/to your left|right" is rejected (`side:egocentric`, added 2026-09-28).

## Design

### 1. Target selection — `src/rover_vlm/landmarks.py` (new, pure numpy, CPU)

Inputs per record, all already on disk next to the sample: `state.npz` (the rover's own
occupancy map from this frame: 0 unknown / 1 free / 2 occupied, `mpp`, `x_min`, `z_min`),
`meta.json` (`pose_xz`, `heading_xz`, `chord` incl. `crossing_xz`, `floor_height`,
`flavor`, candidate `polyline_xz`), `fpv_paths.json` (image size, per-run hidden flags).

Camera: level pinhole, hfov 90°, height 0.8 m above the floor, at `pose_xz` facing
`heading_xz`; image right = `(-h_z, h_x)`. Verified 2026-09-28: projecting
`chord.goal` reproduces the stored goal pixel (256.2, 282.0) vs (256.0, 282.0). Reuses
`rover_vlm.projection` primitives where they fit.

For the record's route (its candidate polyline):

**Revised 2026-09-28 after the geometry audit failed (59/98 right; openings 8/23; 16 of 28
dropped records had an obvious obstacle). User decision: objects only.**

1. **Object**: the visible obstacle the route passes closest to (within 1.0 m, past its first
   0.3 m) that lies on the route's *inner* side at that point — toward the straight chord,
   i.e. what the route bends around. This is exactly "the route passes it on the side its
   label says", so the metadata side word is true of the named object even on routes that
   bend both ways; an obstacle passed the other way round is never named. An obstacle
   counts as visible when part of its floor-to-1.0 m column is in frame and the camera ray
   meets it first (close obstacles have their floor below the image). The outline is the
   convex hull of its visible cells within 1.5 m of the route (a long wall is outlined only
   where the route passes it), floor to 1.0 m.
2. **None**, with a reason code: `no_visible_target`, `target_too_small`, `bad_candidate`,
   `no_state`.

(The first version's opening search, `route_crosses_chord` and `both_sides` drops were
removed: the audit showed users name the object, not the doorway, and most two-way routes
have a nameable obstacle on the label's side.)

Output `data/prepared_habitat_instr_v2/landmarks_{train,eval}.jsonl`, one row per labelset
record: `id`, `target` (`opening` | `object` | `none`), `reason`, `bbox` (normalised),
`mask` (polygon in normalised image coords), `world_xz` (target centre), and diagnostics
(gap width, blob area, visible fraction).

Pairs: the two routes of a pair usually pass different obstacles or the same one on
opposite sides; each record's instruction names what its own route passes.

### 2. Naming and sentence — prompt i4 (`instructions.py`, `label_instructions.py`)

One Cosmos3-Nano call per record (same serving/labelling job as v2). The image shows only
the target: a bright contour on the target, the rest slightly dimmed; **no route drawn**.
Cosmos gets the side and a template sentence skeleton for the target type, and returns:

```json
{"name": "grey sofa", "beside": "fridge", "single": true,
 "instruction": "<paraphrase of the template sentence>"}
```

- `name`: 1–5 words, as a person would say it; `beside` (openings only, optional): the most
  prominent object next to the opening; `single`: false when the outline covers more than
  one thing → drop (`mask_ambiguous`).
- Templates (about 8 per type, chosen by a seeded hash of the record id), e.g. object:
  "Go {side} around the {name}.", "Pass the {name} on the {side}."; opening: "Take the
  {name} on the {side}.", "Go through the {name} to the {side} of the {beside}."
- Text rules on the paraphrase (extends `instruction_rejects`): side word present,
  opposite absent, no egocentric phrasing, no drawing references, the head noun of `name`
  present. A failing paraphrase falls back to the filled template (`source: "template"`),
  so a good name is never lost to a bad paraphrase.
- `INSTRUCTION_PROMPT_VERSION = "i4"`; label files `outputs/instr_labels_v2/*_i4.jsonl`.

### 3. HM3D name check — standalone semantic replay (new script, Habitat env)

`objects.json` lists categories but not where objects are, so the check needs a per-pixel
instance image. New script `scripts/render_semantic_frames.py` (runs in the `habitat` conda
env, imports the generator's `roverbench.sim` from `spaceitup-habitat/src`, never writes
into that repo or into the shared dataset dirs): for each labelset record in a scene with
HM3D semantics, replay one SEMANTIC frame at the sample's pose and save
`outputs/landmark_semantics/<sample>.npz` (instance ids + id→category). sbatch
`slurm/render_semantic_frames.sbatch` targets the **full A100** (`gpu:a100:1`; MIG slices
cannot render); ~6.3k records in ~181 scenes.

The check (`audit_instruction_labels.py semantic`, CPU): the category covering most of the
target mask (walls/floor excluded for objects; for openings the `beside` object is checked
instead) vs the i4 `name`, via a small synonym table (`couch`=`sofa`, …). Reports agreement
rate by target type and kind, and writes the mismatches for review.

### 4. Pipeline and gates

1. `prepare_habitat_instr.py landmarks` (CPU, minutes) → reason-code counts → **geometry
   audit page** (route + target outline, ~100 records incl. pairs). Fix target selection
   here before any GPU time. Pass: target is the right thing in ≥ 95%.
2. `render_semantic_frames.sbatch` (A100, ~30–60 min) — can run in parallel with step 1's
   review.
3. i4 gate labelling (~300 records incl. pairs, the unattended serve+client sbatch) →
   HM3D agreement + human audit (updated audit page). Pass: the success criteria above.
   If Cosmos naming misses the bar, compare Gemma-3-12B (cached) on the same gate set.
4. Full i4 labelling (~8 h) → `finalize-general` (reads i4 rows; unchanged otherwise).

### 5. Testing

pytest stays the CPU gate. Unit tests on synthetic occupancy grids: opening found in a wall
with a gap; object chosen between route and chord; `both_sides` detection; hidden /
out-of-frame targets dropped with the right reason. One real-sample test: the projection
reproduces the stored goal pixel. Text-rule tests for i4 (head noun required, template
fallback). The semantic replay script is exercised on the GPU only (as for training).

## Risks

- The occupancy map is partial (only what this frame saw); flanks of an opening may be
  unobserved → fewer openings found, more fall back to objects or drop. Reason counts make
  this visible at step 1.
- Walls as "objects" ("go left around the wall") are weak landmarks; if they dominate the
  object fallback, exclude border-connected blobs from the object step.
- Dropping records shrinks the instructed set; acceptable — quality over count. Pair
  counts before/after are reported.
