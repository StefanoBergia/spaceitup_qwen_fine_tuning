# Steering eval set — design

Date: 2026-10-07. Branch: `feat/instruction-paths`. Builds on the d1 route descriptions
(`2026-09-30-route-descriptions-design.md`, Addenda A–A5) and its eval (job 92602, adapter
`outputs/runs_desc/d1/adapter`). Eval-only: no training data is built here.

## Why

The d1 eval cannot tell whether the model *steers* — whether changing only the text changes the
route the way the text says. Measured on the 231 `eval_desc_pairs` (2026-10-07):

- both routes of every pair end at the same point (100% within 0.01 uv, the frame's goal);
- they diverge where the floor is hidden: max separation median 0.50 uv, but the visible part
  only 0.036 (> 0.1 in 37% of pairs);
- 47/231 pairs (20%) have identical event text — only the stop phrase differs;
- hence route A satisfies B's events 94% of the time, and shuffled text still "follows" events
  94.5% (matched 95.4%).

A steering test needs pairs whose inputs are identical except the text, and whose correct
routes differ *where the image shows it*, in a way each text rules out for the other.

## Decisions (user, 2026-10-07)

- **Two slices**, one pair-success number each:
  - **side** — same entry, same stop, the route passes a visible named object O on opposite
    sides (tests events: side word + landmark);
  - **destination** — same entry, two different stop targets (tests the stop, d1's failure
    today: 21% of entry-form stops within 0.03; and the x = 0.5 endpoint collapse).
- **Routes planned on CPU on the frame's own map** (`state.npz`), not re-planned in Habitat.
  The `habitat` conda env and `spaceitup-habitat` repo may be used (navmesh check below).
- **Text in two versions per route**: template (`descriptions.template_description`, minimal
  pairs) and composed (Cosmos compose, prompt d3, the d1 text rules). The gap between them
  separates "cannot steer" from "cannot steer through natural phrasing".
- Goal-form records of every route as a control.

## Success criteria (of the set, not of the model)

1. Gate 0 — yield: ≥ 150 pairs per slice, at most one pair per viewpoint per slice.
2. Gate 1 — human geometry audit, 50 pairs (25 per slice): ≥ 90% of pairs "true".
3. Gate 2 — composed text: ≥ 85% of the selected pairs keep both routes' composed text.
4. A text-blind model scores 0% pair success on both slices by construction (unit-tested).

## Facts the design rests on (measured 2026-10-07)

- Eval frames: d1's eval holds 2,336 records over 2,105 frames, 1,452 of the records from
  yaw-augmented frames. A yaw variant shares its parent's position, so pairs are deduplicated
  per **viewpoint** (`parent_sample_id`, else `sample_id`); the original frame is tried first.
- Frame maps: known free floor median 11.2 m² (p10 3.9, p90 21.8), reaching ~6 m ahead (95th
  pct). Grid 0.05 m/cell; ~99% of cells unknown.
- d1/round-4 GT routes keep 0.44–0.51 m (p10–p75) from map obstacles in the frame — the
  generator's 0.5 m rover radius. Only a median 59% of their in-frame points lie on known-free
  cells (the rest are hidden floor).
- Names: eval objects are keyed `<sample_id>:b<blob>`; blob labels come from the frame map,
  independent of the route. Mixed-namer policy (A5): Gemma name
  (`outputs/instr_labels_v3/names_eval_gemma_d1.jsonl`), else Cosmos
  (`names_eval_d1.jsonl`); 3,079 Gemma-named eval objects.
- Navmeshes: `/nfs/projects/spaceitup/rover_navigation/data/hm3d/{val,minival}/<scene>/*.navmesh`.

## Pair construction (`src/rover_vlm/steering.py`, CPU)

**Common to both slices.**
- *Start*: the floor point of the frame's d1 entry (the existing record's `start_uv_norm`,
  back-projected with the level camera). Both routes of a pair start there, so the entry point
  is identical and cannot cue the route.
- *Planner*: 8-connected A* on known-free cells only (unknown and occupied are not
  traversable), every cell ≥ `CLEAR_M = 0.45` m from an occupied cell, then string-pulled
  (line-of-sight shortcut on the same mask). Pure numpy (no scipy in `.venv`).
- *Route validity*: every densified point (0.05 m) projects in frame (`Camera.in_frame`) — so
  v = 1 throughout; length ≥ `MIN_ROUTE_M = 1.5`.
- *Navmesh check*: every densified point navigable on the scene navmesh with clearance
  ≥ `CLEAR_M` (`scripts/navmesh_check.py`, below). Catches low obstacles missing from the map.
- *Visible difference*: the two routes, resampled to 50 points by arclength in image space,
  differ by ≥ `MIN_SEP_UV = 0.1` at some point.
- *Cross-exclusion*: each GT route **fails** the other route's description under the eval
  checker (`eval.description_follow`, same tolerances), and passes its own.
- *Objects*: only blobs with a name under the A5 policy; walls (extent ≥ `WALL_EXTENT_M`)
  never anchor a pair. A name shared by another blob of the same frame (`name_clash`) drops
  the pair — the text would be ambiguous.

**Side slice.**
- O: a named non-wall blob ahead of the start, with known-free floor on both sides.
- S: a shared stop on known-free floor 0.5–1.5 m beyond O's far edge along start → O.
- Route L = A*(start → via_L → S), via_L on O's left at `CLEAR_M` + 0.1 m; route R likewise.
- Two-sidedness: O's cells lie inside the loop L + reversed(R)
  (`enclosed_obstacle_cells`-style even-odd fill), and **no other** occupied blob does.
- Events: `route_events` events + `stop_phrase` on each route. Keep the pair only when the two
  event lists are identical except O's side, and the stop phrases are identical — the two texts
  then differ in one side word.

**Destination slice.**
- O1, O2: two named non-wall blobs, each with an "in front of" stop point T on known-free floor
  (`stop_phrase` returns `near / in front of` with anchor = that blob).
- |u(T1) − u(T2)| ≥ `MIN_DEST_DU = 0.15` (horizontal: endpoints off the x = 0.5 line).
- Routes: A*(start → T_k). Events from `route_events` (may differ between the two routes —
  the destination is what the text changes).

**Candidates and selection.** `steer` plans every feasible candidate pair per frame (side: per
O; destination: per object pair) and writes them with geometry; after the navmesh verdicts,
`steer-select` keeps one pair per viewpoint per slice — the one with the largest visible
separation. Drop reasons are counted and printed per slice: `no_named_object`, `no_free_side`,
`no_stop`, `plan_failed`, `out_of_frame`, `too_close`, `not_two_sided`, `events_differ`,
`name_clash`, `not_exclusive`, `navmesh`.

## Navmesh check (`scripts/navmesh_check.py`, `habitat` conda env)

Standalone, the only file importing `habitat_sim`. Input: the `steer` candidates JSONL (scene
id, floor height, densified route xz per route). Loads each scene's `.navmesh` into a
`habitat_sim.PathFinder` (CPU, no renderer), checks `is_navigable` and
`distance_to_closest_obstacle ≥ CLEAR_M` at every point; writes `{route_id: ok, reason}`.

    conda run -n habitat python scripts/navmesh_check.py \
        --routes data/prepared_habitat_instr_v3/steer/candidates_eval.jsonl \
        --out data/prepared_habitat_instr_v3/steer/navmesh_eval.json

A scene whose navmesh is missing fails its routes with reason `no_navmesh` (counted, not
skipped silently).

## Records

Rows written by `steer-select` follow the d1 `events_*.jsonl` schema (`id, sample_dir,
events, stop, pair_id, ...`) plus `slice` and `route_xz`, so `descriptions.desc_check`,
`template_description` and `label_descriptions.py --stage compose` take them unchanged.
Objects come from the existing eval names; no naming stage runs.

GT answer: the route projected with `landmarks.Camera.uv`, clipped (`clip_polyline_unit`),
resampled (`resample_with_transitions`) and formatted (`format_answer_goal`), v = 1.

Per route, four records, in 8 files (`finalize-steer`), all under
`data/prepared_habitat_instr_v3/`:

    eval_steer_{side,dest}_{template,composed}_{entry,goal}.json

- entry form: `description_entry_prompt(start, text)` (headline);
- goal form: `instruction_endpoints_prompt(start, goal, text)` (control);
- `habitat_meta`: the d1 fields plus `slice`, `pair_id`, `sibling_id`, `steer_anchor`
  (side: O's id and side; dest: target id), `desc_check`.
- Composed: both routes of a pair share the speaking style and sentence-shape hint (hash of
  `pair_id`). A route whose compose fails the text rules after the retry drops its **pair**
  from the composed files only (no template fallback there).

## Metrics (`eval.py`)

All per slice, with 95% bootstrap CI over pairs.

| | headline: pair success | text-blind model |
|---|---|---|
| side | both predictions pass O on their own stated side (the O event of `description_follow` only; the stop is shared and excluded) | 0% |
| destination | each prediction's endpoint is closer to its own target than to its sibling's (ties fail) | 0% |

Secondary:
- strict pair success: full `description_follow` (all events + stop) at `STOP_TOL_UV` 0.03, and
  at 0.05;
- swap control: each prediction scored against its sibling's check (a steering model scores low);
- point error vs own GT, and the rate of predictions closer to own GT than to the sibling's GT;
- goal-form control: same metrics. On the side slice the goal is shared, so this is steering
  without the stop; on the destination slice it shows whether d1 can draw routes to off-centre
  endpoints at all.

Models: d1 (`outputs/runs_desc/d1/adapter`); round-4 LoRA on the goal-form files only
(instruction-blind control, expected 0%).

## Run order and gates

1. `uv run scripts/prepare_habitat_instr.py steer` → `steer/candidates_eval.jsonl` (CPU, here;
   `OMP_NUM_THREADS=4`, at most 4 single-threaded workers as for the events stage).
2. `conda run -n habitat python scripts/navmesh_check.py …` (CPU, here).
3. `uv run scripts/prepare_habitat_instr.py steer-select --navmesh steer/navmesh_eval.json`
   → `steer/events_steer_eval.jsonl`; prints yields and drop reasons. **Gate 0.**
4. `uv run scripts/audit_instruction_labels.py --mode steer` → audit page (artifact with a db
   `marks`, as earlier audits): 25 pairs per slice, both routes drawn on the frame with O / the
   targets outlined and both template texts. Marks per pair: true / wrong side / not visibly
   different / looks undrivable / wrong object / can't tell. **Gate 1** (≥ 90% true). The
   thresholds above are frozen after this gate.
5. `STAGE=steer sbatch slurm/label_descriptions.sbatch` — compose only, eval steering rows,
   existing names (~800 calls, < 1 h on a 3g.40gb slice). **Gate 2.**
6. `uv run scripts/prepare_habitat_instr.py finalize-steer` → the 8 files.
7. `SETS=steer sbatch slurm/eval_descriptions.sbatch` (d1), and
   `SETS=steer_goal ADAPTER=outputs/runs_v4/habitat_train_full/adapter TAG=r4 sbatch slurm/eval_descriptions.sbatch`.
8. `uv run scripts/visualize_steering.py` → report artifact; `uv run scripts/render_desc_videos.py
   --set eval_steer_{side,dest}_template_entry --tag d1` on a fixed 20-pair sample per slice
   (both routes and both predictions).

## Files touched

- new `src/rover_vlm/steering.py` — planner, pair builders, steering metrics' geometry;
- new `scripts/navmesh_check.py` — habitat env only;
- new `scripts/visualize_steering.py` — report;
- `scripts/prepare_habitat_instr.py` — stages `steer`, `steer-select`, `finalize-steer`;
- `scripts/audit_instruction_labels.py` — `--mode steer`;
- `slurm/label_descriptions.sbatch` — `STAGE=steer`;
- `slurm/eval_descriptions.sbatch` — `SETS=steer|steer_goal`;
- `src/rover_vlm/eval.py` — side / destination pair success, swap control;
- `README.md` — "Steering eval" subsection (index-only commit: the user's own README edits
  stay unstaged);
- new `tests/test_steering.py`.

## Testing (test-first, CPU, synthetic maps)

- planner: respects `CLEAR_M` and unknown cells; string-pulling keeps clearance; no path →
  `plan_failed`;
- side: a box obstacle in an open room yields L and R that enclose it; a via that bulges round
  the same side is rejected `not_two_sided`; a second obstacle between the routes →
  `not_two_sided`; event lists differing beyond O's side → `events_differ`;
- destination: two targets closer than `MIN_DEST_DU` → `too_close`; stop anchor not the
  target → `no_stop`;
- cross-exclusion: hand-made pair where one route satisfies both texts → `not_exclusive`;
- shared entry: both routes' first answer waypoint identical;
- selection: one pair per viewpoint, original preferred, largest separation wins;
- metrics: identical predictions for both texts → 0% pair success on both slices; GT
  predictions → 100%; swap control on hand-made paths;
- finalize: 4 records per route, same `pair_id`, composed drops whole pairs;
- navmesh_check: a fake PathFinder (no habitat import at test time) for the verdict logic.

## Out of scope

Steering training data; re-rendering or re-planning in Habitat; new naming runs; changing d1's
stop rules (open question 2 in NOTES.md — this set measures the stop, it does not fix it).

## Addendum A1 (2026-10-08): routes on the scene navmesh at 0.3 m, every visible object named

**Why.** Built as specified (plan Tasks 1–5, commit ff1c8ba), the frame-map planner kept **0 pairs** on
all 2,105 eval frames (no errors). Throwaway probes on 200 original frames
(`.superpowers/sdd/2026-10-07-steering-eval/progress.md`):
- the known-free floor piece the entry connects to is median 3.1 m² (p10 0.05 m²), and floor behind an
  object is never seen, so "a shared stop beyond O on known floor" almost never exists — 0 pairs even with
  every blob named and no minimum distance;
- treating unseen in-view floor as drivable (the generator's own `unknown_blocks=False`): 1 pair / 200;
- on the true scene navmesh (feasibility only, before the text filters): at 0.45 m with d1's names 6% side /
  1.5% dest frames; every visible object named 9% / 20%; **at 0.30 m with every object named 22% / 24%**
  (≈ 185 / 205 of the 845 viewpoints as an upper bound). d1 named only the objects on its own routes: 64% of
  eval frames have no nameable non-wall object.

**Decisions (user, 2026-10-08).** Routes come from the scene navmesh baked at `CLEAR_M = 0.30` m; every
visible non-wall object of the eval frames gets a name (one Gemma job). Everything not listed below is
unchanged: pair rules (`MIN_SEP_UV`, `MIN_DEST_DU`, `DEST_GAP_M`, `MIN_AHEAD_M`, `INSIDE_FRAC`, two-sided
enclosure on the frame map, events identical except O's side, identical stops, exclusivity, name clashes),
text versions, records, metrics, gates (Gate 0 still ≥ 150 pairs per slice; below it, back to the user).

1. **Clearance 0.30 m** (was 0.45): via points sit `0.30 + 0.1` m beside O, the navmesh is baked at agent
   radius 0.30. Steering routes pass objects tighter than training routes (≈ 0.48 m); the side metric only
   asks which side.
2. **Planning in three steps.** (a) `steer-propose` (uv, CPU) runs the pair builders with a planner that
   records every waypoint request (side: start → via → stop for each stop distance; destination: start →
   target) and writes them; (b) `scripts/navmesh_plan.py` (the `habitat` env, run with
   `~/miniforge3/envs/habitat/bin/python` — `conda run` picks up the repo's `.python-version`) snaps every
   waypoint onto the navmesh (≤ 0.3 m), joins one geodesic per leg and writes the routes; (c) `steer` (uv)
   runs the same builders with a planner that answers from those routes. The builders take the planner as
   an argument; unit tests inject the frame-map A* (plan Task 1) as a stand-in. Routes planned on the
   navmesh are navigable by construction, so the separate navmesh check is dropped.
3. **Routes may cross unseen floor and leave the view**, like d1's: the answer is the route densified at
   5 cm, projected, `v = 0` where the frame map has an occupied cell between camera and floor point
   (`route_events.first_blocker`), clipped to the image, resampled with transitions. A route passing less
   than 0.3 m in front of the camera is dropped (`behind_camera`).
4. **Naming every visible object.** `steer-objects` (uv, CPU) writes an object row (as d1's: outline, crop,
   image) for every eval-frame blob that is shorter than `WALL_EXTENT_M`, has ≥ `MIN_VISIBLE_CELLS`
   visible cells and an outline ≥ `MIN_AREA`, and is not named yet; `STAGE=steer_names sbatch
   slurm/label_descriptions.sbatch` names them with Gemma (`label_descriptions.py --stage name
   --all-objects`). Names: primary = d1 Gemma ∪ steer Gemma (disjoint ids), fallback = d1 Cosmos. Name
   quality is judged in the Gate 1 audit ("wrong object").

**Run order (replaces the spec's).** `steer-objects` → `STAGE=steer_names` (GPU) → `steer-propose` →
`navmesh_plan.py` → `steer` → `steer-select` (Gate 0) → audit (Gate 1) → `STAGE=steer` compose (Gate 2) →
`finalize-steer` → eval → report.

## Addendum A2 (2026-10-08): one slice of stop pairs; the side slice is not viable on these frames

**Why.** Built per A1 (plan Tasks 6–8), the navmesh pipeline still keeps almost nothing. On 200 original
eval frames, with a stand-in name for every nameable object and routes planned on the real navmesh at
0.30 m: **0 side pairs and 1 destination pair** (side: vias/stop off the navmesh, nothing else between the
routes; destination: named objects < 1 m from the entry, so the stop point falls behind it). Lowering
`MIN_AHEAD_M` to 0.5 changes nothing. The generator's own two-detour pairs (d1's 231 eval pairs) have a
mapped object between the two routes in only 15 cases (1 with exactly one named object). What does work:
**cutting one ground-truth route at two points with different stop phrases** — 111 / 200 original frames
(55%), ends a median 0.18 apart in the image, each route failing the other's text. Probes:
`.superpowers/sdd/2026-10-07-steering-eval/progress.md`.

**Decisions (user, 2026-10-08).** The set has one slice, **stop**: same entry, same route, two stop points
whose texts differ (in the stop phrase, and in the events the longer route adds). The side slice is dropped
(it would need purpose-built Habitat frames). The side-slice and navmesh-planning code is removed (git
history keeps it: commits 85c2727..2c29b6a). Everything else from the spec and A1 stands: every visible
object is named (A1 §4), text versions, records, exclusivity, name clashes, metrics (the destination rule),
gates — Gate 0 is ≥ 150 pairs for the one slice.

**Pair construction (replaces "Pair construction" and A1 §1–3).**
- Route: the frame's d1 ground-truth candidate polyline (`meta.candidates[candidate_index]`, planned by the
  generator on the navmesh — drivable by construction).
- Cuts at shares `CUT_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 1.0)` of its length; a cut must keep ≥
  `MIN_ROUTE_M = 1.5` m. Each cut is described from the rover's pose (`route_events.describe_xz`, as d1's
  routes), and its answer is the part from where it reaches the image's bottom edge, built as in A1 §3.
- A pair = two cuts whose stop phrases differ (`stop_key`), whose answer ends are ≥ `MIN_SEP_UV = 0.1` apart,
  and which are mutually exclusive under `eval.description_follow`; tags A (earlier stop) / B. One pair per
  viewpoint, the original frame first, then the largest end separation.
- Names: the A5 rule per route; a stop object whose name's head noun is shared by another object of the frame
  drops the pair; two stops at the same object ("just past" vs "well past the chair") are allowed.

**Files.** `eval_steer_stop_{template,composed}_{entry,goal}.json` (4 files). Metric: the destination rule
(prediction's end closer to its own route's end than to its sibling's), text-blind = 0%.

**Run order (replaces A1's).** `steer-objects` → `STAGE=steer_names` (GPU) → `steer` → `steer-select`
(Gate 0) → audit (Gate 1) → `STAGE=steer` compose (Gate 2) → `finalize-steer` → eval → report.
