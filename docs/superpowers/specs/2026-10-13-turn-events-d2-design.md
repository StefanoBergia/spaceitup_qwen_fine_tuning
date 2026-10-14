# Turn events, spur routes and the d2 dataset — design

Date: 2026-10-13. Branch: `feat/turn-events` (off local `main` 2bfed97). Builds on
`2026-09-30-route-descriptions-design.md` (d1, Addenda A–A5) and `2026-10-07-steering-eval-design.md`
(stop pairs, Addendum A2). Produces a new labelling round (d2), a retrained adapter and a rebuilt
steering set. Probe numbers below come from throwaway scratchpad scripts. Results page:
https://claude.ai/artifact/4fJ1sMkuwBcd6XdRfV9vjr

## Why

d1 descriptions never say "turn left / turn right". `route_events.build_events` only emits pass / wall /
between / through events. `descriptions.compose_prompt` forbids extra directions, and `_side_rejects`
rejects any unowned left/right. The user found it on the steering stop pairs: route B continues past A's
stop round a corner, and B's text only moves the stop.

**Measured (2026-10-13).** θ(s) is the route heading taken from a 0.5 m chord, relative to the
start→goal line (left/right = the rover's own).
- The obvious metrics saturate. Net turn ≥ 30° holds on 90% of routes and the largest 1 m local turn
  on 98%: every frame is a decision point with the straight way blocked, so the route leaves the pose a
  median 41° to one side and comes back. That detour is what pass / wall / between already describe.
- A **turn** = the route *ends* ≥ T off its start→goal line. On original frames the line is the camera
  axis (median 0.1°). Measuring against the camera instead double-counts yaw-augmented frames
  (26.5% vs 11.8% at 45°; against the line, 12.4% vs 11.7%).
- d1 train (spurs set aside, n = 20,554): ≥ 30° 20.3%, **≥ 45° 8.2% (1,687)**, ≥ 60° 4.0%, ≥ 90° 1.4%.
  Eval (2,124): 21.7 / 7.3 / 3.7 / 1.3%. No text says "turn" (0 of 1,687).
- Where turns happen:
  - 74% (train) / 88% (eval) swing over in a stretch hidden behind an obstacle, at ~70% of the route
    and ~5.5 m ahead.
  - ~20% are plain turns. The rest first swing ≥ 15° the other way round the obstacle the text names,
    then turn further.
  - 805 of the 902 turns with a wall event keep that wall on the inner side.
- `no_events_bent` drops (2,861 train / 458 eval): gentle 44 / 38%, swerve 37%, plain turn 3%,
  turn after a swing 8 / 10%, spur 5 / 8%.
- Steering (677 pairs): 51 B routes turn (same definition), and A does not in 44 of them. The 23.5%
  first reported counted any A→B heading change, swerve returns included. Measured on cut routes, the
  definition also fires on 76 A routes that stop mid-swerve, so turns must be detected on the full route.
- d1 error, eval (`outputs/eval_desc/eval_desc_{entry,goal}/d1`):
  - Entry form, turn vs gentle + swerve: point +0.025 [95% CI +0.005, +0.040], endpoint +0.021
    [+0.003, +0.060].
  - Goal form: +0.001 [−0.002, +0.003].
  - The cost is in turn-after-swing routes: entry endpoint 0.120 vs 0.054 / 0.077.
  - The prediction bends the right way on 127 of 147 turns, but not far enough.
- Anchors (153 non-spur eval turns):
  - 128 are hidden. In 120 of them the blob hiding the turn point is nameable: 105 are wall-like, and
    108 are already an object in the same route's text.
  - 25 are visible: 7 have a nameable blob within 1 m, mostly on the inner side.
  - In all, 127 / 153 (83%) have an anchor.

**Spur bug (found by the same probe).**
- 1,772 of the 2,026 d1 train "opposite" routes (87%) contain an out-and-back spur: the route drives to a
  via point, reverses ≥ 150° and runs back along the same line, then heads for the goal. All 212 eval
  spurs are opposite routes; 3 instructed (GT) train routes have one.
- 161 of the 212 repeat a vertex exactly (P[k−1] = P[k+1]); the rest are near-mirror retraces or
  V-spikes.
- d1 trained on them, and the d1 eval pairs are built on them. Even with the goal given, d1's point
  error on them is 0.232 vs ~0.016 on other routes.

## Decisions (user, 2026-10-13)

1. Describe **every** turn, seen or hidden, and name an object: "turn right behind the table".
2. A turn with no nameable object → bare **"turn left"** (keep the route).
3. Detector **A**: one final turn per route, **T = 45°**, detected on the full GT route; a cut keeps it
   only when finished.
4. **Drop spur routes** (revised 2026-10-13 after planning measurements, see "Spur routes"; first
   chosen: cut the spur). The fix is part of d2.
5. **Recompose only changed routes**; unchanged routes keep their d1 text.
6. Steering: **turn cuts + a `turn` slice**; Gate 1 redone, stratified.
7. A **new spec** (this file) rather than a sixth addendum: the change spans the descriptions and the
   steering specs, adds a spur fix and a retrain, and the descriptions spec already has five addenda.
   Both older specs get a dated pointer here.

## Success criteria

- G1 geometry audit (human, 100 routes: 50 turn routes ≈ 20 visible / 30 hidden, 20 rescued bent
  routes, 30 routes dropped as `spur`: is dropping them right): **≥ 90% true**.
- G2 label gate (300 changed eval routes): accept ≥ 0.90, template share ≤ 0.10.
- G3 label audit (human, 100 changed routes, half with a turn): **≥ 90% correct**.
- Steering: Gate 0 ≥ 150 pairs and ≥ 50 in the turn slice; Gate 1 (50 pairs, 20 turn + 30 stop)
  ≥ 90% true; Gate 2 ≥ 85% of pairs keep both routes' composed text (unchanged from the steering spec).
- d2 model (v4 eval sets):
  - On turning routes in the entry form, d2 beats d1 on turn follow rate and on endpoint / point error,
    with the 95% CI of the difference excluding 0.
  - On non-turn routes, no regression (CI includes 0 or favours d2).
  - Overall follow rate and pair success ≥ d1's.
  - Steering: d2 − d1 pair success on the turn slice reported with its CI.

## Turn geometry (`src/rover_vlm/route_events.py`)

Constants: `TURN_MIN_DEG = 45`, `HEADING_CHORD_M = 0.5` (heading of a point from the chord ±0.25 m around
it), `TURN_END_WINDOW_M = 0.5`, `TURN_HOLD_DEG = 15`, `TURN_ANCHOR_NEAR_M = 1.0`.

**`detect_turn(route_xz) -> dict | None`** (pure; full route):
- θ(s) = heading at arclength s relative to the start→goal line, + = right (Camera.right handedness),
  unwrapped.
- e = mean θ over the last `TURN_END_WINDOW_M`. No turn when |e| < `TURN_MIN_DEG`.
- **turn point**: the last crossing of e/2 (the first point after which θ stays on e's side of e/2).
- **start**: the last point before the turn point with |θ − θ_pre| ≤ `TURN_HOLD_DEG`, where θ_pre = θ
  1 m before the turn point. **end**: the first point after it with |θ − e| ≤ `TURN_HOLD_DEG`.
- Returns `{side: "left"|"right", angle_deg: e, arc_m, start_arc_m, end_arc_m, xz, pre_heading_xz}`.

**Anchor** (`turn_anchor(frame, turn)`):
1. The turn point is hidden (`first_blocker(frame, xz)` is a blob) and that blob is nameable
   (≥ `MIN_VISIBLE_CELLS` visible cells, `object_outline` area ≥ `MIN_AREA`) → relation **behind**.
   Wall-like blobs are allowed.
2. Else the nearest nameable blob within `TURN_ANCHOR_NEAR_M` of the turn point on the turn's inner side
   (in the pre-turn heading's frame) → relation **round**.
3. Else no anchor (bare turn).

**Event.** `{"type": "turn", "side", "relation": "behind"|"round"|None, "objects": [oid] or [], "arc_m",
"turn_xz", "pre_heading_xz"}`. The anchor's object row gets `kind_hint` "wall" (extent ≥
`WALL_EXTENT_M`) or "object", never "opening". It reuses an existing row when the blob is already an
event or stop object.
- Placed in arc order among the other events; always kept (top-`MAX_EVENTS` obstacle events + the
  turn, ≤ 4).
- Never merged into between / through.
- `stop_phrase` skips turn events: "just after" / "past" refer to the last obstacle event, as today.

**Cuts.** `describe_xz(meta, frame, route, turn=None)`:
- `describe_route` passes `detect_turn(route)` of the full route.
- Steering passes the full route's turn for every cut, and a cut keeps the turn only when its length
  ≥ `turn["end_arc_m"]`.
- `describe_xz` never runs `detect_turn` on a cut itself.

**Zero obstacle events.** A route whose only event is a turn is described only when its stretch from the
start to `start_arc_m` stays within `STRAIGHT_MAX_M` (0.3 m) of its own chord; otherwise `no_events_bent`
as today. The template opens such a route with "go ahead, then …".

**Schema.** `EVENTS_SCHEMA = "d2-turns"`. The labeller, the audit and `steer` refuse other schemas.

## Spur routes (`route_events.has_spur`; dropped at the events stage)

**Why drop, not cut (measured while planning, 2026-10-13).** Cutting the out-and-back leaves the overshoot
it sits on: the route drives past the goal to the branch point, and the spur and the way back meet there
in a hairpin. On the 212 eval spur routes:

| after cutting the spur | plain collapse of the exact retrace | mirror walk 0.3 m, midpoint join |
|---|---|---|
| still on its labelled side | 86% | 42% |
| still has a turn ≥ 120° within 0.6 m | 98% | ~46% |
| usable (both OK, distinct from GT) | 0% | 37 / 212 (17%) |

Normal GT routes have a ≥ 120° turn within 0.6 m in 1 of 300; the largest-offset side matches the label
on 300 / 300 instructed and 70 / 70 non-spur opposite routes, so both measures are sound. These
opposite routes are "drive to the other side and come back". Cutting would add at most ~37 eval pairs, at
the cost of cutting the generator's fpv image path. The d1 eval pairs were never a steering test (route A
satisfies B's events 94% of the time); the steering stop pairs, built from GT routes (no spurs), are.

**`has_spur(route_xz) -> bool`**: the chord heading (`heading_profile`, the same 0.5 m chord as turns)
turns by ≥ `SPUR_REVERSAL_DEG = 150` within `SPUR_CHECK_M = 0.6` m of arclength. This is the probe's
detector: 1,775 d1 train routes (1,772 of them opposite) and 212 eval routes (all opposite). Normal routes
reach at most ~100° within 0.6 m.

**Rule.** `describe_xz` returns reason `spur` (added to `REASONS`) for a route with a spur, before anything
else. Applied to every route, so the 3 GT train routes with a spur go too. A dropped opposite route leaves
its GT sibling as a single record: finalize already keeps only pairs with both routes. Steering cuts come
from GT routes; a cut of a spur route would be dropped the same way.

No new stage and no change to records: `events` (and `finalize-descriptions`) read the v2 labelsets
through a `--labelset-dir` argument, so v4 holds only what d2 produces.

## Text (`src/rover_vlm/descriptions.py`)

- `event_phrase`: "turn {side} behind the {X}" / "turn {side} round the {X}" / "turn {side}".
  `template_description` opens with "go ahead, then" when the first event is a turn.
- `compose_prompt`, `PROMPT_VERSION = "d4"`. One rule added, keyed on the step's wording: *in a 'turn
  left / turn right' step, keep 'turn left' / 'turn right' (the rover's own turn) and keep its 'behind' /
  'round' and its object; never add a turn the steps don't have.* "Add no other objects, sides or
  directions" stays.
- `description_rejects` / `_side_rejects`:
  - A side word right after `turn|turns|turning` (optionally "to the" / "to your") belongs to the turn
    event, never to the next name. "turn to your left" is not `side:egocentric`.
  - A turn event needs its direction (`turn:side:missing`) and no opposite (`turn:side:opposite`).
  - A turn phrase in a text without a turn event → `turn:stray`.
  - An anchored turn needs its relation word in its clause (`behind`; `round|around`) and its anchor name
    in route order. A pronoun (`it`) is accepted when the anchor is an object of the step just before.
  - Everything else unchanged.
- Naming: only ids not named in d1 (`names_{split}_gemma_d1.jsonl`, `names_{split}_d1.jsonl`) go through
  the name stage, with the A5 mix rule.

## Data flow and run order

All d2 data goes in `data/prepared_habitat_instr_v4/`, labels in `outputs/instr_labels_v3/*_d2.jsonl`,
the adapter in `outputs/runs_desc/d2`. d1 files (v2 events, v3 dataset, d1 labels, d1 adapter) are not
touched.

| # | Step | Where | Output |
|---|---|---|---|
| 1 | `events --labelset-dir v2 --out-dir v4` | CPU, 4 workers, ~40 min | `events_{split}.jsonl`, `objects_{split}.jsonl` (schema d2-turns) |
| 2 | `d2-seed` | CPU | label files seeded with what d1 already has (below); `events_eval_changed.jsonl` for the gate; counts |
| 3 | `ROUND=d2 DATA=v4 NAMER=mix sbatch slurm/label_descriptions.sbatch` (gate), then the same with `STAGE=full` | GPU | names of new objects, compose of changed routes, all with `--resume` |
| 4 | `finalize-descriptions --round d2` | CPU | v4 `train_desc.json`, `eval_desc_{entry,goal,pairs,shuffled}.json` |
| 5 | `slurm/train_habitat.sbatch` (d1 settings, 2 epochs) | GPU, ~6 h | `outputs/runs_desc/d2/adapter` |
| 6 | `DATA=v4 OUT_DIR=outputs/eval_desc_d2 slurm/eval_descriptions.sbatch`, d1 and d2 | GPU | metrics, report, path videos |

- **Seeding** (step 2). The label job resumes by id and skips ids already in its output file. `d2-seed`
  therefore writes the d2 label files pre-filled with d1 rows that still hold:
  - `names_{split}_gemma_d2.jsonl` / `names_{split}_d2.jsonl`: every d1 Gemma / Cosmos name row whose
    object id is in the v4 objects file;
  - `desc_{split}_mix_d2.jsonl`: the d1 desc row of every **unchanged route** (same route, same events
    (type, side, objects) and same stop as its d1 events row), tagged `"reused_from": "d1"`.
  The label job then names only new objects and composes only changed routes. The gate (`--limit 300`)
  runs on `events_eval_changed.jsonl`, the changed eval routes, so it judges d4 compose only.
- Finalize re-checks every text (reused or new) against the d2 rules and drops any that fail (counted).
- **Cost.** Cosmos composes ~14 routes/min. ~2–2.5k changed routes (turns ~1.8k train + rescued bent
  routes + eval) ≈ 3 h, plus a short Gemma naming pass. `d2-seed` prints the counts before any submit.
- SLURM: at most 3 jobs at once; every GPU submit is asked for first.

## Eval (`src/rover_vlm/eval.py`)

- `desc_check` carries each turn event's `turn_xz`, `pre_heading_xz`, `side`.
- `_event_followed` for a turn: the predicted floor path's end lies ≥ `TURN_FOLLOW_M = 0.5` m to the
  stated side of the GT pre-turn line (through `turn_xz` along `pre_heading_xz`). Stopping short or
  going straight on fails. `gt_follow_rate` gives the ceiling.
- `aggregate_description_metrics` adds `by_turn`: for records with / without a turn event, n, follow rate,
  endpoint and point error medians. `event_rate_by_type` gains "turn".

## Steering (`src/rover_vlm/steering.py`, stages `steer` / `steer-select` / `finalize-steer`)

- Source: the d2 eval GT routes. Turns are detected once per route (`detect_turn`).
- Cuts: `CUT_FRACTIONS` as today, plus for a turning route two turn cuts. Both must be ≥ `MIN_ROUTE_M`.
  - **A_turn** at `start_arc_m − 0.3` m.
  - **B_turn** at min(`end_arc_m + 1.0` m, route end).
- Every cut is described with `describe_xz(…, turn=<full-route turn>)`.
- A pair needs different **texts**: stop phrase OR events. The other rules are unchanged: ends ≥
  `MIN_SEP_UV`, mutually exclusive under `description_follow` (with the turn check).
- One pair per viewpoint: a valid turn pair (A_turn, B_turn) is the frame's pair (`slice: "turn"`), else
  today's rule (`slice: "stop"`).
- Names: `STAGE=steer_names` only for ids still missing, wall-like turn anchors included.
- Gate 0: ≥ 150 pairs and ≥ 50 turn pairs (expected ~80–120). Gate 1: new 50-pair audit, 20 turn +
  30 stop, seed 0; the paused 4/50 audit is abandoned. Gate 2: compose with d4, ≥ 85% of pairs keep both
  composed texts.
- Eval: `SETS=steer` for d1 and d2 (+ goal-form control); pair success overall and per slice
  (`audit_instruction_labels.py --mode steer` stratified, `visualize_steering.py` per slice).

## Testing (test-first, CPU, synthetic `state.npz` grids)

- `tests/test_route_events.py`:
  - `detect_turn`: L-route left/right (sign), swerve → None, swing-then-turn, 44° vs 46°.
  - Cut before / after `end_arc_m`.
  - Anchors: behind / round / bare.
  - Turn as first event; top-3 + turn; straight lead-in rescued vs swing kept as `no_events_bent`.
  - `has_spur`: exact retrace and a near-mirror spike → True; an L-turn, a swerve and a 120° bend → False;
    `describe_xz` on a spur route → reason `spur`.
- `tests/test_descriptions.py`:
  - The three template forms and "go ahead, then".
  - `turn:side:missing/opposite/stray`, missing relation, pronoun anchor accepted.
  - "turn left behind the sofa" gives the sofa no side.
  - A reused d1 text passes d2 rules.
- `tests/test_eval.py`:
  - Turn check: straight fails, turned passes, short fails.
  - `by_turn` split.
- `tests/test_steering.py`: turn cuts, turn-pair preference, texts-differ rule, slice field.
- Stage tests: `events --labelset-dir`, `d2-seed` (reuse vs compose, seeded names, changed-eval file),
  `finalize-descriptions --round d2` (re-check drops a text that breaks the d2 rules).

## Files touched

- `src/rover_vlm/route_events.py`: heading_profile, detect_turn, turn_anchor, turn event, has_spur,
  describe_xz(turn=),
  schema.
- `src/rover_vlm/descriptions.py`: event_phrase, template, compose prompt d4, rejects, desc_check turn
  fields.
- `src/rover_vlm/eval.py`: turn follow, by_turn, steering per slice.
- `src/rover_vlm/steering.py`: turn cuts, text-differs rule, slice.
- `scripts/prepare_habitat_instr.py`: `events --labelset-dir`, `d2-seed`, `finalize-descriptions --round d2`,
  `steer` / `steer-select` / `finalize-steer` changes.
- `slurm/label_descriptions.sbatch`: `ROUND` (file tag, default d1); with `ROUND=d2` the gate reads
  `events_eval_changed.jsonl`; steering stages take `STEER_DIR`.
- `scripts/audit_instruction_labels.py`: `--mode turns` (G1: turns, rescued bent routes, spur drops);
  `--mode d1` reads the labelset next to the events file (G3); stratified steer audit.
- `scripts/visualize_steering.py`, README (index-only commits; the user's uncommitted README row is
  never staged).
- Dated pointer lines at the end of `2026-09-30-route-descriptions-design.md` and
  `2026-10-07-steering-eval-design.md`.

## Plan refinements (2026-10-13, from prototyping the plan's synthetic scenes)

- **A pass of the turn's anchor after the turn point is dropped.** When a single-object pass of the turn's anchor
  object comes at or after the turn point, the pass event is left out. Prototype: without this, a visible
  post round which the route turned read "turn left round the post, then go right of the post". A pass
  *before* the turn stays ("go right of the box, then turn left behind it").
- **Turn check threshold.** A turn counts as followed when the prediction ends at least
  `min(TURN_FOLLOW_M, 0.5 × gt_lateral_m)` to the stated side of the pre-turn line. `gt_lateral_m`, the GT
  end's own offset, is stored in the turn event and in `desc_check`. A GT route that ends shortly after its
  turn still follows its own text, so `gt_follow_rate` stays the ceiling.
- Plan: `docs/superpowers/plans/2026-10-13-turn-events-d2.md`.

## Out of scope / known issues

- Fixing spurs in the generator (spaceitup-habitat); cutting spurs out of routes (measured, rejected);
  new Habitat rendering.
- Multiple turns or jogs (detector B; jogs are 1.7% of train routes).
- Real-image eval of d2.
- **Known issue, not fixed here:** on ~12% of routes the generator's fpv path differs from our level
  0.8 m camera projection by > 10 px. These are scenes where the floor is not 0.8 m below the camera
  (scene 00831: 50–100 px). This also affects `route_events` outlines and visibility for those frames,
  in d1 as in d2.

## Addendum T1 (2026-10-14): G1 round 1 result, turn point at the first bend, G1 round 2

**G1 r1** (https://claude.ai/artifact/PERNvXynzchRVetGXdYycZ, 83 train routes, seed 0; only 3 rescued bent routes
exist in train): **69 / 83 true (83%) — below 90%**. Visible turns 18/20, hidden turns 22/30, spur drops 28/30,
rescued bent 1/3. Diagnosed route by route (top-down renders):
- **Turn point too late (00641).** A turn made in two bends got its point on the second bend: the final heading is
  109° off the line, the first bend reaches only ~35°, so "half the final angle" fell on bend 2. The user puts it at
  bend 1.
- **"Behind" far from the turn is not the cause.** The user marked OK 12 of the 16 hidden turns whose object is
  ≥ 3 m before the turn point (train median 3.2 m). Only 00552 is noted: name the post the route passes, not the
  barrel by the camera.
- **Stop errors from d1's unchanged stop logic** (00005 ×2, 00330): the stop object is near on the map, but its
  outline or the hidden end of the route makes it look far in the image. Keeping the stop outline near the stop
  was prototyped and rejected: it does not change 00330, swaps 00005 for another doubtful stop, and changes 3.5%
  of stops without bringing outlines closer.
- Single cases: a missed first turn of a jog (00770; detector A names only the final turn), an ambiguous route
  that bends both ways (00469), a route through the mapped sofa (00518), 3 "can't tell" on odd renders, 2 spur
  drops the user would keep (not separable from the 28 agreed drops by leg gap).

**Decision (user, 2026-10-14):** fix the turn point, then a turn-focused G1 r2.

**Rule change.** `detect_turn`'s turn point is the **first bend that takes the route clearly onto the turn side**:
the first point, after the route last headed along or away from that side of the start→goal line, at least
`TURN_MIN_DEG / 2` (22.5°) onto it. A route that never heads along or away from the turn side falls back to the
r1 rule (last crossing of half the final angle). Start / end arcs are computed from the new point as before, so a
two-bend turn starts at bend 1 and ends after bend 2. Prototype: on 1,000 random train routes 12 of 88 turns move
≥ 0.5 m, 10 change their anchor or relation; 00641 moves from 6.6 m to 5.1 m.

**G1 r2.** A fresh sample (seed 1) of the re-run events: 25 visible + 35 hidden turns, every rescued bent route,
20 spur drops. A new verdict "Wrong stop" separates d1-era stop errors from "Other step wrong". **Gate: ok /
(n − wrong-stop) ≥ 90%**; "can't tell" counts as not true (also reported without it).

**G1 r2 result** (https://claude.ai/artifact/Tgds6HXXbcsuaUszc8dFds): **74 / 80 = 92.5% — PASSED** (raw 89%). Spur
drops 20/20, visible turns 22/25, hidden turns 31/35, rescued bent 1/3. Open notes, not blocking: 2 "round / behind
object is empty space" (ghost outlines), 2 "the pillar is missing as an object" (00469, 00518), 00106 between vs
through, 3 d1-era stops.

## Addendum T2 (2026-10-14): endings — "proceed behind X" for routes that end out of sight

**Why.** Before G2 the user found the stop clause too vague. A CPU probe (scratchpad; page
https://claude.ai/artifact/2TggHhKtBsP8VTpvcuuBPw; 21,096 described d2 train routes, a stratified sample of 1,988
for visibility, d1's entry-form eval for model behaviour) showed why:
- 45% of routes say "stop well past X", 28% "a little past X", 17% "just after it", 9% "in front of X". A "past"
  anchor is the obstacle where the route **starts** (median 1.4 m in front of the camera, 53% within 1.5 m), while
  "well past" routes end 6.8 m out.
- **78% of route ends are hidden** in the image (dataset flag; 88% of "well past", 48% even of "in front of X").
  For 72% of hidden "well past" ends the thing hiding the end is X itself.
- A hidden end is not where the route disappears: median 0.21 uv away for "well past", almost all of it sideways;
  only 5% within the eval's 0.03 stop tolerance. No phrase a person can judge from the image pins it.
- Better anchors near the end are rare where needed: for "well past" another nameable object lies within 1 m of the
  end on 4% of routes, within 2 m on 16% (94% of those walls).
- "In front of X" with a hidden end: the end lies outside X's horizontal image extent 79–93% of the time, behind a
  different, nearer object. This is G1's "near on the map, far in the image" (00005, 00330, 00389, 00475).
- Turn routes: 73% say "well past" an object passed before the turn (median 7.4 m); they end 2.2 m after the turn
  (p10 1.3, p90 4.0 m).
- d1 hits the stop on 20–26% of records in every stop class, precise ones included, and its endpoint loses to the
  constant goal (0.062 vs 0.054). The miss is sideways (|du| 0.056 vs |dv| 0.016) and 73% end deeper than the GT.
  About half the endpoint error is the prediction ending off the GT path, so a stop phrase can fix only part of it.

A second probe on the 1,471 hidden-end routes of the sample measured the replacement: 90% of routes go out of sight
once (visible blips < 0.3 m merged), the hidden tail is a median 2.1 m; the blob hiding the route just past the
disappearance is found on the map 97% of the time and nameable 95% (**92.5% usable**); it is an object of the last
event 86%, the blob hiding the end 88%; 7% of routes have an obstacle event after the disappearance; for hidden
"behind" turns the turn anchor is that blob 82% of the time.

**Decisions (user, 2026-10-14).**
1. A route whose end is hidden is described up to where it goes out of sight and ends with "…, then proceed behind
   X" (X = the object it disappears behind). **The answer keeps the full GT route**, hidden waypoints (v = 0)
   included; only the text changes.
2. A turn made out of sight is the ending: "…, then turn right behind the table." No stop clause follows.
3. Visible ends keep today's stop rules, except that a would-be "well past X" re-anchors to a nameable object near
   the end when one exists.
4. New texts come from **rewriting the ending of the reused d1 text** (approach B); Cosmos composes only what a
   rewrite cannot serve. Rewrites use four verbs: proceed / continue / carry on / go on behind.
5. Steering, gates and the run book follow (below). The spec stays in round d2: no d2 label exists yet, the data dir
   (v4), label files (`*_d2`) and adapter path are unchanged.

### T2.1 Geometry — the ending (`route_events.py`)

- New input `hidden_from_arc_m`: the arclength where the route's **final hidden stretch** starts, or `None` when the
  end is visible. Visible blips shorter than `HIDDEN_BLIP_M = 0.3` m inside hidden stretches count as hidden (they
  are flicker along an occluder's edge). The caller computes it with the flags its own answer uses:
  - candidates (`describe_route`): `fpv_paths.json` run flags; the fpv points are the route resampled every
    0.05 m (checked on 6 routes: L / (N − 1) = 0.049–0.055), counted back from the route's end;
  - steering cuts: `first_blocker` per densified point, as `steering.answer_path` flags them.
- The ending, first rule that fits:
  1. **turn ending**: the end is hidden and the route's turn event (if any) has `arc_m ≥ hidden_from_arc_m`. The
     turn row is unchanged (relation behind / round / none as today); the stop row is
     `{"type": "turn", "relation": null, "object": null}`.
  2. **behind**: the end is hidden. X = the nameable blob hiding the route `BEHIND_READ_M = (0.15, 0.3, 0.5, 0.8)` m
     past the disappearance (first that hits; `_first_hit`, outline by `_anchor_outline` as for turn anchors), else
     the nameable blob hiding the end. Stop row `{"type": "behind", "relation": "behind", "object": X}`. No
     nameable X → the route is dropped, reason **`no_behind_anchor`** (expected ~6% of routes).
  3. **visible end**: `stop_phrase` as today, except that a stop that would be "well past X" first tries the near
     rule with radius `REANCHOR_M = 1.5` m instead of `STOP_NEAR_M = 1.0` m ("in front of / just past Y", a
     wall-like Y only "in front of"); if nothing fits it stays "well past X".
- Behind and turn endings store `disappear_arc_m` in the stop row. Obstacle events with `arc_m >
  hidden_from_arc_m` are dropped (the text describes what is visible); a turn before the disappearance stays an
  event and the ending follows it.
- `EVENTS_SCHEMA = "d2-endings"`; `REASONS` gains `no_behind_anchor`. Expected on train: ~16.4k behind / turn
  endings, ~4.7k visible stops (a few hundred re-anchored), ~1.2k new drops.

### T2.2 Text (`descriptions.py`)

- **Template**: behind → "…, then proceed behind it." when X is the last object of the last event in the body, else
  "…, then proceed behind the X."; with no events "Go straight ahead, then proceed behind the X." A turn ending is
  the body alone ("…, then turn right behind the table."). Visible stops: `stop_text` unchanged.
- **Rewrite** (CPU, in `d2-seed`) of a reused d1 text whose events are unchanged and whose stop became behind, when
  X has a d1 name: cut the text's final stop clause (its last "stop" to the end, plus the connector before it:
  ", then" / ", and then" / ", and" / " and" / ". Then" / ". ") keeping the text's own end punctuation ("?" in question
  styles), and append ", then <verb> behind it / the <name of X>". The verb is chosen by a stable hash of the route
  id from {proceed, continue, carry on, go on}. The result must pass `description_rejects` with the new stop;
  otherwise (or when no stop clause is found) the route is composed. Rows are tagged `"rewritten_from": "d1"`.
- **Buckets** (`d2-seed`): *reuse* = events and stop identical to d1 (as today); *rewrite* = above; *compose* =
  turn routes, re-anchored visible stops, routes whose events changed, X without a d1 name, failed rewrites.
  Estimate: train reuse ~4.4k, rewrite ~12–13k, compose ~4–5k (~5–6 h Cosmos); eval ~10% of that. `seed_meta.json`
  gains `rewritten`, `rewrite_failed` (by reason); `events_{split}_changed.jsonl` holds the compose bucket only.
- **Compose prompt d5**: the stop line becomes an ending line — "end with: then proceed behind <X>" / "end with the
  turn" / today's stop line — and for behind and turn endings "do not say where it stops".
- **Rejects**: a behind ending needs "behind" and X's name (or "it" when X is an object of the last step, as the
  turn pronoun rule) in the last clause, and no `\bstop` anywhere (`ending:stop_word`); a turn ending keeps the turn
  rejects and also forbids `\bstop`; amount words only for visible distance stops (unchanged). Finalize re-checks
  every text, reused and rewritten included.

### T2.3 Eval (`eval.py`)

- `desc_check` gains `ending`: `{"type": "stop"}`, `{"type": "behind", "outline": <X's image polygon>}` or
  `{"type": "turn"}`.
- `description_follow`: **stop** unchanged (last waypoint within `STOP_TOL_UV` of the GT end). **behind**: the
  prediction ends with v = 0 and the first waypoint of its final v = 0 run lies inside X's outline dilated by
  `STOP_TOL_UV` ("it went out of sight behind X"; where the hidden tail ends is not judged). **turn**: no ending
  check beyond the turn event's own. `follows` = every event and the ending.
- Metrics: `stop_rate` over visible-stop records only; new `behind_rate`, `ending_rate`, and `by_ending` (stop /
  behind / turn: n, follow rate, GT ceiling, endpoint error median, constant-goal median of the group). Endpoint
  metrics are unchanged (the GT is the full route).
- **Ceiling check before any GPU job**: on the v4 eval split the GT paths pass their own behind check at
  `gt_follow_rate ≥ 0.95`. If the 10-waypoint resampling puts the GT's first hidden waypoint too far past X's edge,
  the tested point changes (e.g. the midpoint of the last visible and first hidden waypoints), not the tolerance.
- **Ceiling result (2026-10-14, after the events rerun).** First hidden waypoint: 0.898 eval / 0.885 train (FAIL); the
  GT mostly goes out of sight 0.03–0.05 uv past the outline's edge (outlines are tighter than the silhouette), a few
  routes dip behind a nearer object first. Midpoint: the same. **Decision (user): the tested point is any waypoint of
  the final hidden run** (`BEHIND_POINT = "any_hidden"`, tolerance unchanged at 0.03): 0.955 eval / 0.968 train, PASS.

### T2.4 Steering (`steering.py`)

- Every cut gets its own `hidden_from_arc_m` from its answer's `first_blocker` flags and is described with T2.1.
  `stop_key` becomes the ending key: ("stop", relation, amount, object) / ("behind", X) / ("turn",).
- Visible stop A vs behind B: valid and exclusive (A ends in view, so fails B's behind check; B ends ≥ `MIN_SEP_UV`
  away, so fails A's stop). Two cuts behind the same X: same text → `same_text`.
- **Hidden-turn pairs disappear**: A_turn just before a hidden turn reads "proceed behind X", which B_turn also
  follows, so the pair is not exclusive. The turn slice keeps visible-turn pairs only. Gate 0 (≥ 50 turn pairs) was
  already expected to fail (~52 viewpoints) and stays the user's decision. Not done: giving "behind" a "no turn"
  condition (the text does not say "straight").
- The rebuild runs after training, as before; its CPU stage prints pairs per slice next to the previous numbers.
- **Turn set (user, 2026-10-14, built before the full labelling job).** A CPU dry run on the d1 eval frames gave 473 stop /
  21 turn viewpoints; naming every object would give 579 / 24 (1,135 pairs were lost to `unnamed_object`, mostly
  wall-like "behind" anchors that `steer-objects` never names). Two changes: (1) `turn_cuts` ends A at the earlier of
  the turn start and the route's disappearance (minus 0.3 m), so a hidden turn's A stops in view and the pair is
  exclusive again — prototype 39 turn viewpoints (586 stop) with every object named; (2) `frame_pairs(missing=)` collects
  the object rows that left pairs unnamed and `steer` merges them into `steer/objects_eval.jsonl` for `steer_names`, then
  `steer` runs again. **Gate 0 turn threshold lowered to >= 35 (user)**; the stop slice keeps >= 150. Before finalize the
  stages read the d1 eval frames (`--frames data/prepared_habitat_instr_v3/eval_desc_entry.json`).

### T2.5 Gates and run order

| # | Step | Where | Gate |
|---|---|---|---|
| 1 | back up v4 events to `data/prepared_habitat_instr_v4/events_d2turns_backup/`; `events` into v4 | CPU | counts by ending and reason |
| 2 | ending ceilings on eval GT (stop / behind / turn) | CPU | behind `gt_follow_rate ≥ 0.95` |
| 3 | `d2-seed --force` | CPU | reuse / rewrite / compose counts, Cosmos hours |
| 4 | **G1e ending audit** (`audit_instruction_labels.py --mode endings`, seed 0): 30 behind endings shown with the actual rewritten sentence, 15 hidden-turn endings, 15 visible stops (10 re-anchored + 5 unchanged), 10 `no_behind_anchor` drops; FPV with route (hidden part dashed) and X outlined. Marks: true / wrong object / should stop in view / wording broken / can't tell | human | **≥ 90% true**, "can't tell" counts as not true |
| 5 | G2 label gate: `ROUND=d2 DATA=data/prepared_habitat_instr_v4 NAMER=mix sbatch slurm/label_descriptions.sbatch` (compose bucket of eval) | GPU, ask first | accept ≥ 0.90, template ≤ 0.10 |
| 6 | G3 audit on the gate output | human | ≥ 90% |
| 7 | full labels (`STAGE=full`; ~5–6 h, possibly two resumed jobs) → `finalize-descriptions --round d2` → train d2 (~6 h) → eval d1 + d2 on v4 with path videos → steering rebuild (Gate 0) → report | GPU, ask before each | as in the d2 spec |

**Success criteria added by T2** (the d2 spec's still hold): G1e ≥ 90% true; behind GT ceiling ≥ 0.95; on v4 eval
(entry form) d2's `ending_rate` and `behind_rate` beat d1's (d1 never saw "proceed behind": its rate is the
text-blind floor), with the 95% CI of the difference excluding 0; per-ending endpoint error reported against the
constant goal of that group.

**Files touched (T2)**: `route_events.py` (ending, `hidden_from_arc_m`, re-anchor, schema),
`descriptions.py` (template endings, rewrite, rejects, seed buckets, prompt d5, `desc_check` ending), `eval.py`
(behind check, `by_ending`), `steering.py` (per-cut hidden flags, ending key), `scripts/prepare_habitat_instr.py`
(`events` reads fpv flags, endings ceiling stage, `d2-seed` rewrite bucket), `scripts/audit_instruction_labels.py`
(`--mode endings`), README run book (index-only commits).

**Testing** (test-first, CPU, synthetic grids as in the d2 plan): fpv index → arclength; blip merging and the final
disappearance of a route that reappears; behind X from the disappearance, fallback to the end's occluder,
`no_behind_anchor`; hidden turn after the disappearance → turn ending; obstacle event after the disappearance
dropped; visible "well past" re-anchored at 1.2 m, wall-like only "in front of"; template endings (it / the X / no
events); rewrite on each connector form, a question-style text, a text with no stop clause, a rewrite failing
rejects; rejects `ending:stop_word` and the pronoun rule; seed buckets; behind check (inside / outside the outline /
ends visible), turn ending with no stop component, `by_ending`; steering ending key, visible-vs-behind exclusive,
hidden-turn pair not exclusive.

**Risks.** The behind check leans on the model's v flags (d1 `path_visibility_acc` 0.87). ~7% of hidden-end routes
lose an out-of-sight event from their text. Rewritten endings use four verbs in one frame (G1e and G3 read them).
On ~12% of routes the fpv path and our 0.8 m projection differ by > 10 px (known issue above): X's outline and the
GT's first hidden waypoint can disagree there, which step 2 measures. Hidden-turn steering pairs are gone.

Plan: `docs/superpowers/plans/2026-10-14-d2-endings.md`.

**G1e result (2026-10-14, https://claude.ai/artifact/N6sEFJWCV6QTyQcs298AgM, 70 train routes, seed 0): 56/70 = 80% raw —
accepted by the user.** Behind endings 29/30 (the miss: an object name, not the geometry); hidden-turn endings 12/15;
re-anchored stops 8/10; unchanged visible stops 4/5 (1 "can't tell"); `no_behind_anchor` drops 3/10 right (4 "should not be
dropped": a sofa / corridor door / chair + sofa / fireplace does hide the end; 3 "should stop in view"). Population-weighted
over described routes ~92% true. Decision (user): keep the drops (171 train routes, 0.8%; no wrong text enters training)
and go to G2. Open notes, outside T2: steps missing from the event selection (00647 x2 black sofa, 00611 second pillar,
00551 a final "through the door"), a ghost "empty space" outline (00792), the drop rule's misses.

**G2 (job 93455): PASS** — accept 0.957, template 0.063 (300 of the 327 eval routes to compose; 9 name clashes, 4 unnamed).
**G3 label audit (https://claude.ai/artifact/1Z3ZCbkSxF3uKkKjGVS3Uw, 100 eval routes: 40 rewritten d1 texts, 57 Cosmos, 3
template): 85/100 correct — accepted by the user.** Rewritten 36/40, Cosmos 46/57, template 3/3; endings ~97% right (1 wrong
ending in scene 00831, 2 unsure). The 12 wrong names are the namer's (d1's own label audit was 84/100): 7 are the behind
object, mostly a wall named as furniture — 68% of behind objects are long blobs and the wall naming prompt offers "a long
piece of furniture such as a sofa or a counter"; only 27% of them get a wall-type head noun. Open: a wall-first naming pass.
