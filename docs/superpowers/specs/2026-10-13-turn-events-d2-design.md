# Turn events, spur cut and the d2 dataset — design

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
4. **Cut the spur** (not drop, not a generator fix). The fix is part of d2.
5. **Recompose only changed routes**; unchanged routes keep their d1 text.
6. Steering: **turn cuts + a `turn` slice**; Gate 1 redone, stratified.
7. A **new spec** (this file) rather than a sixth addendum: the change spans the descriptions and the
   steering specs, adds a spur fix and a retrain, and the descriptions spec already has five addenda.
   Both older specs get a dated pointer here.

## Success criteria

- G1 geometry audit (human, 100 routes: 50 turn routes ≈ 20 visible / 30 hidden, 20 rescued bent
  routes, 30 spur-cut opposite routes): **≥ 90% true**.
- G2 label gate (300 changed eval routes): accept ≥ 0.90, template share ≤ 0.10.
- G3 label audit (human, 100 changed routes, half with a turn): **≥ 90% correct**.
- Steering: Gate 0 ≥ 150 pairs and ≥ 50 in the turn slice; Gate 1 (50 pairs, 20 turn + 30 stop)
  ≥ 90% true; Gate 2 compose accept ≥ 0.9.
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

## Spur cut (`route_events.clean_route` + stage `spur-cut`)

**`clean_route(polyline_xz) -> (polyline, cuts)`**, where `cuts` = intervals of the *original* arclength
that were removed. Applied to every route.
- At each vertex reversing ≥ `SPUR_REVERSAL_DEG = 150`, walk outwards in arclength t (0.05 m steps)
  while P(s_k − t) and P(s_k + t) stay within `SPUR_MIRROR_M = 0.3` m.
- Remove [s_k − t*, s_k + t*] and join the two ends at their midpoint.
- Repeat until the chord-based check (≥ 150° within 0.6 m) finds no reversal. If it still finds one,
  the route is dropped (`spur`).
- Tested on the 212 eval spur routes: 209 cleaned (12.1 → 6.9 m median), 3 V-spikes dropped; 0 of 200
  normal routes changed.

**Answer path.** `fpv_paths.json` points are spaced uniformly along the route (~0.05 m; verified on
three routes, including a frame where the floor is not 0.8 m below the camera). The same original-arclength
intervals are cut from the candidate's point list, which keeps the generator's projection and visibility
flags. The result is then clipped and resampled exactly as `build_record` does. A re-projection of the
cleaned floor route is NOT used: on ~12% of routes it is > 10 px off the generator's path.

**Keep rule (opposite routes).** Kept in its pair only when:
- the cleaned route's answer path is still ≥ 0.05 image Fréchet from the GT route's (the d1 pair rule);
  and
- its largest offset from the start→goal line is still on its own side.

Else dropped (`spur_collapsed`), and the GT record stays as a single record (no `pair_id`). From the
exact-retrace measurement: ~80% keep their side, 97% stay ≥ 1 m from the GT route.

**Stage `prepare_habitat_instr.py spur-cut`** (CPU) reads the v2 labelsets and writes
`data/prepared_habitat_instr_v4/labelset_{split}.json`. Records carry `habitat_meta.route_xz` (cleaned)
and `habitat_meta.spur_cuts`, with the rebuilt answer in `conversations`. It prints counts per reason.
`event_row` describes `habitat_meta.route_xz` through `describe_xz` when present.

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
| 1 | `spur-cut` | CPU | v4 `labelset_{split}.json` |
| 2 | `events --out-dir v4` | CPU, 4 workers, ~40 min | `events_{split}.jsonl`, `objects_{split}.jsonl` (schema d2-turns) |
| 3 | `d2-diff` | CPU | `compose_{split}.jsonl` (changed routes); `desc_{split}_reuse_d2.jsonl` (d1 rows of unchanged routes); `objects_to_name_{split}.jsonl` |
| 4 | `STAGE=d2_gate`, then `STAGE=d2 sbatch slurm/label_descriptions.sbatch` | GPU | names (new ids), `desc_{split}_mix_d2.jsonl` |
| 5 | `finalize-descriptions --d2` | CPU | v4 `train_desc.json`, `eval_desc_{entry,goal,pairs,shuffled}.json` |
| 6 | `slurm/train_habitat.sbatch` (d1 settings, 2 epochs) | GPU, ~6 h | `outputs/runs_desc/d2/adapter` |
| 7 | `slurm/eval_descriptions.sbatch` on v4 sets, d1 and d2 | GPU | metrics, report, path videos |

- **Unchanged route** (step 3): same route polyline, same events (type, side, relation, objects) and the
  same stop as its d1 row. Its d1 text is reused. Finalize re-checks every text (reused or new) against
  the d2 rules.
- **Cost.** Cosmos composes ~14 routes/min. ~4–4.5k changed routes (turns ~1.8k, spur-cut opposites
  ~1.9k, rescued bent routes, eval) ≈ 5–6 h, plus a short Gemma naming pass. Counts are printed by
  `d2-diff` before any submit.
- SLURM: at most 3 jobs at once; every GPU submit is asked for first.

## Eval (`src/rover_vlm/eval.py`)

- `desc_check` carries each turn event's `turn_xz`, `pre_heading_xz`, `side`.
- `_event_followed` for a turn: the predicted floor path's end lies ≥ `TURN_FOLLOW_M = 0.5` m to the
  stated side of the GT pre-turn line (through `turn_xz` along `pre_heading_xz`). Stopping short or
  going straight on fails. `gt_follow_rate` gives the ceiling.
- `aggregate_description_metrics` adds `by_turn`: for records with / without a turn event, n, follow rate,
  endpoint and point error medians. `event_rate_by_type` gains "turn".

## Steering (`src/rover_vlm/steering.py`, stages `steer` / `steer-select` / `finalize-steer`)

- Source: the d2 eval GT routes (cleaned). Turns are detected once per route (`detect_turn`).
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
  30 stop, seed 0; the paused 4/50 audit is abandoned. Gate 2: compose with d4.
- Eval: `SETS=steer` for d1 and d2 (+ goal-form control); pair success overall and per slice
  (`audit_instruction_labels.py --mode steer` stratified, `visualize_steering.py` per slice).

## Testing (test-first, CPU, synthetic `state.npz` grids)

- `tests/test_route_events.py`:
  - `detect_turn`: L-route left/right (sign), swerve → None, swing-then-turn, 44° vs 46°.
  - Cut before / after `end_arc_m`.
  - Anchors: behind / round / bare.
  - Turn as first event; top-3 + turn; straight lead-in rescued vs swing kept as `no_events_bent`.
  - `clean_route`: exact retrace, near-mirror, uncleanable V-spike → drop, normal route unchanged.
  - fpv point-list cut on the same intervals.
- `tests/test_descriptions.py`:
  - The three template forms and "go ahead, then".
  - `turn:side:missing/opposite/stray`, missing relation, pronoun anchor accepted.
  - "turn left behind the sofa" gives the sofa no side.
  - A reused d1 text passes d2 rules.
- `tests/test_eval.py`:
  - Turn check: straight fails, turned passes, short fails.
  - `by_turn` split.
- `tests/test_steering.py`: turn cuts, turn-pair preference, texts-differ rule, slice field.
- Stage tests: `spur-cut` (reasons, answer rebuilt), `d2-diff` (reuse vs compose, objects to name).

## Files touched

- `src/rover_vlm/route_events.py`: detect_turn, turn_anchor, turn event, clean_route, describe_xz(turn=),
  schema.
- `src/rover_vlm/descriptions.py`: event_phrase, template, compose prompt d4, rejects, desc_check turn
  fields.
- `src/rover_vlm/eval.py`: turn follow, by_turn, steering per slice.
- `src/rover_vlm/steering.py`: turn cuts, text-differs rule, slice.
- `scripts/prepare_habitat_instr.py`: `spur-cut`, `events` (route_xz), `d2-diff`,
  `finalize-descriptions --d2`, `steer` / `steer-select` changes.
- `scripts/label_descriptions.py`, `slurm/label_descriptions.sbatch`: `STAGE=d2_gate|d2`.
- `scripts/audit_instruction_labels.py`: G1 geometry audit for turns and spurs; G3; stratified steer
  audit.
- `scripts/visualize_steering.py`, README (index-only commits; the user's uncommitted README row is
  never staged).
- Dated pointer lines at the end of `2026-09-30-route-descriptions-design.md` and
  `2026-10-07-steering-eval-design.md`.

## Out of scope / known issues

- Fixing spurs in the generator (spaceitup-habitat); new Habitat rendering.
- Multiple turns or jogs (detector B; jogs are 1.7% of train routes).
- Real-image eval of d2.
- **Known issue, not fixed here:** on ~12% of routes the generator's fpv path differs from our level
  0.8 m camera projection by > 10 px. These are scenes where the floor is not 0.8 m below the camera
  (scene 00831: 50–100 px). This also affects `route_events` outlines and visibility for those frames,
  in d1 as in d2.
