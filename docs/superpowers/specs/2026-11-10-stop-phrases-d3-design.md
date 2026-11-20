# Stop phrases that say where the rover ends up, and the d3 dataset — design

Branch `feat/stop-phrases` (from `main` 87e814e, which holds d2). Builds on
`2026-09-30-route-descriptions-design.md` (events, stop rules, Addendum A),
`2026-10-13-turn-events-d2-design.md` (+ Addendum T2 endings) and `2026-10-15-object-naming-design.md`
(+ Addendum C, the C1 names). Status: design approved in conversation 2026-11-10; plan not written yet.

## Why

The d2 eval (report https://claude.ai/artifact/T3SdLHN3zsz9HbDcXqqZ3R, `outputs/eval_desc_d2/`) fixed the
"behind" endings (94.3% vs d1 77.5%, ground truth 95.5%) but left one thing broken: routes that end **in plain
sight with a stop phrase**. They are 25% of the eval routes (532 of 2,145) and both models meet the stop only
about one time in five (d2 22.2%, d1 20.9%). It is also why steering pairs that differ only in their stop reach
just 8.9% (turn pairs 36.1%).

Today's visible stops come from three rules in `route_events.stop_phrase` (eval / train counts, v4 events):

| type | text | eval | train |
|---|---|---|---|
| `distance` | "stop a little past / well past / well before X" (coarse amounts: just < 1.5 m, a little 1.5–3 m, well > 3 m) | 270 | 2,356 |
| `after_last` | "stop just after it" (≤ 1.5 m of route after the last event) | 138 | 1,222 |
| `near` | "stop in front of / just past X" (a blob within 1.0 m) | 132 | 1,137 |

Distance words name lengths a single frame cannot show, so the model cannot learn where "well past" ends.

## Decisions (user, 2026-11-10)

1. **No distance words.** A visible stop names where the rover ends up:
   - an object close to the end: "stop **in front of** the sofa", "stop **next to** the table", "stop **behind** the armchair";
   - nothing close: a **place**: "in the middle of the room", "at the end of the hallway", and similar.
2. **Geometry chooses the place, from a short fixed list**; Cosmos only words the sentence, and the label
   checks reject a text whose place word does not match. (Free description by Cosmos from the image was
   offered and declined: nothing could check it.)
3. **Only visible stops change.** Hidden "behind" endings, turn endings, route events and object names stay
   as in d2.
4. The pipeline stays the d2 one: the same gates (human audit pages with marks, label gate, label audit),
   relabel only what changed, train **d3** with d2's recipe, compare against d2 on the same eval routes, rebuild
   the steering pairs with the new stops, path videos and a report.
5. Every GPU job is asked for first (thor, max 3 jobs).

## Success criteria

- **Main:** the stop is met on the new visible-stop eval routes well above d2's rate on the *same* routes
  (d2 is re-evaluated on the d3 eval texts; 22% on the old texts is the reference).
- **Steering:** stop-pair success (entry form, text only) above d2's on the rebuilt steering set (8.9% on the old set).
- **No regression** (same eval routes): hidden "behind" ending ≥ 93% (d2 94.3%), turn ending not lower than d2
  beyond its interval (73.7%, n = 76), overall follow rate not below d2's.
- **Labels:** G1 (new stops on frames) ≥ 90% true, every stop type reported on its own; G3 (texts) ≥ 90%.
- Report the stop rate **per stop type** (each relation, each place) next to a fixed-guess baseline, so a type
  that does not work is visible rather than averaged away.

## Measured before design (2026-11-10, scratchpad probe over v4 events + objects)

Distance from the goal to the nearest segmented blob of the frame, visible-stop routes:

| within | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|
| any blob (train 4,715) | 1% | 35% | 46% | 64% |
| `object` blob only | 1% | 22% | 27% | 41% |
| eval (540), any blob | 1% | 31% | 45% | 65% |

Nothing is within 0.5 m because the planner keeps the rover's 0.5 m radius clear. HM3D semantic labels (with
numbered regions, never room types) exist for only 104 of 529 train scenes and 20 of 66 eval scenes, so room
information must come from geometry. Every frame already loads its occupancy grid (`load_frame` →
`Frame.grid`, `state.npz`: FREE / OCC cells, `grid.mpp` metres per cell).

## The new stop (`src/rover_vlm/route_events.py`)

A new stop function replaces rules 1–4 for routes whose end is visible (`ending` is not behind / turn). First
rule that fits:

**1. Object relation** — the nearest visible, nameable **object** blob (not wall-like: extent < `WALL_EXTENT_M`)
whose visible cells come within `CLOSE_M` = 1.5 m of the goal (≈ 1 m of gap given the 0.5 m radius). The
relation is read from where the object sits relative to the goal:
- **in front of** — the object is ahead of the goal along the final heading (as today's `_near_stop`:
  along > |lateral|);
- **behind** — seen from the camera, the object stands between the camera and the goal (near the camera→goal
  ray and nearer the camera than the goal) while the goal itself stays visible. This is the visible twin of
  the T2 hidden "behind" ending and uses the camera, not the route heading;
- **next to** — the object is beside the goal: |lateral| ≥ along, and the goal lies within the object's
  extent along the final heading (beside it, not diagonally past a corner).

  The relations are tried in the order behind, in front of, next to; the first that fits wins, and an object
  that fits none (diagonal to the goal) is skipped for the next-nearest one. An object that hides the goal
  itself is not this rule's case: that route already has a hidden T2 "behind" ending.

  "Next to" was removed on 2026-10-01 (Addendum A of the 2026-09-30 spec: wrong in 12 of 29 audited routes,
  mostly through imprecise outlines). It comes back with the stricter "within the object's extent" rule and is
  scored on its own line at G1; if it still fails there, it is dropped again (user decides at that gate) and
  those routes fall through to the place rule.

**2. Place** — when no object qualifies, the place at the goal, read from the occupancy grid around it. The
fixed list, tried in this order, first fit wins (initial definitions; thresholds calibrated in the CPU dry run
and at G1):
- **in a doorway** — the goal within ~0.75 m of an opening between two wall-like blobs (reuse the opening /
  `through` geometry);
- **in a corner** — wall-like occupied cells within ~1.5 m of the goal in two roughly perpendicular directions
  (60–120° apart);
- **at the end of the hallway** — the free width across the final heading stays ≤ ~2.2 m along ≥ ~2 m of route
  before the goal, and the corridor ends (blocked ahead, or widening into a room) within ~1.5 m past the goal;
- **against a wall** — one wall-like stretch within ~1.0 m of the goal, none of the above;
- **in the middle of the room** — clearance ≥ ~1.5 m in every direction around the goal.

  A goal no rule fits is **dropped** (counted by reason in `events_meta.json`, like `no_stop_anchor`); no
  distance fallback. The dry run reports the drop share before G1; if it is large the thresholds are revisited
  before the audit.

Walls are deliberately not "objects" for rule 1: "in front of the long white wall" does not say where along it;
the place types carry wall information instead (my call, see the list at the end).

The stop record keeps the d2 shape so downstream code needs little change: `{"type": "object", "relation":
"in front of" | "behind" | "next to", "anchor": {...blob, outline, kind}}` or `{"type": "place", "place":
"doorway" | "corner" | "hallway_end" | "wall" | "room_middle"}`, plus the measured quantities the rule used
(distances, widths, angles) for the audit page. `desc_check` carries the stop type for the per-type eval split.

## Text (`src/rover_vlm/descriptions.py`)

- `stop_text` renders the new types: "stop in front of the {name}", "stop next to the {name}", "stop behind the
  {name}", and for places "stop in the middle of the room" / "stop at the end of the hallway" / "stop in the
  corner" / "stop against the wall" / "stop in the doorway". Small wording variety is allowed (as
  `STOP_RELATION_WORDS` does today), each variant listed.
- The compose prompt gets the new stop clause; the rejects check that the text uses the stop's relation or place
  words, names the anchor for an object stop, and contains **no distance amount words** ("a little", "well",
  "just past", "short of", metres). The d2 checks for events, sides, behind and turn endings are unchanged.
- **Seeding (`seed_d3_labels`, like `seed_d2_labels`):** routes whose stop did not change keep their d2 text
  (tagged `reused_from: d2`). For a changed visible stop, the d2 text's stop clause is replaced in place when it
  can be located unambiguously (tagged `rewritten_from: d2`); otherwise the route is composed by Cosmos. The
  d2-seed rewrite path for T2 is the model to follow. Names are the C1 names already on every blob
  (`names_*_C1_n2` cover all 29,556 train objects); an unnamed anchor goes through the existing name stage.

## Data flow and run order (every sbatch: ask first)

New outputs in `data/prepared_habitat_instr_v5` (v4 stays as it is for the d2 comparison); label files
`outputs/instr_labels_v3/*_d3.jsonl`; `ROUND=d3` in `slurm/label_descriptions.sbatch`.

1. `events --round d3` (CPU) → `$V5/events_*.jsonl`, `objects_*.jsonl`, `events_meta.json` (stop types, drops).
2. CPU dry run summary: stop-type shares, drop share, examples per type.
3. **G1 (human):** audit page of new stops on frames (`audit_instruction_labels.py --mode stops`): ~100 train
   routes, stratified so every relation and place type has ≥ 10; outline / place drawn; marks in the page db;
   gate ≥ 90% true overall, each type reported.
4. `d3-seed` (CPU) → reused / rewritten / to-compose counts (≈ 5,300 changed routes at most; compose ≈ 13/min).
5. **G2:** `ROUND=d3 ... sbatch slurm/label_descriptions.sbatch` (gate job): accept ≥ 0.90, template ≤ 0.10.
6. **G3 (human):** label audit page on gate + seeded eval texts, ≥ 90% correct (publish and wait before step 7).
7. `STAGE=full` labelling job → `finalize-descriptions --round d3` → `$V5/train_desc.json`, `eval_desc_*.json`.
8. Train d3: `sbatch slurm/train_habitat.sbatch $V5/train_desc.json outputs/runs_desc/d3 --epochs 2 --save-steps 500 --resume`.
9. Steering rebuild on v5 (below), its gates 0–2.
10. Eval d3 **and d2** on the v5 description sets and steering sets → `outputs/eval_desc_d3/`; path videos;
    report (same shape as the d2 report, plus the per-stop-type section).

## Eval (`src/rover_vlm/eval.py`)

The stop check itself does not change: the predicted end within `STOP_TOL_UV` (0.03) of the GT end, so place
stops need no new scoring. Add a **`by_stop`** breakdown to the description metrics (each relation and each place:
n, stop rate, ending rate, endpoint error median, fixed-guess error median), next to the existing `by_ending`.

## Steering (`src/rover_vlm/steering.py`)

Stop pairs cut a route at two points whose stop phrases differ; the cuts now use the new stop function, so a
pair can read "stop in front of the sofa" vs "stop in the middle of the room". Gate 0 (≥ 150 pairs; the turn
slice keeps its ≥ 35), Gate 1 audit (≥ 90%), Gate 2 (≥ 85% composed) as for d2. The turn slice is unchanged.

## Testing

Test-first on CPU with synthetic `state.npz` grids, as for d2: one scene per relation and per place type
(including near-misses: an object diagonal to the goal is not "next to"; a wide corridor is not a hallway;
an object behind the goal along the heading but not on the camera ray is not "behind"); text rendering and
rejects (distance words rejected, place words required); seeding (reuse / rewrite / compose); `by_stop` metrics;
steering cuts with new stops. The full suite (1,249 at the start) stays green.

## Files touched (expected)

`src/rover_vlm/route_events.py`, `src/rover_vlm/descriptions.py`, `src/rover_vlm/eval.py`,
`src/rover_vlm/steering.py`, `scripts/prepare_habitat_instr.py` (`events --round d3`, `d3-seed`,
`finalize-descriptions --round d3`, steering defaults for v5), `scripts/audit_instruction_labels.py`
(`--mode stops`), `slurm/label_descriptions.sbatch` (`ROUND=d3`), README run book, tests.

## Out of scope / anti-goals

- Hidden "behind" endings, turn endings, route events and object naming stay as in d2. A change that makes
  them worse is a failure even if stops improve.
- The comparison with d2 must stay like for like: same eval routes, d2 re-run on the d3 texts.
- No distance words in any visible stop, and no free-form place descriptions without a geometric check.
- No new naming round; no new model recipe.

## Calls made in writing this (not discussed; check them)

1. `CLOSE_M` = 1.5 m (≈ 1 m of gap); the user's "close" was not given a number.
2. Walls are not anchors for the object relations; wall information goes into the place types.
3. "Behind" for a visible end is judged from the camera (object between camera and goal), not from the route.
4. "Next to" returns with a stricter rule and its own G1 line; it is dropped again if it fails there.
5. Goals that fit no rule are dropped rather than given a fallback phrase.
6. Changed stops are rewritten in the d2 text where possible, composed otherwise.
7. The place thresholds above are starting values, calibrated in the dry run and at G1.
8. Relation order behind → in front of → next to, and place order doorway → corner → hallway end → wall →
   middle of the room; the first fit wins.

## Addendum A1 (2026-11-10): the dry-run gate, more places, the wall as a backup

**Finding.** After the final review's fixes, a probe of the d3 code on 800 v4 routes that end in view (400 eval +
400 train, seed 0) dropped **45%** as `no_stop_place` (gate: 15%): against a wall 31%, corner 11%, middle of the room
6%, object relations ~6% in all, doorway 1%, end of the hallway 0%. Why: the explored grid is mostly unknown around the
end (a median 13 of 16 rays stopped on an unknown cell before reaching a wall that was there); a third of the drops had
their nearest wall 1.0–1.5 m away (too far for "against the wall", too close for the middle of the room); the rest were
open space less than 60% seen. Skipping the goal's own cell (45%) or letting rays cross ≤ 2 unknown cells (43.6%) barely
helped. Under the best tuning (below, before the new types) 33.5% were still dropped: corridors without their end
11.9%, 2–4 walls at similar distances 11.9%, one clearly nearest wall 4.6%, open but < 45% seen 3.6%.

**Decision (user, 2026-11-10): option 3 (more place types) with option 2 (the wall as an anchor) as a backup.**

1. **Rays see through unknown cells** to find walls (an OCC cell beyond unknown ones is still a wall), but a ray that
   crossed unknown never reports "free"; unknown cells still never count as free space.
2. **"in the hallway"** (new place): two facing walls (directions ≥ 150° apart) at most `HALL_MAX_W` (2.2 m) apart,
   and the corridor as narrow 1 m along its own axis on at least one side, not at its end (the end of the hallway is
   tried first). The axis need not follow the rover's heading: requiring the walls at the rover's left and right fired
   for 0.5% of the probe, because most corridors ran across or diagonal to the final heading (ruling while building). Place order: doorway → corner → end of the hallway → in the hallway → by the wall → middle of the room.
3. **"by the wall"** replaces "against the wall": the nearest wall within `WALL_M` = 1.5 m (was 1.0), when it is
   alone or at least 0.5 m nearer than any other wall within that reach.
4. **The middle of the room** needs ≥ 45% of its 1.5 m disc seen free (was 60%).
5. **Backup (option 2):** when no place fits, a visible wall-like blob within `CLOSE_M` ahead of the goal (along >
   |lateral|, as `_near_stop`) gives "stop in front of the <wall name>" (stop type `wall`, kind "wall:in front of").
   Only then; otherwise the route is dropped. Open space less than 45% seen stays dropped (calling it open would be a
   guess).

Expected on the probe sample: ~17% dropped without the backup, ~13% with it. The calibration gate of the plan (Task 9)
still applies to the full events run.

## Addendum A2 (2026-11-16): furniture is not "the wall"

**Finding.** G3 (84 eval items) scored 63%; 7 of its 9 wrong-stop routes have one cause: `_wall_like` is size only
(blob extent ≥ 1.5 m) on a 2D map, and `load_frame` merges blobs one empty cell apart, so a sofa set, a dining set or
a kitchen island with stools is a "wall". Two effects: "by the wall" fires next to furniture in open floor, and
`object_relation` skips the big sofa and names a farther object. A per-cell height test on the depth image tells them
apart (below). Applied everywhere it lost a blind A/B judged by the user (67 changed items: old right 33, new right
23): corridors formed by furniture and a wall are real hallways to the user (old 14, new 5). Only where the old stop
was "by the wall" did it win: new 10, old 8, and 6 of those 8 were routes the new rule dropped. With "never drop":
new 10, old 2, neither 2.

**Decision (user, 2026-11-16): the targeted fix (option 1).**

1. **Height class of an occupied cell** (`route_events.cell_heights`), from the depth profile in the cell's own image
   column (relative, so the depth fit's 0.3–0.7 m bias at range does not matter): `d_obj` = the median planar depth
   at 0.1–0.85 m above the floor (every 0.15 m) over the readings within [fwd − 1.0, fwd + 0.5] m of the cell's own
   forward distance (legs seen through and nearer things are left out); at 1.5 / 1.7 / 1.9 m a reading farther than
   `d_obj` + 0.5 m says low, one within ±0.3 m says tall. With ≥ 2 usable high readings: **low** if more than half say
   low, **tall** if more than half say tall, else **unknown**. Unknown also without a depth image, without a fit, or
   with a fit residual above `DEPTH_FIT_MAX_RESID_M`.
2. **Only a "by the wall" stop is looked at again**, with furniture seen as furniture:
   - a wall-sized blob is **low furniture** when ≥ 8 of its visible cells within 2.0 m of the goal have a class and
     ≥ 60% of those are low; it is then an object for `object_relation`, cleared from the place grid, and not a wall
     for `wall_ahead`;
   - occupied cells classed low within 3.5 m of the goal are cleared from the place grid;
   - then `stop_d3`'s order runs again (object relation → place → wall ahead). Its stop replaces "by the wall";
     when it finds none, **"by the wall" stays** (the fix never drops a route).
3. Every other stop is unchanged: hallways, doorways, corners and object relations keep furniture as walls.

**Expected.** About 26–30% of "by the wall" stops change (G3 11 of 43, G1 3 of 10), about 1 stop in 8; the drop share
does not change; stop accuracy up a few points (~81% → mid-to-high 80s). The rule was chosen after seeing the A/B
marks, so fresh G1 and G3 samples (a new seed, not the A/B routes) confirm it. G3 will still miss 90%: name and wording
errors alone cap it at ~82%.

**Run order.** Rebuild the v5 events (CPU), `d3-seed --force`, the G2 gate job (ask first), fresh G1 stop and G3
pages, then the user's decision on STAGE=full.
