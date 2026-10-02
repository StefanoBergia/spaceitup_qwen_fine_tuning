# Route descriptions (prompt d1) — design

Date: 2026-09-30. Branch: `feat/instruction-paths`. Supersedes the single-landmark target of
`2026-09-28-grounded-landmarks-design.md` (prompt i4, `landmarks.select_target`). The v2
round-4 splits, labelsets and the Cosmos serving/labelling job pattern are unchanged.

## Why

i4 train spot check (100 random labels, 2026-09-30, artifact
https://claude.ai/artifact/Dg8RFzCKUfwKf4bV6399mg, db `marks`): 84 correct, 12 wrong
landmark, 3 can't tell, 1 note-only (00243 was a misclick → correct). Every flagged frame,
rendered with every obstacle the route passes (review page
https://claude.ai/artifact/JP3aNTQtVzxpDtgojWLYa3), falls into:

| group | frames | cause |
|---|---|---|
| A several obstacles, wrong one | 6 | crossing-first rule takes what the straight line grazes (wall, thin post, far fragment) while the route bends around something else |
| B right obstacle, wrong name | 2 | the floor-to-1 m outline also covers what stands behind; Cosmos names that |
| C map artefact at the camera | 3 | `_seen_first` treats occupied cells below the image as occluders; a strip beside the rover hides everything |
| D nothing clear to name | 5 | bend out of view, tiny/thin target, barely bending route |

15/17 targets came from the crossing rule. Offline, scoring the obstacles between the
straight line and the route by √area × route offset fixes 5/6 of group A and keeps 73/83 of
the correct picks.

Beyond fixing these, the end goal is a VLM that **generates a path from a text description**.
One landmark per route under-specifies a path; the goal point in today's prompt pins it
(round 4: 0.011 median error with no text), so the text barely matters.

## Decisions (user, 2026-09-30)

- The model's input is **image + entry + description** (no goal point), trained in a mix
  with **image + entry + goal + description**: 70 / 30 (option D).
- Descriptions list up to **3 events** along the route plus a **stop phrase**.
- Stop phrase: near object → "just after" the last landmark → rough distance from an object
  → drop the route. Rough metric distances in words are accepted.
- Routes described: the round-4 ground-truth route **and** the second feasible route of the
  same frame (Fréchet ≥ 0.05 apart), which restores steering pairs.
- Labelling: **name each object on its own crop, then a text-only call writes the sentence**
  (option 3). Run time does not matter.
- Group D stays (no extra drop rule this round). Fix group B (depth-trimmed outline) and C.
- Principle kept from i4: geometry decides what is true; Cosmos only names and phrases.

## Success criteria

1. Geometry audit (human, 100 routes incl. 30 pair routes): **≥ 90% "true"**.
2. Label gate (300 eval routes): ≥ 90% pass the text rules, ≤ 10% template fallback; human
   audit of 100: **≥ 90% correct** (i4 baseline 84%).
3. The finalized dataset has B-form and goal-form records at 70 / 30, pairs in the same form,
   scene-disjoint train/eval (re-asserted).

## Components and data flow

```
round-4 frames ─► events            (CPU)  prepare_habitat_instr.py events
                  └► events_{train,eval}.jsonl   one row per route
                  └► objects_{train,eval}.jsonl  one row per (frame, object), deduplicated
               ─► name              (GPU)  label_descriptions.py --stage name
                  └► outputs/instr_labels_v3/names_{split}_d1.jsonl
               ─► compose           (GPU)  label_descriptions.py --stage compose
                  └► outputs/instr_labels_v3/desc_{split}_d1.jsonl
               ─► finalize-descriptions (CPU)  prepare_habitat_instr.py finalize-descriptions
                  └► data/prepared_habitat_instr_v3/{train_desc,eval_desc_*}.json
```

New modules (instead of growing `instructions.py`, 832 lines):

- `src/rover_vlm/route_events.py`: pure numpy geometry — candidates, events, stop phrase,
  depth calibration, trimmed outlines, fixed occlusion test. Reuses `landmarks.Grid`,
  `Camera`, `components`, `densify`, `convex_hull`.
- `src/rover_vlm/descriptions.py`: d1 naming and composing prompts, reply parsers, text
  rules, template fallback, style/shape choice (reuses i4's `STYLES`, `_stable_index`).

Changed: `scripts/prepare_habitat_instr.py` (stages `events`, `finalize-descriptions`),
a new `scripts/label_descriptions.py` (`--stage {name,compose}`; reuses `labelling_io`'s
endpoint, resume and output rules — `label_instructions.py`'s main loop is record/pair-shaped
and stays for i1–i4), `scripts/audit_instruction_labels.py` (modes `events` and `d1`),
`src/rover_vlm/habitat_data.py` (`description_entry_prompt`), `src/rover_vlm/eval.py`
(description-followed metric), a new sbatch `slurm/label_descriptions.sbatch` (serve + both
stages in one job, resumable, `STAGE=gate|full`). `landmarks.select_target` and i4 stay
for reproducibility.

## Event geometry (`route_events.py`)

On the frame's own `state.npz` and camera; route resampled every 5 cm; obstacles are
gap-merged occupied blobs (1-cell dilation, as `crossing_target`).

**Occlusion fix (group C).** Occupied cells within `NEAR_CAMERA_M = 0.5` m (the rover radius) of the camera are
neither occluders nor candidates: the rover's start pose is collision-free, and in 00372 such a
strip (0.22–0.45 m ahead, 2 m wide, bare floor in the image) hid the armchair from every ray; 0.35 m was not enough.

**Candidates.** A blob with (a) closest approach ≤ `PASS_M = 0.7` m, ignoring the first
0.3 m of route, (b) ≥ 4 visible cells, (c) depth-trimmed outline ≥ `MIN_AREA = 0.01` of
the image.

**Side.** Per candidate, from the route normal at the closest point. Convention unchanged:
obstacle on the route's right → "left of X". Each event has its own side, so routes bending
both ways are allowed; `route_bends_both_ways` / `side_mismatch` no longer apply.

**Score.** √(outline area) × max(route offset from the entry→goal line at that point, 0.1 m).
Keep the top `MAX_EVENTS = 3` events, then order them by arclength.

**Merging.** Two candidates on opposite sides whose closest points are ≤ 0.5 m apart along
the route become one `between` event (counts as one of the 3). If both are wall-like
(extent > 1.5 m) it is a `through` event; its outline is the gap between them.

**Stop phrase** (separate from the 3 events), from the goal and the route's final heading:
1. `near`: a salient visible object (candidate rules (b)(c)) within 1.0 m of the goal;
   relation from its position vs the final heading: `in front of` (ahead), `just past`
   (behind), `next to` (side). (`next to` removed -- superseded by Addendum A.)
2. `after_last`: the goal lies ≤ 1.5 m of route after the last event's closest point →
   "…and stop just after it".
3. `distance`: a visible object within 4 m on the line of approach → "about N metres
   before/past X", N ∈ {1, 1.5, 2, 3, 4} (nearest), written in words.
4. none → drop the route (`no_stop_anchor`).

(Rule 3's "about N metres" and the 4 m limit are superseded by Addendum A: coarse amounts
"just / a little / well", no limit past the last obstacle.)

A route with 0 events is kept when it has a stop phrase.

**Depth-trimmed outline (group B).** `depth.png` is min-max normalised per frame with the
range not stored (`spaceitup-habitat/src/roverbench/dataset/writer.py:42`). Per frame, fit
pixel value → metres by least squares on visible free-floor pixels (planar depth
`fwd = 0.5·CAM_HEIGHT_M / (v − 0.5)`). The object's mask = pixels in its floor-to-1 m column
whose depth lies in the object's own `fwd` range ± 0.3 m; outline = hull of the mask; naming
crop = mask box + 15% margin. Fit residual > 0.3 m → column outline as today, object flagged
`depth_fallback`.

**Rows.** `events_*.jsonl`: `id, sample_dir, candidate_index, kind (labelset kind: `instructed` = ground
truth, `opposite` = second route), pair_id,
events[{type, side, object_ids, arc_m}], stop{type, relation, object_id, distance_words},
reason`. `objects_*.jsonl`: `object_id (frame + blob), polygon, bbox, crop_box, kind_hint
(object|opening), depth_fallback, area`. Drop reasons are counted and printed:
`no_state`, `bad_candidate`, `no_stop_anchor`.

## Cosmos stages (`descriptions.py`, prompt d1)

**name** — one call per object row, shared by both routes of a frame. One composite image
(multi-image requests are untested on the Cosmos server): left, the full frame with the object
outlined and the rest dimmed (as i4); right, the crop enlarged. Reply `{"name", "kind"}`. Rejected: > 4 words,
generic ("object", "thing", "area", "item"), drawing words ("outline", "box", "highlighted",
"marked"), kind ≠ event need (a `through` gap named as an object). One retry with seed + 1;
still rejected → object unnamed → routes needing it dropped (`unnamed_object`).

**compose** — one text-only call per route (new `labelling_io.chat_text`). Input: the ordered events with names and sides,
the stop phrase, one of the 4 i4 speaking styles and a sentence-shape hint (stable hash of
the id). Output: 1–2 sentences, ≤ 45 words. Text rules (`description_rejects`):
- every name's head noun appears, in route order;
- each `pass` has its side word in its own clause and no opposite side word there;
- `between` / `through` wording present;
- stop relation + name present; `distance` stops carry their distance words;
- no drawing references; no egocentric "on/to your left/right" (as i4).
One retry; still rejected → template assembly, `instruction_source: "template"`.

## Gates

1. **Geometry audit** (CPU; `audit_instruction_labels.py --mode events`): 100 random train
   routes with 30 pair routes mixed in; frame with numbered outlines, route, stop anchor and a
   template sentence. Marks: true / wrong object / misses an obvious obstacle / wrong stop /
   can't tell. Pass ≥ 90% true. Thresholds (PASS_M, MIN_AREA, MAX_EVENTS, stop radii) are
   tuned here and frozen before any GPU run.
2. **Label gate** (GPU, `STAGE=gate`): names for the objects of the first 300 eval routes,
   then compose. Machine checks, then human audit page (`--mode d1`): correct / wrong name /
   wrong side / wrong stop / unnatural. Pass ≥ 90% correct.
3. **Full run** (`STAGE=full`, train + eval, `--resume`, re-submit after a 14 h TIMEOUT).
   One SLURM slot; never more than 3 jobs.

## Dataset (`finalize-descriptions`)

One record per described route; answer = full path (`format_answer_goal`). Prompt form by
stable hash of the frame id (so both routes of a pair share it): 70% B form
`description_entry_prompt(start, text)`, 30% goal form `instruction_endpoints_prompt`.
Eval files (scene-disjoint round-4 eval):
- `eval_desc_entry.json` — every described eval route, B form (headline);
- `eval_desc_goal.json` — same routes, goal form;
- `eval_desc_pairs.json` — frames with both routes described, B form;
- `eval_desc_shuffled.json` — B form with a description from another frame (text-reading
  control).

## Evaluation (`eval.py`)

- Point error and **endpoint error** (the goal is not given in B form), reported next to a
  **constant-goal baseline** (every prediction ends at the set's mean goal): 41% of goals sit at
  x = 0.5 and goal y spans only 0.545–0.629 (10th–90th pct), so a text-blind guess is close.
- **Description followed**: back-project the predicted image path to the floor (level camera,
  inverse of `landmarks.Camera.uv`) on the frame's map; `pass` → obstacle on the stated side
  at the closest point; `between` → both on their sides; `through` → path crosses the gap
  segment; stop → endpoint within 0.75 m of the true goal (superseded by Addendum A: image space,
  within `STOP_TOL_UV = 0.03` of the GT end; also `wall` events, multi-object passes, a graze
  under `PASS_TIE_M = 0.15` m ties, and `gt_follow_rate` = the check on the GT path, the
  ceiling). A route is followed when all hold.
- **Pair success**: both routes of a frame followed (text-blind ≈ 0%).
- Shuffled-text control: followed-rate should fall well below the matched set.
- mp4 videos of predicted paths for a fixed sample of eval frames (standing rule;
  `scripts/render_desc_videos.py`, ffmpeg like `render_clip_predictions.py`).

## Testing

Test-first, CPU only, synthetic `state.npz` grids + tiny images in `tests/test_route_events.py`
and `tests/test_descriptions.py`:
- side per obstacle, S-route with two events of opposite sides, top-3 by score, arclength order;
- `between` / `through` merging; the grazed-wall case (group A: a pillar beats a wall the line
  only grazes); occluder below the frame no longer hides a far obstacle (group C);
- each stop rule and the fallback order; `no_stop_anchor` drop (distances in metres here are
  superseded by Addendum A: coarse amounts, thresholds 1.5 / 3 m);
- depth calibration recovers a known range; trimming drops background pixels; poor fit →
  `depth_fallback`;
- text rules: accept correct sentences, reject missing/misordered names, swapped sides,
  missing stop, drawing words; template fallback always passes its own rules;
- finalize: 70/30 split, pairs share a form, scene disjointness; eval metric on hand-made paths.

## Out of scope

Training runs (later, `slurm/train_habitat.sbatch data/prepared_habitat_instr_v3/train_desc.json …` as for v2), new Habitat rendering (metric depth, extra
routes), HM3D `objects.json` checks (informative only), dropping group D frames, the
"panel"→"wall" rename question (superseded: names come from crops now).

## Addendum A — after the Gate 1 audit (2026-10-01)

Gate 1 (https://claude.ai/artifact/KQNZv2axTcX7WBZUHV6MEm, 100 train routes): **72 true**, 15
wrong stop, 7 wrong object, 4 misses an obstacle, 2 can't tell — below the 90% bar. Causes:
"next to" stops wrong 12/29 (24% of the dataset); rough-metre stops wrong 11/54 and not judgeable
by a person; wrong objects mostly from imprecise outlines (convex hulls over large wall blobs and
over several things at one depth); 30% of routes read "left of A, then right of B" (opposite
passes 1.2–2.9 m apart) where a person says "between A and B".

User decisions:
1. **Coarse stop amounts** replace metres: distance stops say "stop just / a little / well
   past (or before) X" — just < 1.5 m, a little 1.5–3 m, well > 3 m. The "past the last
   obstacle" fallback has no upper limit, so long routes are described instead of dropped.
   The `next to` near relation is removed.
2. **Walls and doors are named, in wall language**: a long blob the route runs alongside is a
   `wall` event, "keep the white wall on your left" / "follow the glass partition on your
   right". For walls the side word is the rover's own side (wall on your left), for objects it
   stays the side the route passes on ("go right of the table"). Doorways stay `through`.
   A wall-like blob is never a `pass` object, and is a stop anchor only "in front of".
3. **Between**: two consecutive passes (objects or walls) on opposite sides with nothing in
   between merge into "go between A and B" (any arc gap up to 3 m); two consecutive object
   passes on the same side merge into one pass of two objects ("go left of the A and the B").
4. **Tighter outlines**: an object's outline is the hull of the depth-band mask's connected
   pieces that contain its projected cells, not of the whole band; a wall's outline covers only
   the stretch within 0.8 m of the route.

Eval: a `wall` event with side s is followed when the wall stays on the rover's side s
(the obstacle is on the route's side s); a multi-object pass needs every object on the stated
side. Stop checking is unchanged (image space).

Implemented as (README "Route descriptions"): outlines use the depth band minus floor pixels and
only the connected pieces that hold the object, `MIN_AREA = 0.005`; names are report-only in the
labelling job (two passes), compose runs twice and only compose pass 2 carries `--gate`;
`OVERWRITE=1` restarts the gate. Eval: stop judged within `STOP_TOL_UV = 0.03` of the GT end in
image space, a graze under `PASS_TIE_M = 0.15` m ties (per member for `between` too),
`gt_follow_rate` is the ceiling, and the constant-goal baseline stays. 94–98% of routes described
(smoke probes).

Clarifications (final review, 2026-10-01):
- (a) "A wall-like blob is never a `pass` object" applies to blobs the route runs alongside (a
  `wall` event). A long blob the route only comes near, not alongside for `ALONG_MIN_M` (1 m),
  stays a `pass` object.
- (b) A wall-like blob ahead on the line of approach can anchor a `before` distance stop: "stop
  just / a little before the wall" (rule 4 refuses only "past" for wall-like blobs).
- (c) "Wall" is any long blob the route runs alongside (median outline diagonal 2.7 m): it
  includes long furniture (sofas, counters, beds), and a wall-like near "in front of" anchor is
  also kind wall. The wall naming prompt says "something long that the rover drives alongside or
  stops in front of: a wall, a partition, a door in a wall, or a long piece of furniture such as
  a sofa or a counter"; a name of kind wall or object is accepted, opening is rejected.
- A `through` event ends at its crossing (`end_arc_m = arc_m`), never at a joined wall's far
  end, so "just after the doorway" means just after the door.
- An outline with no seeded piece hulls the floor-free depth band (`depth_fallback` false); the
  column outline is used only when nothing but floor is at the object's depth. A third seed at
  height min(top, 0.25 m) catches objects lower than top/2.
- `events_meta.json` carries `"schema": "d1-addendumA"`; the labeller and the events/d1 audit
  refuse events files without it.

## Addendum A2 (2026-10-02, Gate 1 r2)

- **Rule.** A blob whose closest distance to the usable route (arc > `START_SKIP_M`) is below
  `TOUCH_M = 0.2` m is a "touching" blob: `candidates()` marks it and `build_events` lets it join only
  a `through` event (a doorway jamb), never a pass / wall / between. The route runs *through* it,
  so the left/right side at the single closest-approach point (which lies on the obstacle) is
  undefined. Touching singles are dropped before the left-to-right merge, so their clear
  neighbours merge as before.
- **Evidence.** Gate 1 r2 (89/100): 5 of the 11 flagged routes (4 "wrong relative position", 1
  "wrong object") have such a blob (00660: the rack is on the rover's right at 36 route points,
  on its left at 6; the closest point was one of the 6). Over 800 random train routes 8.1% have
  one, 64/67 of those events are walls, and the distance is bimodal: 65 events < 0.05 m, none in
  0.05-0.2 m, the rest >= 0.2 m (the planner keeps ~0.45-0.5 m clearance elsewhere).
- **Doorway exception.** A blob the route threads through a gap (`_threaded_gap` not None and
  extent >= `WALL_EXTENT_M`) is never touching. A doorway of two separate wall halves whose jamb is
  within `TOUCH_M` of the route still becomes a `through` (a touching blob may pair with its
  neighbour into a through, ignoring the side test); doorway behaviour is unchanged.
- **Zero-event routes must be straight.** A route with no events is phrased "Go straight ahead,
  and stop ...". Of the 114 zero-event train routes (Addendum A run) the max offset from the
  entry->goal line has median 0.71 m (77 > 0.5 m, 36 > 1 m), so the sentence is false for most,
  and the touch rule creates more (00423, 00599 became "go straight ahead" while bending round
  a door / left-then-right). Decision 2026-10-02: a zero-event route whose max offset exceeds
  `STRAIGHT_MAX_M = 0.3` m is dropped, reason `no_events_bent`; straighter ones are unchanged,
  and routes with >= 1 event are unaffected whatever their offset.
- **Schema.** `events_meta.json` carries `"schema": "d1-addendumA2"`; the labeller and the
  events/d1 audit refuse events files built before this rule.
- **Not changed.** Depth-fallback outlines stay (fix B, dropping them, was declined); stop rules,
  merging and scores are untouched.

## Addendum A3 (2026-10-02, label gate r1)

- **Evidence.** Gate job 92114 (300 eval routes, events schema d1-addendumA2): accept 0.890 < 0.9
  (33 `unnamed_object`), template share 0.262 > 0.1.
  - Names: 27/377 objects rejected. 16 are doors / windows on object or wall outlines that Cosmos
    names well ("open doorway", "glass door", "window with blind") but types `kind: opening`, which
    we scored `kind:mismatch`. 3 are "<x> unit" names flagged generic ("white shelving unit").
    4 are thinking runs cut at max_tokens 2048 (parse failures). Offline rescoring of the stored
    raws with N1 + N2 recovers 14 of the 33 routes (accept ~0.937).
  - Compose: 70 routes fell back to the template; 96 of their 140 Cosmos replies never reached
    `</answer>` (cut at 2048 tokens) while Cosmos argued with a contradiction in the prompt ("keep
    the word left or right that goes with it" vs "for other objects never say 'your left/right'"),
    worst on `between` steps (24/55 fell back), which have no side.
- **Decisions.**
  1. **N1.** `name_rejects`: kind "opening" for need "wall" is not a `kind:mismatch` (doors /
     windows in a wall; the wall naming prompt already offers "a door in a wall"). Need "object" +
     kind "opening" and need "opening" + kind != "opening" stay mismatches.
  2. **N2.** A name whose head noun is "unit" is not generic when the word before it is in
     `UNIT_MODIFIERS` (shelving, shelf, storage, kitchen, wall, tv, display, corner, drawer,
     cabinet); "unit", "white unit", "large unit" stay generic.
  3. **C1.** `label_descriptions.py --max-tokens` default 2048 -> 4096 (both stages).
  4. **C2.** Compose prompt d2 (`PROMPT_VERSION = "d2"`): the contradicting sentences are replaced
     by "Keep every step in this order and keep each object's name. Copy each step's direction
     words exactly as written: in a 'keep the ... on your left/right' step, 'on your left' / 'on
     your right' stays that way, whatever the thing is (a wall, a sofa, a counter); in a 'go left
     of / right of' step, 'left of' / 'right of' stays that way; a step with 'between' or
     'through' has no left or right, so do not add one. Keep where to stop." (final review I1: the
     first d2 wording keyed the egocentric form on "after a wall",      but wall-kind events are
     often furniture -- a counter, a sofa -- so Cosmos again asked whether it      was a wall;
     34/70 gate-r1 fallbacks came from that group.) The naming prompt is unchanged (its rows also
     carry `prompt_version` d2, the constant is shared). Text rules and the template are unchanged.
- **Declined.** N1+: accepting `opening` for need "object" too (objects typed as openings stay
  mismatches). File names `*_d1.jsonl` and `--mode d1` are kept.

## Addendum A4 (2026-10-02, namer A/B)

- **Audit of gate r2** (job 92136): 84 correct / 11 wrong name / 2 unnatural / 2 unsure / 1 wrong
  stop of 100 -- below the 90% gate. No side errors. The wrong names mostly name a small detail
  inside the outline instead of the outlined thing ("metallic door handle", "light switch",
  "electrical outlet cover", "carved wooden chair leg", "white door panel"); 6/11 are `between`
  events. Both unnatural labels come from the style "a command that also says briefly why or where
  it leads" ("... to avoid the obstacle").
- **Rejected.** A part-word reject list, scored offline: it catches 3/11 wrong names and drops
  3/84 correct ones.
- **Decisions** (user chose Gemma 3 12B as the namer, Cosmos3-Nano d2 still composes):
  1. `NAMER=cosmos|gemma` in `slurm/label_descriptions.sbatch` (default cosmos, unchanged);
     `gemma` serves `google/gemma-3-12b-it` for the two name passes, stops it, then serves
     Cosmos3-Nano for the compose passes; outputs carry a `_gemma` tag suffix
     (`names_gate_gemma_d1.jsonl`, `desc_gate_gemma_d1.jsonl`). Same `--max-tokens` for both.
  2. Naming prompt gains, after instruction 1: "Name the whole thing the outline covers, not a
     small part on it (a handle, a switch, a socket, a leg)." Same prompt for both namers.
  3. `D1_STYLES` = shared `STYLES` minus "a command that also says briefly why or where it
     leads", plus "a calm, step-by-step instruction" (stable per route id); shared `STYLES` (i4)
     unchanged.
  4. `PROMPT_VERSION = "d3"`.
  5. `audit_instruction_labels.py --mode d1 --ids FILE` re-audits exactly the routes of the
     previous page (file order, `--n` ignored) and prints how many ids were not passed, so the
     Gemma-named run compares directly against the 100 routes already marked.
  6. The naming image is letterboxed to a square (black bands above and below the two panels) so
     Gemma's 896x896 resize keeps each panel's aspect (final review I1).

## Addendum A5 (2026-10-02, mixed namer)

- **Evidence** (same-route re-audit of 99 routes): Gemma names 84 ok vs Cosmos names 83 ok; the
  errors are complementary (only 2 routes wrong under both). Gemma's characteristic new error is
  two different objects of one route getting the same name ("go between the wooden bench and the
  wooden bench", "keep the wooden dining table on your left, then ... on your right"): 30/296
  passed Gemma routes vs 6/282 Cosmos. Offline estimate on the audited routes: Gemma names, Cosmos
  names on same-name routes (when distinct), else drop -> 86/97 ok (88.7%), 8 fallbacks, 2 drops.
  The user chose this setup; Cosmos3-Nano keeps composing.
- **Clash rule.** `name_clash(row, names)` (descriptions.py): a route clashes when two DIFFERENT
  object ids among its event members + stop object have the same `head_noun` under the names used.
  Unnamed objects are not a clash (that stays `unnamed_object`). `label_route(..., fallback_names)`:
  all objects named in `names` and no clash -> `names` (`"namer": "primary"`); else if
  `fallback_names` names all of them with no clash -> those (`"namer": "fallback"`); else
  `unnamed_object` if some object is unnamed in both, `name:clash` if names exist but clash
  (not passed). A Cosmos-only run now also rejects its clashing routes as `name:clash`.
  `--stage compose --fallback-names FILE`; an object named in either file is not a retryable gap.
  `summarize_d1` reports the `namer` counts of passed rows.
- **`NAMER=mix`** in `slurm/label_descriptions.sbatch`: Gemma name passes ->
  `names_<tag>_gemma_d1.jsonl`; stop Gemma; serve Cosmos (one switch); Cosmos name passes ->
  `names_<tag>_d1.jsonl`; Cosmos compose with `--names names_<tag>_gemma_d1.jsonl
  --fallback-names names_<tag>_d1.jsonl` -> `desc_<tag>_mix_d1.jsonl`. Gate: eval only (gated on
  compose pass 2); full: eval then train for each phase. `NAMER=cosmos|gemma` are unchanged.
- **Full run:** `NAMER=mix STAGE=full sbatch slurm/label_descriptions.sbatch` (re-submit the same
  line after a TIMEOUT; never `OVERWRITE=1` on a full run).
- **Finalize:** `uv run scripts/prepare_habitat_instr.py finalize-descriptions --desc-suffix _mix`
  (default suffix "" keeps reading `desc_<split>_d1.jsonl`).
