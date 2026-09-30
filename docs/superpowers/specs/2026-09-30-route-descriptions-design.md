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

**Occlusion fix (group C).** Occupied cells within `NEAR_CAMERA_M = 0.35` m of the camera are
neither occluders nor candidates: the rover's start pose is collision-free, and in 00372 such a
strip (0.31 m ahead, bare floor in the image) hid the armchair from every ray.

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
   (behind), `next to` (side).
2. `after_last`: the goal lies ≤ 1.5 m of route after the last event's closest point →
   "…and stop just after it".
3. `distance`: a visible object within 4 m on the line of approach → "about N metres
   before/past X", N ∈ {1, 1.5, 2, 3, 4} (nearest), written in words.
4. none → drop the route (`no_stop_anchor`).

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
  segment; stop → endpoint within 0.75 m of the true goal. A route is followed when all hold.
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
- each stop rule and the fallback order; `no_stop_anchor` drop;
- depth calibration recovers a known range; trimming drops background pixels; poor fit →
  `depth_fallback`;
- text rules: accept correct sentences, reject missing/misordered names, swapped sides,
  missing stop, drawing words; template fallback always passes its own rules;
- finalize: 70/30 split, pairs share a form, scene disjointness; eval metric on hand-made paths.

## Out of scope

Training runs (later, `slurm/train_habitat.sbatch data/prepared_habitat_instr_v3/train_desc.json …` as for v2), new Habitat rendering (metric depth, extra
routes), HM3D `objects.json` checks (informative only), dropping group D frames, the
"panel"→"wall" rename question (superseded: names come from crops now).
