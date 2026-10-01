# Route Descriptions — Addendum A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the Gate 1 decisions: coarse stop amounts, wall events in wall language, "between" merging, tighter outlines.

**Architecture:** Changes stay inside the existing d1 modules: geometry in `src/rover_vlm/route_events.py`, wording/rules/prompts/finalize in `src/rover_vlm/descriptions.py`, the follow check in `src/rover_vlm/eval.py`. Row shapes gain one event type (`wall`), multi-object `pass` events, and a stop `amount`; everything else keeps its shape.

**Tech Stack:** Python 3.12 (uv), numpy, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-route-descriptions-design.md` — read **Addendum A** at the end first; it is binding.

## Global Constraints

- Shared 96-core box: export `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` before any python/pytest; use `uv run --no-sync`.
- Never load a model or contact an endpoint. Don't touch `data/` (the controller reruns the events stage afterwards).
- Test-first for every logic change; full suite green before each commit.
- The user's uncommitted files (`README.md` hunk, `scripts/evaluate.py`, `NOTES.md`, untracked `scripts/*`, `slurm/*`): never `git add -A`/`.`/`commit -a`; add files by name.
- Side conventions (binding): object `pass` side = the side the ROUTE passes the object on ("go right of the table" = table on the route's left). `wall` side = the side of the ROVER the wall is on ("keep the wall on your left" = wall on the route's left). In `candidates()`, `side == "left"` means the obstacle is on the route's right, so a wall's side is the OPPOSITE of its candidate `side`.
- Commit messages end with: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`

## Review Focus

1. A wall blob the route runs alongside AND passes through a gap of → `through` wins (doorway), not `wall`.
2. Merging never crosses a `through` event and never merges two `wall` events into "between" when they are the same blob.
3. A wall-like blob as the stop anchor gives only "in front of"; otherwise that anchor is skipped and the next stop rule runs.
4. An egocentric phrase ("on your left") next to an OBJECT name is still rejected; next to a WALL name it is required to match the wall's side.
5. Old rows (distance stops with `distance_m`, no `amount`) are not silently mis-rendered: `stop_text` raises a clear error if `amount` is missing.

---

### Task 13: Tighter object outlines (connected depth pieces)

**Files:** Modify `src/rover_vlm/route_events.py` (`object_outline`, new constant `WALL_KEEP_M` is Task 14's); Test `tests/test_route_events.py`.

**Interfaces:** `object_outline(frame, vis_xz, top=OBJECT_TOP_M)` keeps its signature and return shape.

Behaviour: after computing the depth-band mask `keep` (as today), label its 8-connected pieces on a mask downsampled so its longer side is ≤ 128 px (use `landmarks.components`; nearest-neighbour downsample by striding). Keep only the pieces that contain at least one "seed" pixel: the projections (`frame.cam.uv`) of `vis_xz` at heights 0, `top/2` and `top`, clipped to the image. If no piece holds a seed, fall back to the column outline (`depth_fallback=True`). The outline is the convex hull of the kept pieces' pixels (full-resolution coordinates, `(x+0.5)/W`).

- [ ] Step 1: failing test — synthetic frame with the box of `_box_frame` plus a second region at the SAME depth band but separated in the image (e.g. a block 0.4 image widths to the side, painted at depth 1.3 in `_depth_image`): the trimmed outline's bbox must NOT extend into the second region (assert `bbox[2] < second_region_u0`), and `depth_fallback is False`.
- [ ] Step 2: run → FAIL (hull covers both).
- [ ] Step 3: implement.
- [ ] Step 4: run `tests/test_route_events.py` (all existing outline tests must still pass) and the full suite.
- [ ] Step 5: commit `feat: object outline keeps only the depth pieces holding the object`.

---

### Task 14: Walls, merging and coarse stops (geometry)

**Files:** Modify `src/rover_vlm/route_events.py`; Test `tests/test_route_events.py`.

**Interfaces produced (later tasks rely on these exact names):**
- New constants: `ALONG_MIN_M = 1.0` (route length within `PASS_M` of a blob for it to be a wall the route runs alongside), `WALL_KEEP_M = 0.8`, `MERGE_ARC_M = 3.0`, `AMOUNT_STEPS = ((1.5, "just"), (3.0, "a little"), (inf, "well"))`.
- Candidate dict gains `wall: bool` (extent ≥ `WALL_EXTENT_M` and along-length ≥ `ALONG_MIN_M`) and `along_m: float`. For walls, `vis_xz`/`outline` use only cells within `WALL_KEEP_M` of the route.
- Event types: `pass` (side = route side; `members` may hold 1 or 2 candidates — same side, objects only), `wall` (side = rover side of the wall = opposite of the candidate `side`; one member), `between` (two members, side None; may include a wall), `through` (unchanged).
- Stop row: `{"type", "relation", "amount", "distance_m", "object"}` — `amount` ∈ {"just", "a little", "well"} for `distance` stops, None otherwise; `distance_m` is now the raw rounded metres (info only). `near` relations: "in front of" | "just past" only.

Behaviour:
1. `build_events`: classify each candidate first. A candidate with a crossing `gap` stays `through` (Review Focus 1). A `wall` candidate becomes a `wall` event. Other candidates are `pass` events. The old 0.5 m `between` pairing is removed. After building the route-ordered list, merge ADJACENT events (no event between them) whose arc gap ≤ `MERGE_ARC_M`:
   - `pass`+`pass` with opposite sides, or `pass`/`wall` + `pass`/`wall` whose OBSTACLES are on opposite sides of the route (a `wall` with side "left" has its obstacle on the route's left; a `pass` with side "left" has its obstacle on the route's right) → `between` with both members; never merge two members of the same blob (Review Focus 2); never merge across or with `through`.
   - `pass`+`pass` (objects, same side) → one `pass` with two members.
   Score of a merged event = sum. Then keep the top `MAX_EVENTS` by score, returned in route order (as today).
2. `stop_phrase`: near rule — drop the "next to" relation (an anchor whose lateral offset dominates is skipped; try the next-nearest near anchor, then fall through). A wall-like anchor (extent ≥ `WALL_EXTENT_M`) is accepted only when the relation is "in front of" (Review Focus 3). Last-event fallback: no `STOP_FAR_M` limit — any remaining length > `AFTER_LAST_M` gives `{"type": "distance", "relation": "past", "amount": amount(remaining), "distance_m": round(remaining, 1)}`. Cone rule unchanged except `amount` instead of a step. `DISTANCE_STEPS` is removed (grep for users; descriptions.py uses `DISTANCE_WORDS` — Task 15 replaces it).
3. `describe_route`: event rows `{"type", "side", "objects", "arc_m", "gap_xz"?}` (objects list may have 2 for a merged pass); objects get `kind_hint` "wall" for wall members (also when inside a `between`), "opening" for openings, else "object". Stop row as above.

- [ ] Step 1: failing tests (synthetic grids, reuse `write_sample`/`_free`):
  - long wall x=3.4 from z 0.5 to 4.5 with the route running alongside at x≈2.9 → one `wall` event with side "left" (heading +z, the rover's right is −x, so a wall at larger x than the route is on the rover's LEFT). Assert side "left" and that its outline bbox is narrower than the column outline of all wall cells.
  - S-route fixture `S_ROUTE`/`_s_state` (box A then box B on opposite sides, ~2.1 m apart) → ONE `between` event with both boxes.
  - two boxes on the same side 1 m apart → one `pass` with two objects.
  - wall + box on opposite sides → `between` (members kind wall + object in describe_route objects' kind_hint).
  - doorway fixture → still `through` (not wall).
  - near stop with dominating lateral offset → not "next to" (falls through to after_last/distance).
  - remaining 6 m after the last event → distance/past/"well"; 2 m → "a little"; 1.2 m → handled by after_last as today (≤ 1.5).
  - wall anchor beside the goal → not a near stop.
- [ ] Step 2: run → FAIL. Step 3: implement. Step 4: update existing tests whose expectations the addendum changes (the old `between` 0.5 m test, distance-step tests) — list each changed test and why in the report. Step 5: run the three real-frame regressions (they must still find the named obstacle in some event, `pass` or `between`), full suite. Step 6: commit `feat: wall events, between/pass merging, coarse stop amounts`.

---

### Task 15: Wording, rules, prompts, finalize and eval for the new events

**Files:** Modify `src/rover_vlm/descriptions.py`, `src/rover_vlm/eval.py`; Test `tests/test_descriptions.py`, `tests/test_description_eval.py`.

Behaviour:
1. Templates (`event_phrase`, `stop_text`, `template_description`):
   - `pass` 1 object: "go {side} of the {A}"; 2 objects: "go {side} of the {A} and the {B}".
   - `wall`: "keep the {A} on your {side}".
   - `between`: "go between the {A} and the {B}" (unchanged).
   - stop `distance`: "stop {amount} {relation} the {X}" ("stop well past the sofa"); `stop_text` raises `ValueError("distance stop without amount")` when `amount` is missing (Review Focus 5). `near` unchanged ("in front of", "just past"). `DISTANCE_WORDS` and the metre rule are removed.
2. `description_rejects`:
   - Amount words for distance stops (checked in the stop clause): just → (just | right); a little → (a little | a bit | slightly | shortly); well → (well | far | a good way). Missing → "stop:amount". Keep `STOP_RELATION_WORDS` for the relation; remove the ("near","next to") entry.
   - Multi-object pass: one side word owned by ANY of its names satisfies it; no opposite word on any of them.
   - Wall event: its side word (bound by the existing owner rule) must equal the wall side; an egocentric phrase ("on/to your left/right") is allowed only when it is owned by a wall event's name and matches its side; anywhere else it is still "side:egocentric" (Review Focus 4). Implement by computing ownership for `_EGOCENTRIC` matches with the same next-name/previous-name rule before deciding.
   - Template invariant: extend the parametrized all-combinations test to every event type (pass1, pass2, wall left/right, between, through) × every stop type/relation/amount; all pass.
3. `compose_prompt`: the step lines come from `event_phrase` (so walls read "keep the white wall on your left"); add one sentence: "For walls, keep the words 'on your left/right' exactly as given; for other objects never say 'your left/right'."
4. Naming: `naming_prompt(need)` gains need "wall" ("The outlined region is part of a wall or a door in a wall that the rover drives alongside."); `name_rejects` with need "wall" accepts kind "wall" or "object" (a door), rejects "opening". `label_object` uses `kind_hint` as need (already).
5. `desc_check`: wall events carry `{"type": "wall", "side", "obstacles": [cells]}`; multi-object pass carries all objects' cells.
6. `eval._event_followed`: `wall` → followed when the obstacle is on the route's side `side` (i.e. `_pass_side` returns the OPPOSITE word) within `FOLLOW_NEAR_M`, with the `PASS_TIE_M` tie as for pass; multi-object `pass` → every group satisfies the pass rule.
- [ ] Step 1: failing tests for each numbered item (wall templates + rules both directions; egocentric next to an object still rejected; amounts accepted/rejected; multi-object pass; missing amount raises; eval wall followed/not followed with the side mapping; eval two-object pass).
- [ ] Step 2: run → FAIL. Step 3: implement. Step 4: full suite. Step 5: commit (one or two commits).

---

### Task 16: Docs

**Files:** Modify `README.md` (route descriptions section only — index-only commit, ruling R2: build the index blob from `HEAD:README.md` + your edit, leave the user's framing-row hunk uncommitted), the spec's Testing/Evaluation text where Addendum A or the final-review fixes changed numbers (stop tolerance 0.03 uv, graze 0.15 m, `gt_follow_rate`, sbatch step order).

- [ ] Update the README "Route descriptions (d1)" section: wall events and their side convention, between/pass merging, coarse amounts, no "next to", tighter outlines, the corrected labelling step 3 (names report-only, compose twice with the gate on pass 2), metric text (stop within 0.03 of the GT end in image space, grazing passes < 0.15 m tie, `gt_follow_rate`).
- [ ] Verify `git diff --cached README.md` shows only your section, `git diff README.md` only the user's row; commit.
