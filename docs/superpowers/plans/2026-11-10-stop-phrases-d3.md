# Stop Phrases (d3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Visible route stops say where the rover ends up ("in front of / behind / next to X", or a place such as
"in the middle of the room") instead of distance words, and the d3 dataset, labels, eval and steering are built on it.

**Architecture:** A new pure-numpy module `stop_places.py` reads the place from the frame's occupancy grid;
`route_events.stop_d3` tries an object relation first, then the place, and `describe_xz(..., stops="d3")` uses it
for visible ends only (hidden behind / turn endings unchanged). Text, seeding, eval, steering and the audit page
gain the new stop types next to the d2 ones; d2 stays the default everywhere, so v4 can still be rebuilt.

**Tech Stack:** Python 3.12, numpy, PIL, pytest; `uv run`; SLURM sbatch files on thor for the GPU runs.

**Spec:** `docs/superpowers/specs/2026-11-10-stop-phrases-d3-design.md` (read it with this plan).

## Global Constraints

- `uv run` only; tests are CPU-only. Before any compute: `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`.
- Never load Qwen / Cosmos / Gemma on the login node. Every `sbatch` is asked for first; at most 3 jobs on thor.
- Never `git add -A`, never stash: the user's uncommitted `README.md` and `scripts/evaluate.py` edits are not ours.
  README changes are committed **index-only**: write the hunk as a patch against `HEAD:README.md`, then
  `git apply --cached <patch>` and `git apply <patch>`; never `git add README.md`.
- `CLOSE_M = 1.5` (goal to the object's nearest visible cell). Relation order: behind → in front of → next to.
  Place order: doorway → corner → hallway end → against a wall → middle of the room. First fit wins.
- Walls (blobs with extent ≥ `WALL_EXTENT_M` = 1.5 m) are never anchors of an object relation.
- No distance words in any visible d3 stop: "a little", "a bit", "slightly", "well", "far", "a good way" before
  past/before/beyond/after; "just past / just after / just beyond"; "short of"; metres / meters / steps.
- Unknown grid cells (`landmarks.UNKNOWN`) are **no evidence**: never counted as free, never as occupied.
  (Measured 2026-11-10: a median 39% of the 1.5 m disc around a visible eval stop is unknown.)
- d2 behaviour is the default (`stops="d2"`, `--round d2`): v4 outputs must stay reproducible.
- Hidden "behind" endings, turn endings, route events and object names do not change.
- Full suite green at the end of every task (1,249 tests at the start: `uv run pytest -q`).

## Review Focus

1. **A goal whose surroundings are mostly unknown** (the real-map case): no place may fire on unknown space; a
   1.5 m disc that is < 60% free is never "middle of the room". → Task 1 `test_room_middle_needs_known_free_space`.
2. **A goal near the edge of the grid array**: cells outside the array are unknown, no IndexError.
   → Task 1 `test_goal_at_grid_edge_is_safe`.
3. **A reused d2 text with a hidden behind or turn ending** must not be flagged by the d3 re-check (it has no stop
   clause). → Task 3 `test_d3_rejects_leaves_behind_and_turn_texts_alone`.
4. **A d2 text with two stop words** ("Stop by the sofa ... then stop well past it"): only the final clause is
   replaced. → Task 4 `test_rewrite_stop_replaces_only_the_final_clause`.
5. **Two steering cuts that differ only in their place** must count as different texts.
   → Task 6 `test_stop_key_tells_places_apart`.

---

### Task 1: Place geometry (`stop_places.py`)

**Files:**
- Create: `src/rover_vlm/stop_places.py`
- Test: `tests/test_stop_places.py`

**Interfaces:**
- Consumes: `rover_vlm.landmarks.Grid` (`.s`, `.mpp`, `.rc(xz)`, `.xz(rc)`), `UNKNOWN, FREE, OCC`.
- Produces:
  - `PLACES = ("doorway", "corner", "hallway_end", "wall", "room_middle")`
  - constants `N_DIRS = 16`, `RAY_MAX_M = 3.0`, `DOORWAY_HALF_M = 0.9`, `DOORWAY_DEPTH_M = 0.6`, `CORNER_M = 1.5`,
    `CORNER_DEG = (60.0, 120.0)`, `CORNER_OPEN_DEG = 30.0`, `HALL_MAX_W = 2.2`, `HALL_BACK_M = (0.0, 1.0, 2.0)`,
    `HALL_END_M = 1.5`, `HALL_OPEN_W = 3.2`, `WALL_M = 1.0`, `ROOM_CLEAR_M = 1.5`, `ROOM_MIN_FREE = 0.6`
  - `ray_hits(grid, xz, head_xz, n_dirs=N_DIRS, max_m=RAY_MAX_M) -> list[tuple[float, str, float]]`:
    `(angle_deg, outcome, dist_m)` per direction; angle 0 = `head_xz`, +90 = right (`right = (-h_z, h_x)`, as
    `route_events.stop_phrase`); outcome `"occ"` (first OCC cell), `"unknown"` (first UNKNOWN or off-array cell),
    `"free"` (reached `max_m`).
  - `width_across(grid, xz, head_xz) -> float | None`: left + right OCC distances when both side rays hit OCC.
  - `place_at(grid, route_xz) -> dict | None`: `{"place": str, "measures": dict[str, float]}` (≤ 6 rounded floats),
    heading from the route's last 0.5 m.

Rules (thresholds are starting values; Task 9 calibrates them):
- doorway: OCC on both sides (rays ~±90°) each ≤ `DOORWAY_HALF_M`, and `DOORWAY_DEPTH_M` ahead **and** behind
  along the heading the same side rays are not both OCC within `DOORWAY_HALF_M` (a short gap, not a corridor).
- corner: two OCC hits ≤ `CORNER_M` whose angles differ by `CORNER_DEG`, and neither has an OCC hit ≤ `CORNER_M`
  within `CORNER_OPEN_DEG` of its opposite direction (a dead-end corridor is not a corner).
- hallway_end: `width_across` ≤ `HALL_MAX_W` at each point `HALL_BACK_M` back along the route, and within
  `HALL_END_M` ahead either the ahead ray hits OCC or the width there is ≥ `HALL_OPEN_W`.
- wall: exactly one cluster of OCC hits (adjacent directions) ≤ `WALL_M`.
- room_middle: no OCC hit ≤ `ROOM_CLEAR_M`, and ≥ `ROOM_MIN_FREE` of the cells in the `ROOM_CLEAR_M` disc are FREE.

Test scenes: `Grid(state, 0.05, 0.0, 0.0)` on a 100×100 grid (5 m × 5 m; row ← z, col ← x), route straight along +z
from (2.5, 0.2) to the goal. `_free()` = all FREE.

- [ ] **Step 1: Write the failing tests**

```python
def test_room_middle_in_open_space():
    assert place_at(Grid(_free(), .05, 0, 0), [(2.5, .2), (2.5, 2.5)])["place"] == "room_middle"

def test_room_middle_needs_known_free_space():                      # Review Focus 1
    s = np.full((100, 100), UNKNOWN, np.uint8); s[:, 46:54] = FREE   # only a 0.4 m strip seen
    assert place_at(Grid(s, .05, 0, 0), [(2.5, .2), (2.5, 2.5)]) is None

def test_goal_at_grid_edge_is_safe():                                # Review Focus 2
    out = place_at(Grid(_free(), .05, 0, 0), [(2.5, .2), (2.5, 4.95)])
    assert out is None or out["place"] in PLACES

def test_wall_beside_the_goal():         # wall cols 32-34 (x 1.6-1.7), all z; goal 0.8 m from it
def test_corner():                       # walls x 1.7 (cols 32-34) and z 3.3 (rows 66-68); goal (2.5, 2.5) -> "corner"
def test_hallway_end():                  # walls x 1.75 / 3.25 for z 0-4, end wall z 4.0 (rows 80-82); goal (2.5, 3.0)
def test_dead_end_corridor_is_not_a_corner():   # same scene: place == "hallway_end", never "corner"
def test_wide_corridor_is_not_a_hallway():      # walls x 1.0 / 4.0, end wall 1.0 m ahead -> place != "hallway_end"
def test_doorway():                      # wall rows 40-42 with gap cols 40-60 (1.0 m); goal (2.5, 2.05) -> "doorway"
def test_past_the_doorway_is_not_a_doorway():   # same wall, goal (2.5, 3.0) -> place != "doorway"
def test_ray_hits_reports_unknown_not_free():   # UNKNOWN band ahead -> the 0-degree ray's outcome == "unknown"
def test_measures_are_small_and_rounded():      # every value a float, len(measures) <= 6
```

Each stub's comment gives the scene and the assertion (`place_at(...)["place"] == ...`).

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_stop_places.py -q`
Expected: FAIL (`ModuleNotFoundError: rover_vlm.stop_places`)

- [ ] **Step 3: Implement `stop_places.py`** (module docstring: what it does, spec link; pure numpy, no frame or
  camera). Rays march in `grid.mpp / 2` steps; any off-array cell is UNKNOWN.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_stop_places.py -q` → all pass; `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/stop_places.py tests/test_stop_places.py
git commit -m "feat: stop places from the occupancy grid (d3: doorway, corner, hallway end, wall, middle of the room)"
```

---

### Task 2: Object relations, `stop_d3`, and `describe_xz(stops="d3")`

**Files:**
- Modify: `src/rover_vlm/route_events.py` (constants block ~l.344; `stop_phrase` neighbourhood ~l.544–620;
  `describe_route` / `describe_xz` ~l.828–932; `EVENTS_SCHEMA` / `check_events_schema` ~l.935–947)
- Test: `tests/test_stop_d3.py`; update `tests/test_route_endings.py::test_schema_and_reason`

**Interfaces:**
- Consumes: Task 1 `place_at(grid, route_xz)`; existing `_blobs_near`, `_anchor`, `_wall_like`, `route_geometry`,
  `Frame.cam` (`.pos`), `load_frame`.
- Produces:
  - `CLOSE_M = 1.5`, `BEHIND_RAY_M = 0.6`; `REASONS` gains `"no_stop_place"`.
  - `object_relation(frame, goal, head, right, blobs) -> dict | None`:
    `{"type": "object", "relation": "behind" | "in front of" | "next to", "anchor": <_anchor dict, kind "object">}`.
    Wall-like blobs skipped; blobs farther than `CLOSE_M` skipped; nearest first; per blob the first relation that
    fits in the order behind → in front of → next to:
    behind = the blob's nearest visible cell lies within `BEHIND_RAY_M` of the camera→goal segment and nearer the
    camera than the goal; in front of = along > |lateral| (as `_near_stop`); next to = |lateral| ≥ along **and**
    the goal's along-coordinate lies within the blob's visible cells' along-extent.
  - `stop_d3(frame, route_xz) -> dict | None`: `object_relation`, else
    `{"type": "place", "relation": None, "place": str, "measures": dict, "anchor": None}` from `place_at`, else None.
  - `stop_kind(stop) -> str | None`: `"object:<relation>"`, `"place:<place>"`, None for every other stop type.
  - `describe_route(sample_dir, candidate_index, stops="d2")`, `describe_xz(..., stops="d2")`: with `stops="d3"`
    a visible end uses `stop_d3` (None → `_none("no_stop_place", n_events=...)`); hidden ends unchanged.
    The `st` record of a d3 stop adds `"place"` (None for object stops) and, for places, `"measures"`.
  - `EVENTS_SCHEMA_D3 = "d3-stops"`, `EVENTS_SCHEMAS = (EVENTS_SCHEMA, EVENTS_SCHEMA_D3)`;
    `check_events_schema` accepts either.

Scenes reuse `write_sample` / `_free` from `tests/test_route_turns.py` (pose (2.5, 0.2), heading +z, mpp 0.05).

- [ ] **Step 1: Write the failing tests**

```python
def test_in_front_of_an_object_ahead():          # 0.6 m box centred (2.5, 3.9); route ends (2.5, 3.0) -> relation "in front of"
def test_behind_an_object_between_camera_and_goal():   # post (2.0, 2.4); goal (2.5, 3.0), still visible -> "behind"
def test_next_to_an_object_beside_the_goal():    # post x 1.75-1.85, z 2.8-3.2; goal (2.5, 3.0) -> "next to"
def test_diagonal_object_is_not_next_to():       # post x 1.75-1.85, z 3.4-3.8 -> no object relation (stop["type"] == "place")
def test_walls_are_not_object_anchors():         # 2 m wide box 0.9 m ahead -> stop["type"] == "place"
def test_object_beyond_close_m_is_ignored():     # box 1.6 m ahead -> stop["type"] == "place"
def test_d3_stop_record_and_reason(tmp_path):
    row = describe_route(write_sample(tmp_path / "s", _free(), [[(2.5, .2), (2.5, 2.5)]]), 0, stops="d3")
    assert row["stop"]["type"] == "place" and row["stop"]["place"] == "room_middle" and row["stop"]["object"] is None
def test_no_place_drops_the_route_as_no_stop_place(tmp_path):    # UNKNOWN around the goal -> reason "no_stop_place"
def test_hidden_ends_keep_their_t2_endings_under_d3(tmp_path):   # BEHIND scene of test_route_turns -> same stop as stops="d2"
def test_d2_stays_the_default(tmp_path):         # describe_route(...) == describe_route(..., stops="d2")
def test_stop_kind():
    assert stop_kind({"type": "object", "relation": "next to"}) == "object:next to"
    assert stop_kind({"type": "place", "place": "corner"}) == "place:corner"
    assert stop_kind({"type": "distance", "relation": "past"}) is None
def test_schema_check_accepts_d2_and_d3(tmp_path):   # meta schema "d2-endings" and "d3-stops" pass; "x" raises SystemExit
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_stop_d3.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement** the interfaces above; extend the module docstring with a "d3 (2026-11-10)" paragraph.

- [ ] **Step 4: Run** `uv run pytest tests/test_stop_d3.py tests/test_route_events.py tests/test_route_endings.py tests/test_route_turns.py -q`, then `uv run pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/route_events.py tests/test_stop_d3.py tests/test_route_endings.py
git commit -m "feat: d3 visible stops — object relation within 1.5 m, else a place; describe_xz(stops='d3')"
```

---

### Task 3: Stop text, rejects and the eval check (`descriptions.py`)

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (`STOP_RELATION_WORDS` ~l.49, `stop_text` ~l.84, `description_rejects`
  ~l.268, `PROMPT_VERSION` ~l.367, `_check_ending` ~l.798; new `d3_rejects` next to `d2_rejects`)
- Test: `tests/test_stop_text_d3.py`

**Interfaces:**
- Consumes: Task 2 `stop_kind(stop)`.
- Produces:
  - `PLACE_TEXT = {"doorway": "in the doorway", "corner": "in the corner", "hallway_end": "at the end of the hallway",
    "wall": "against the wall", "room_middle": "in the middle of the room"}`
  - `PLACE_WORDS` (accepted variants per place): doorway ("doorway", "door opening"), corner ("corner",),
    hallway_end ("end of the hallway", "end of the corridor", "end of the hall"), wall ("against the wall",
    "beside the wall", "by the wall"), room_middle ("middle of the room", "centre of the room", "center of the room").
  - `STOP_RELATION_WORDS` gains `("object", "in front of"): ("in front of",)`, `("object", "behind"): ("behind",)`,
    `("object", "next to"): ("next to", "beside")`.
  - `_DISTANCE` regex for the Global Constraints' distance words.
  - `stop_text(stop, name, wall=False)`: `"stop <relation> the <name>"` for object stops,
    `"stop " + PLACE_TEXT[place]` for places.
  - `description_rejects`: for object / place stops, `"stop:distance"` when `_DISTANCE` matches after the last stop
    word; for place stops `"stop:place"` when no `PLACE_WORDS[place]` variant follows the last stop word (object
    stops reuse the existing `stop:name` / `stop:relation` checks).
  - `visible_stop_rejects(text, stop) -> list[str]`: the names-free part of the two checks above.
  - `d3_rejects(text, ev) -> list[str]` = `d2_rejects(text, ev) + visible_stop_rejects(text, ev.get("stop"))`.
  - `_check_ending(stop, objects)`: a stop with `stop_kind(stop)` not None returns `{"type": "stop", "stop": kind}`;
    every other case unchanged.
  - `PROMPT_VERSION = "d6"`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("stop,text", [
    ({"type": "object", "relation": "next to", "object": "o"}, "stop next to the table"),
    ({"type": "place", "place": "room_middle", "object": None}, "stop in the middle of the room"),
    ({"type": "place", "place": "hallway_end", "object": None}, "stop at the end of the hallway")])
def test_stop_text_d3(stop, text): assert stop_text(stop, "table") == text

def test_template_passes_rejects_for_every_d3_stop():   # each relation and each place: description_rejects(template) == []
def test_distance_words_rejected():   # place room_middle, "Go left of the sofa, then stop well past the sofa." -> "stop:distance" and "stop:place" in rejects
def test_place_word_required():       # place corner, "..., then stop by the window." -> "stop:place"
def test_variant_place_words_pass():  # "..., then stop at the end of the corridor." for hallway_end -> []
def test_d3_rejects_leaves_behind_and_turn_texts_alone():   # Review Focus 3: d3_rejects(behind text, behind ev) == d2_rejects(...)
def test_check_ending_carries_the_stop_kind():   # {"type":"stop","stop":"place:corner"}; a d2 "near" stop -> {"type":"stop"}
def test_compose_prompt_has_the_new_stop_line():  # "Then: stop next to the table." in compose_prompt(...)
def test_prompt_version(): assert PROMPT_VERSION == "d6"
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_stop_text_d3.py -q` → FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** `uv run pytest tests/test_stop_text_d3.py tests/test_descriptions.py tests/test_endings_text.py -q`, then `uv run pytest -q` → green.
- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py tests/test_stop_text_d3.py
git commit -m "feat: d3 stop text — object relations and places, distance words rejected, stop kind in desc_check (prompt d6)"
```

---

### Task 4: Seeding and the d3 stages (events, d3-seed, finalize, label job)

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (`rewrite_ending` ~l.122 → shared `_replace_final_clause`; new
  `rewrite_stop`, `seed_d3_labels` next to `seed_d2_labels`)
- Modify: `scripts/prepare_habitat_instr.py` (`V5_OUT`; `stage_events` `--stops`; new `stage_d3_seed`;
  `stage_finalize_descriptions` `--round d3`; argparse ~l.985–1003; module docstring run book)
- Modify: `slurm/label_descriptions.sbatch` (l.63–72: `ROUND` d1|d2|d3, d3 like d2)
- Test: `tests/test_d3_stages.py`; update `tests/test_slurm_thread_caps.py:38`

**Interfaces:**
- Consumes: Task 2 `describe_route(..., stops)`, `EVENTS_SCHEMA_D3`, `stop_kind`; Task 3 `stop_text`, `d3_rejects`.
- Produces:
  - `_replace_final_clause(text, clause) -> tuple[str | None, str | None]` (the body of today's `rewrite_ending`:
    reasons `"no_stop_clause"`, `"no_connector"`); `rewrite_ending` keeps its signature and behaviour.
  - `rewrite_stop(text, events, stop, names) -> tuple[str | None, str | None]`: final clause → `", then " +
    stop_text(...)`; reason `"rejects"` when the result fails `description_rejects`.
  - `seed_d3_labels(old_events, new_events, new_objects, old_names, old_desc) -> dict` with the keys of
    `seed_d2_labels` (`names`, `desc`, `changed`, `rewrite_failed`); tags `"reused_from": "d2"`,
    `"rewritten_from": "d2"`; a route is rewritten when its events are unchanged and its new stop is an object or
    place stop.
  - `V5_OUT = REPO_ROOT / "data" / "prepared_habitat_instr_v5"`.
  - `events --stops {d2,d3}` (default d2): passes `stops` to `describe_route` (via the pool's row function), stamps
    `EVENTS_SCHEMA_D3` for d3, refuses to overwrite a different schema without `--force`; `events_meta.json` per
    split adds `"stop_kinds"` (Counter of `stop_kind`) and `"visible_drop_share"` = no_stop_place / (described
    visible stops + no_stop_place).
  - `d3-seed` (`--d2-events-dir` default `V4_OUT`, `--events-dir` default `V5_OUT`, `--labels` default
    `D1_LABELS`, `--force`): reads `names_<s>_gemma_d2.jsonl`, `names_<s>_d2.jsonl`, `desc_<s>_mix_d2.jsonl`; writes
    `names_<s>_gemma_d3.jsonl`, `names_<s>_d3.jsonl`, `desc_<s>_mix_d3.jsonl`, `$V5/events_eval_changed.jsonl`,
    `$V5/objects_eval_changed.jsonl`, `$V5/seed_meta.json` (as `stage_d2_seed`); refuses non-empty d3 label files
    without `--force`.
  - `finalize-descriptions --round d3`: events and out dir `V5_OUT`, recheck `d3_rejects`, `meta_d3.json`.
  - `ROUND=d3` in the sbatch behaves as d2 (gate on `eval_changed`; missing-file message names `d3-seed`).

- [ ] **Step 1: Write the failing tests**

```python
def test_rewrite_stop_turns_a_distance_stop_into_a_place():
    ev = [{"type": "pass", "side": "left", "objects": ["o"]}]
    st = {"type": "place", "relation": None, "place": "room_middle", "object": None}
    assert rewrite_stop("Go left of the sofa, then stop well past the sofa.", ev, st, {"o": "sofa"}) == \
        ("Go left of the sofa, then stop in the middle of the room.", None)
def test_rewrite_stop_replaces_only_the_final_clause():   # Review Focus 4: two stop words, only the last clause changes
def test_rewrite_ending_unchanged():                      # existing T2 rewrite gives the same output as before the refactor
def test_seed_d3_reuses_rewrites_and_lists_changed():     # unchanged behind -> reused_from d2; changed visible stop -> rewritten_from d2; changed events -> changed
def test_seed_d3_counts_rewrite_failures():               # a text without a connector -> changed, rewrite_failed == {"no_connector": 1}
def test_events_stops_d3_stamps_schema_and_stop_kinds(tmp_path):   # like test_d2_stages::test_events_writes_into_out_dir_...: meta["schema"] == "d3-stops", "stop_kinds" and "visible_drop_share" per split
def test_d3_seed_stage_writes_d3_files(tmp_path)          # desc_eval_mix_d3.jsonl, names_eval_gemma_d3.jsonl, events_eval_changed.jsonl, seed_meta.json
def test_d3_seed_refuses_to_overwrite_label_work(tmp_path)
def test_finalize_round_d3_rechecks_with_d3_rejects(tmp_path)   # a reused text with a distance stop on a d3 place route is dropped (recheck_failed == 1)
```

and in `tests/test_slurm_thread_caps.py:38` expect `case "$ROUND" in d1|d2|d3) ;; *) echo "ROUND must be d1, d2 or d3`.

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_d3_stages.py tests/test_slurm_thread_caps.py -q` → FAIL.
- [ ] **Step 3: Implement** the interfaces above (DRY: `stage_d3_seed` and `stage_d2_seed` share one helper
  parameterised by old/new round, old events dir and seed function).
- [ ] **Step 4: Run** `uv run pytest tests/test_d3_stages.py tests/test_d2_stages.py tests/test_slurm_thread_caps.py -q`, then `uv run pytest -q` → green.
- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py scripts/prepare_habitat_instr.py slurm/label_descriptions.sbatch tests/test_d3_stages.py tests/test_slurm_thread_caps.py
git commit -m "feat: d3 stages — events --stops d3 into v5, d3-seed (rewrite d2 stop clauses), finalize --round d3, ROUND=d3"
```

---

### Task 5: Per-stop-type eval (`by_stop`)

**Files:**
- Modify: `src/rover_vlm/eval.py` (`aggregate_description_metrics` ~l.842–915)
- Test: `tests/test_description_eval.py` (append)

**Interfaces:**
- Consumes: desc_check `ending["stop"]` from Task 3.
- Produces: `aggregate_description_metrics(records)["by_stop"]: dict[str, dict]`, keyed by stop kind, each with
  the `_ending_group` fields (n, follow_rate, gt_follow_rate, endpoint_error_median, point_error_median,
  ending_rate, gt_ending_rate, constant_goal_endpoint_error_median); `{}` when no record carries a stop kind.

- [ ] **Step 1: Write the failing tests**

```python
def test_by_stop_groups_visible_stops_by_kind():   # 3 records "place:corner", 2 "object:next to" -> by_stop[...]["n"] == 3 / 2, stop rate per group
def test_by_stop_is_empty_for_d2_records():        # records with ending {"type": "stop"} only -> by_stop == {}
```

- [ ] **Step 2: Run** → FAIL (`KeyError: 'by_stop'`). **Step 3: Implement.** **Step 4: Run** the file, then `uv run pytest -q` → green.
- [ ] **Step 5: Commit** — `git add src/rover_vlm/eval.py tests/test_description_eval.py && git commit -m "feat: description metrics by stop kind (d3)"`

---

### Task 6: Steering with d3 stops

**Files:**
- Modify: `src/rover_vlm/steering.py` (`stop_key` ~l.87, `stop_pairs` ~l.171, `frame_pairs` ~l.246)
- Modify: `scripts/prepare_habitat_instr.py` (`_steer_init` / `steer_frame` / `stage_steer`: `--stops`)
- Test: `tests/test_steering.py` (append)

**Interfaces:**
- Consumes: Task 2 `describe_xz(..., stops)`.
- Produces: `stop_key(stop)` returns `("stop", relation, amount, object, place)` for visible stops (behind / turn
  keys unchanged); `stop_pairs(frame, meta, route_xz, turn=None, stops="d2")`; `frame_pairs(..., stops="d2")`;
  `steer --stops {d2,d3}` (default d2) threaded to every frame.

- [ ] **Step 1: Write the failing tests**

```python
def test_stop_key_tells_places_apart():   # Review Focus 5
    a = {"type": "place", "relation": None, "place": "corner", "object": None}
    assert stop_key(a) != stop_key({**a, "place": "room_middle"})
def test_stop_pairs_use_d3_stops(tmp_path):   # a frame whose two cuts end at a corner and in open space -> one pair, both routes' desc["stop"]["type"] == "place"
def test_steer_stage_passes_stops(tmp_path, monkeypatch)   # stage_steer(--stops d3) calls frame_pairs with stops="d3"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `uv run pytest tests/test_steering.py -q`, then `uv run pytest -q` → green.
- [ ] **Step 5: Commit** — `git add src/rover_vlm/steering.py scripts/prepare_habitat_instr.py tests/test_steering.py && git commit -m "feat: steering stop pairs with d3 stops (--stops d3)"`

---

### Task 7: Stop audit page (`--mode stops`, G1)

**Files:**
- Modify: `scripts/audit_instruction_labels.py` (next to `sample_endings` ~l.189; `VERDICTS`; `main()` mode branch;
  `--mode` choices l.668)
- Modify: `scripts/audit_instruction_labels.html` (a `DATA.mode === "stops"` branch, title "Stop Phrases Audit")
- Test: `tests/test_d3_stages.py` (append)

**Interfaces:**
- Consumes: Task 2 `stop_kind`, `REASONS` ("no_stop_place"); existing `draw_route`, `draw_numbered`,
  `template_text`, `placeholder_names`, `_read_marks`.
- Produces:
  - `sample_stops(rows, seed, n=100, min_per_kind=10, n_drop=10) -> tuple[list[tuple[str, dict]], dict[str, int]]`:
    every stop kind with ≥ `min_per_kind` rows gets ≥ `min_per_kind`; `n_drop` "no_stop_place" drops; the rest
    random; total ≤ `n` + `n_drop`; shuffled; deterministic per seed.
  - `stops_item(rec, row, objects, kind) -> dict` (shape of `endings_item`): route drawn, the stop object outlined
    green for object stops, the goal ringed for place stops, the place's measures in the item caption, the
    template text; a drop asks "DROPPED: no place fits the end of this route. Is dropping it right?".
  - `VERDICTS["stops"]`: ok "True" (the stop says where the route ends; for a drop, dropping is right) /
    relation "Wrong relation" / place "Wrong place" / object "Wrong object" / keep "Should not be dropped" /
    unsure "Can't tell".
  - `--mode stops --events $V5/events_train.jsonl --objects $V5/objects_train.jsonl [--seed] [--out]` →
    `outputs/stops_audit.html`; `--mode stops --score-marks DIR` (same args) re-draws the sample and prints, per kind
    and overall, ok / n and the gate (ok / (n − unsure) ≥ 0.90).
  - G3 for d3: `select_d3_rows(ok, events_by_id, n, seed, rewritten_share=0.4) -> list[dict]`: rewritten rows are
    those with `rewritten_from == "d2"` (reused d2 rows can carry an older `rewritten_from: "d1"` and are left out,
    as are all `reused_from` rows); the rest are rows composed with `PROMPT_VERSION` ("d6"), split evenly between
    object and place stops (by `stop_kind` of the events row) where both pools allow. `--mode d1 --round d3` uses it.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_d3_rows_takes_d2_rewrites_and_d6_composed_rows():   # a reused row tagged rewritten_from "d1" is never picked; composed rows are ~half object, ~half place
def test_sample_stops_covers_every_kind():   # 8 kinds (3 relations + 5 places) x 15 rows + 20 drops -> each kind >= 10, drops == 10, same picks for the same seed
def test_score_stops_reports_each_kind_and_the_gate(tmp_path)   # marks for 4 items (3 ok, 1 relation) -> per-kind lines + "GATE" line
def test_audit_template_has_a_stops_mode():
    html = audit.TEMPLATE.read_text()
    assert 'DATA.mode === "stops"' in html and "Stop Phrases Audit" in html
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `uv run pytest tests/test_d3_stages.py -q`, then `uv run pytest -q` → green.
- [ ] **Step 5: Commit** — `git add scripts/audit_instruction_labels.py scripts/audit_instruction_labels.html tests/test_d3_stages.py && git commit -m "feat: stop phrases audit page and per-kind scoring (d3 G1)"`

---

### Task 8: Run book

**Files:**
- Modify: `README.md` (new section "Stop phrases and d3", after "Turn events and d2"), index-only (Global Constraints)
- Modify: `scripts/prepare_habitat_instr.py` module docstring (d3 commands)

- [ ] **Step 1:** Write the section: what d3 changes (spec link), the commands of the run order below with their
  outputs and gates, and `$V5` = `data/prepared_habitat_instr_v5`.
- [ ] **Step 2:** `uv run pytest -q` → green (README is read by no test; the docstring change must not break imports).
- [ ] **Step 3: Commit (index-only README)**

```bash
git diff README.md > /dev/null   # README already holds the user's edits: do NOT git add it
git show HEAD:README.md > "$SCRATCH/README.head" && cp "$SCRATCH/README.head" "$SCRATCH/README.new"
# add the section to $SCRATCH/README.new, then:
diff -u --label a/README.md --label b/README.md "$SCRATCH/README.head" "$SCRATCH/README.new" > "$SCRATCH/readme.patch"
git apply --cached "$SCRATCH/readme.patch" && git apply "$SCRATCH/readme.patch"
git add scripts/prepare_habitat_instr.py
git commit -m "docs: README run book for stop phrases (d3)"
```

---

### Task 9: CPU run — events, dry-run gate, G1 (stop for the user)

No code unless calibration needs it.

- [ ] **Step 1:** `uv run scripts/prepare_habitat_instr.py events --stops d3 --labelset-dir data/prepared_habitat_instr_v2 --out-dir data/prepared_habitat_instr_v5`
  (CPU, ~45 min; d2's took 38 min). Report `events_meta.json`: stop_kinds per split, `visible_drop_share`,
  described counts vs v4.
- [ ] **Step 2 (gate):** if `visible_drop_share` > 0.15 in either split, or any place kind is < 2% or > 60% of the
  visible stops, **stop and report to the user** with the per-rule measures before changing thresholds (a
  threshold change is a Task 1 constant edit with its test updated, then Step 1 again).
- [ ] **Step 3:** `uv run scripts/audit_instruction_labels.py --mode stops --events data/prepared_habitat_instr_v5/events_train.jsonl --objects data/prepared_habitat_instr_v5/objects_train.jsonl --out outputs/stops_audit.html`;
  publish it as an artifact with the `db` capability (marks), record the URL in memory
  (`visualization-artifacts.md`), and **wait for the user's marks**. Score with `--score-marks`; gate ≥ 0.90
  overall, each kind reported; "next to" failing goes to the user (drop it again or not).

### Task 10: Labels, training, steering, eval (every sbatch asked first)

- [ ] **Step 1:** `uv run scripts/prepare_habitat_instr.py d3-seed` → report `seed_meta.json` (reused / rewritten / to_compose, est. hours).
- [ ] **Step 2 (G2, GPU):** `ROUND=d3 DATA=data/prepared_habitat_instr_v5 NAMER=mix sbatch slurm/label_descriptions.sbatch` → accept ≥ 0.90, template ≤ 0.10.
- [ ] **Step 3 (G3, human):** `uv run scripts/audit_instruction_labels.py --mode d1 --round d3 --events data/prepared_habitat_instr_v5/events_eval.jsonl --objects data/prepared_habitat_instr_v5/objects_eval.jsonl --desc outputs/instr_labels_v3/desc_gate_mix_d3.jsonl outputs/instr_labels_v3/desc_eval_mix_d3.jsonl --out outputs/instr_d3_audit.html`; publish with marks; wait; ≥ 90%.
- [ ] **Step 4 (GPU):** `ROUND=d3 DATA=data/prepared_habitat_instr_v5 NAMER=mix STAGE=full sbatch --time=12:00:00 slurm/label_descriptions.sbatch`; then `uv run scripts/prepare_habitat_instr.py finalize-descriptions --round d3 --desc-suffix _mix` (recheck_failed ~ 0).
- [ ] **Step 5 (GPU):** `sbatch slurm/train_habitat.sbatch data/prepared_habitat_instr_v5/train_desc.json outputs/runs_desc/d3 --epochs 2 --save-steps 500 --resume`.
- [ ] **Step 6:** steering on v5 with the d3 frames: `steer-objects --frames $V5/eval_desc_entry.json --out-dir $V5/steer`, `steer --stops d3 --frames $V5/eval_desc_entry.json --names .../names_eval_gemma_d3.jsonl --fallback-names .../names_eval_d3.jsonl --out-dir $V5/steer` (+ `STAGE=steer_names` job if objects are unnamed, asked), `steer-select --out-dir $V5/steer` (Gate 0), `--mode steer` audit (Gate 1, marks), `ROUND=d3 STEER_DIR=$V5/steer STAGE=steer` job (asked), `finalize-steer --frames $V5/eval_desc_entry.json --steer-dir $V5/steer --desc .../desc_steer_eval_d3.jsonl --out-dir $V5` (Gate 2).
- [ ] **Step 7 (GPU):** evals into `outputs/eval_desc_d3`, desc and steering sets, for d3 **and** d2:
  `DATA=$V5 OUT_DIR=outputs/eval_desc_d3 ADAPTER=outputs/runs_desc/<tag>/adapter TAG=<tag> [SETS=steer] sbatch slurm/eval_descriptions.sbatch`.
- [ ] **Step 8:** path videos (`render_desc_videos.py --eval-dir outputs/eval_desc_d3 --data-dir $V5`), steering report,
  and the d3-vs-d2 report built from `outputs/eval_desc_d2/d2_report/src/` (add a per-stop-kind section from `by_stop`).
