# Route events: drop obstacles the route runs through (Gate 1 r2 fix A1)

**Spec:** `docs/superpowers/specs/2026-09-30-route-descriptions-design.md` (Addendum A binding);
this plan adds Addendum A2 (one rule), decided by the user 2026-10-02.

**Why.** Gate 1 r2 (89/100): 4 of the 6 "wrong relative position" routes and 1 "wrong object"
route have an event whose blob the route runs *through* — the closest distance between the
route and the blob's occupied cells is ~0.01 m, while every other event keeps the planner's
~0.45-0.5 m clearance. `candidates()` takes the side at that single closest-approach point,
which then lies on the obstacle itself, so left/right is a coin flip (00660: the rack is on the
rover's right at 36 route points, on its left at 6; the closest point was one of the 6).
Population (800 random train routes): 8.1% of routes, 64/67 of these events are walls; the
closest distance is bimodal — 65 events < 0.05 m, none in 0.05-0.2 m, the rest >= 0.2 m.
User chose A1: such a blob is not an event candidate at all. (Fix B, dropping depth-fallback
outlines, was declined.)

## Global Constraints

- `TOUCH_M = 0.2` (metres): a blob whose closest distance to the usable route
  (arc > START_SKIP_M) is below `TOUCH_M` is not a candidate.
- Exception: a blob the route threads through a gap of (`_threaded_gap(...)` is not None and
  extent >= WALL_EXTENT_M, i.e. what build_events turns into a `through` event) stays a
  candidate — doorway behaviour must not change.
- `EVENTS_SCHEMA` becomes `"d1-addendumA2"` so the labeller and audits refuse events files
  built before this rule.
- Nothing else in route_events changes (stop rules, outlines, merging, scores).
- Pure numpy, CPU; tests are synthetic grids in `tests/test_route_events.py` (see its
  `write_sample`, `_free`, the `_wall_left_state` / doorway helpers). Run tests with
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_route_events.py -q`
  and the full suite with `uv run pytest -q -x` (957 passed before this plan).
- `README.md` has an uncommitted row that belongs to the user: commit README changes
  index-only (build a patch of only your hunk and `git apply --cached` it); never `git add README.md`.
  `scripts/evaluate.py` is also the user's — never stage it.

## Task 1: TOUCH_M rule + schema bump + docs

Files: `src/rover_vlm/route_events.py`, `tests/test_route_events.py`,
`docs/superpowers/specs/2026-09-30-route-descriptions-design.md`, `README.md` (index-only).

1. Tests first (must fail before the change):
   - A long wall the route crosses / runs over (route passes within 0.05 m of the wall's
     cells — e.g. a wall segment whose end the route clips, with most of the wall on one side
     and two cells on the other) yields no candidate for that blob, and `build_events` returns
     no wall event for it.
   - A small object the route runs over (route passes through its cells) is not a candidate.
   - An obstacle at 0.3 m from the route is still a candidate (rule does not over-reach).
   - The existing doorway tests (`test_doorway_in_one_wall_is_a_through_event`,
     `test_two_separate_wall_halves_with_an_opening_are_through`, ...) still pass unchanged;
     add one asserting a doorway whose jambs are < TOUCH_M from the route (a gap of exactly
     MIN_GAP_M-ish width threaded off-centre) is still a `through` event.
   - Constant test: `TOUCH_M == 0.2`; schema test updated to `"d1-addendumA2"`.
2. Implement in `candidates()`: after `j, k = ...argmin`, skip when `dist[j, k] < TOUCH_M` and
   the blob is not a threaded gap (compute `_threaded_gap` once and reuse it for the `gap` field).
   Constant `TOUCH_M` next to `PASS_M` with a comment citing Gate 1 r2 (route runs through the
   blob, closest-point side undefined; 8% of routes).
3. Module docstring: one sentence on the rule.
4. Spec: append "Addendum A2 (2026-10-02, Gate 1 r2)" after the Addendum A clarifications:
   the rule, the evidence (5/11 flagged, 8% of routes, bimodal distance), the doorway exception,
   the schema bump, and that depth-fallback outlines stay (user declined fix B).
5. README "Route descriptions (d1)" section: one line for the rule + schema `d1-addendumA2`
   (index-only commit).
6. Commit(s) with the attribution trailer from the session.

Done when: new tests fail on the old code and pass on the new; full suite green.

## Task 2: zero-event routes must be straight

**Why.** A route with no events is described "Go straight ahead, and stop ..." (descriptions.py
empty-events fallback). Of the 114 zero-event train routes (Addendum A run), the max offset of
the route from its entry->goal line has median 0.71 m (77 > 0.5 m, 36 > 1 m): the sentence is
false for most of them, and Task 1 creates more (00423, 00599 re-described as "go straight
ahead" while bending round a door / left-then-right). User decided 2026-10-02: keep only the
straight ones.

Files: `src/rover_vlm/route_events.py`, `tests/test_route_events.py`, spec (extend Addendum A2),
`README.md` (index-only), and wherever the events stage's reason list is enumerated if it is
not derived from `REASONS` (check `scripts/prepare_habitat_instr.py`).

- `STRAIGHT_MAX_M = 0.3`: in `describe_route`, when `build_events` returns no events and the
  route's max offset from its entry->goal line (`route_geometry(route)["offset"].max()`)
  exceeds `STRAIGHT_MAX_M`, return `_none("no_events_bent")` (checked before the stop phrase is
  computed is fine; the result must not depend on the order). Add `"no_events_bent"` to `REASONS`.
- A zero-event route with offset <= 0.3 m is unchanged (still described with its stop).
- Routes with >= 1 event are unaffected regardless of offset.
- Tests first: bent zero-event route -> reason `no_events_bent`; straight zero-event route with a
  stop object ahead -> described, events == []; bent route WITH an event -> described.
  Constant test `STRAIGHT_MAX_M == 0.3`.
- Schema stays `"d1-addendumA2"` (same rerun as Task 1); spec Addendum A2 gets a paragraph with
  the evidence above; README d1 section gets one line (index-only commit).
