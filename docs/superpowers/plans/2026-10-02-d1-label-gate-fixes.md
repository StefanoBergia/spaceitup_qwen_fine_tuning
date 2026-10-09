# d1 label gate fixes (gate job 92114 failed)

**Spec:** `docs/superpowers/specs/2026-09-30-route-descriptions-design.md` (Addendum A + A2 binding);
this plan adds "Addendum A3 (label gate r1)". Decided by the user 2026-10-02: N1 (walls only) + N2 + C1 + C2.

**Why.** Gate job 92114 (300 eval routes, events schema d1-addendumA2): accept 0.890 < 0.9 (33
`unnamed_object`), template share 0.262 > 0.1.
- Names: 27/377 objects rejected. 16 are doors/windows on object/wall outlines that Cosmos names
  well ("open doorway", "glass door", "window with blind") but types `kind: opening` -> our
  `kind:mismatch`; the wall naming prompt itself lists "a door in a wall". 3 are "<x> unit"
  names flagged generic ("white shelving unit", "shelf unit"). 4 are thinking runs cut at
  max_tokens 2048 (parse). Offline rescoring of the stored raws: accepting `opening` for need
  "wall" + the unit fix recovers 14/33 routes (accept ~0.937).
- Compose: 70 routes fell back to the template; 96 of their 140 Cosmos replies never reached
  `</answer>` (cut at 2048 tokens) while Cosmos argued with a contradiction in the prompt:
  "keep each object's name and the word left or right that goes with it" vs "for other objects
  never say 'your left/right'", worst on `between` steps (24/55 fell back), which have no side.

## Global Constraints

- Name rule N1: in `descriptions.name_rejects`, a reply of kind "opening" for need "wall" is NOT a
  `kind:mismatch` (doors / windows in a wall). Need "object" + kind "opening" stays a mismatch;
  need "opening" + kind != "opening" stays a mismatch.
- Name rule N2: a name whose head noun is "unit" is not generic when the word before "unit" is in
  `UNIT_MODIFIERS = {"shelving", "shelf", "storage", "kitchen", "wall", "tv", "display", "corner",
  "drawer", "cabinet"}` ("white shelving unit" passes; "unit", "white unit", "large unit" stay generic).
- C1: `scripts/label_descriptions.py --max-tokens` default 4096 (was 2048); both stages.
- C2: compose prompt d2 (`PROMPT_VERSION = "d2"`). Replace the sentence "Keep every step in this
  order, keep each object's name and the word left or right that goes with it, and keep where to
  stop." and the line "For walls, keep the words 'on your left/right' exactly as given; for other
  objects never say 'your left/right'." with this exact text:

      Keep every step in this order and keep each object's name. Copy each step's direction
      words exactly as written: 'on your left' / 'on your right' after a wall stays that way;
      'left of' / 'right of' before an object stays that way; a step with 'between' or 'through'
      has no left or right, so do not add one. Keep where to stop.

  Everything else in `compose_prompt` (steps list, Then: line, style, shape, small-words
  sentence, numbers as words, 45 words, answer block) is unchanged. The naming prompt is
  unchanged (its rows also carry prompt_version "d2" since the constant is shared).
- Text rules (`description_rejects`) and the template are unchanged.
- Pure Python, CPU. Tests in `tests/test_descriptions.py`; run with
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_descriptions.py -q`;
  full suite `uv run pytest -q -x` (969 passed before this plan). Never load a model.
- `README.md` has an uncommitted user row: commit README changes index-only (patch of your hunk
  + `git apply --cached`), never `git add README.md`; never stage `scripts/evaluate.py`.

## Task 1: name rules N1 + N2

Files: `src/rover_vlm/descriptions.py`, `tests/test_descriptions.py`.
- Tests first: need "wall" + kind "opening" ("open doorway") -> []; need "object" + kind
  "opening" -> ["kind:mismatch"]; need "opening" + kind "object" -> ["kind:mismatch"];
  "white shelving unit" / "shelf unit" -> no name:generic; "unit", "white unit" -> name:generic.
- Implement in `name_rejects` with `UNIT_MODIFIERS` as a module constant next to GENERIC_NAMES.
- Commit.

## Task 2: compose prompt d2 + max tokens 4096 + docs

Files: `src/rover_vlm/descriptions.py`, `scripts/label_descriptions.py`, `tests/test_descriptions.py`,
spec (Addendum A3), `README.md` d1 section (index-only), module docstrings that say "prompt d1"
where they describe the compose prompt version (file names `*_d1.jsonl` and `--mode d1` stay).
- Tests first: `compose_prompt` output contains the exact new paragraph; no longer contains
  "the word left or right that goes with it" nor "never say 'your left/right'"; PROMPT_VERSION ==
  "d2"; the --max-tokens default is 4096 (test the argparse default or a module constant).
- Spec Addendum A3: the gate r1 numbers above, the four decisions, that N1+ (objects) was declined.
- Commit.
