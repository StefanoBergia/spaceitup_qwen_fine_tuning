# Mixed namer: Gemma names, Cosmos fallback on same-name routes

**Spec:** `docs/superpowers/specs/2026-09-30-route-descriptions-design.md` (Addenda A–A4 binding); this
plan adds "Addendum A5 (mixed namer)". Decided by the user 2026-10-02.

**Why.** Same-route re-audit (99 routes): Gemma names 84 ok vs Cosmos names 83 ok; the errors are
complementary (only 2 routes wrong under both). Gemma's characteristic new error: two different
objects of one route get the same name ("go between the wooden bench and the wooden bench",
"keep the wooden dining table on your left, then ... on your right") — 30/296 passed Gemma
routes vs 6/282 Cosmos. Offline estimate on the audited routes: Gemma names, Cosmos names on
same-name routes (when distinct), else drop -> 86/97 ok (88.7%), 8 fallbacks, 2 drops. User
chose this setup; Cosmos3-Nano keeps composing.

## Global Constraints

- Same-name rule: a route has a name clash when two DIFFERENT object ids among its event members
  + stop object have the same `head_noun` (rover_vlm.instructions.head_noun) under the names used.
  Function `name_clash(row, names) -> bool` in descriptions.py (unnamed objects are not a clash;
  that stays `unnamed_object`).
- Compose (`label_route`) takes optional `fallback_names`. Order: if all route objects are named
  in `names` and there is no clash -> use `names`, row field `"namer": "primary"`. Else if
  `fallback_names` is given, all route objects are named there and there is no clash -> use
  `fallback_names`, `"namer": "fallback"`. Else: if some object is unnamed in both ->
  `rejects: ["unnamed_object"]` as today; if names exist but clash -> `rejects: ["name:clash"]`,
  not passed. With no `fallback_names` the clash rule still applies (a Cosmos-only run now
  rejects its clashing routes as `name:clash`).
- `scripts/label_descriptions.py --stage compose`: new `--fallback-names FILE` (name rows, same
  format as --names). The compose "retryable gap" check (`names_missing`) must consider an object
  named in either file as present.
- `slurm/label_descriptions.sbatch`: new `NAMER=mix`: Gemma name passes -> `names_<tag>_gemma_d1.jsonl`;
  stop Gemma; serve Cosmos; Cosmos name passes -> `names_<tag>_d1.jsonl`; Cosmos compose passes with
  `--names names_<tag>_gemma_d1.jsonl --fallback-names names_<tag>_d1.jsonl` -> `desc_<tag>_mix_d1.jsonl`.
  `NAMER=cosmos` and `NAMER=gemma` unchanged except that the clash rule now applies to their compose.
  Header usage line: `NAMER=mix STAGE=full sbatch slurm/label_descriptions.sbatch   # re-submit the
  same line after a TIMEOUT; never OVERWRITE=1 on a full run`.
- `scripts/prepare_habitat_instr.py finalize-descriptions`: new `--desc-suffix` (default "" =
  today's `desc_{split}_d1.jsonl`); `--desc-suffix _mix` reads `desc_{split}_mix_d1.jsonl`.
- `summarize_d1` (compose) additionally reports `namer` counts among passed rows.
- No model loading on the login node. Tests: `tests/test_descriptions.py`, the existing sbatch
  test file `tests/test_cosmos3_serve_lib.py` (fake uv/vllm/curl pattern), the finalize tests.
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest -q -x` (995 passed before).
- README.md has a user-owned uncommitted row: index-only commits; never stage scripts/evaluate.py.

## Task 1: name_clash + compose fallback

Files: `src/rover_vlm/descriptions.py`, `scripts/label_descriptions.py`, `tests/test_descriptions.py`.
Tests first: clash detection (same head noun on two ids; same id twice is not a clash; unnamed
ignored); label_route primary / fallback / name:clash / unnamed_object paths with a fake ask_text;
`names_missing` with fallback; summarize_d1 namer counts.

## Task 2: NAMER=mix, finalize --desc-suffix, docs

Files: `slurm/label_descriptions.sbatch`, `scripts/prepare_habitat_instr.py`, tests, README d1
section (index-only), spec Addendum A5 (evidence above, rule, NAMER=mix, full-run command).
