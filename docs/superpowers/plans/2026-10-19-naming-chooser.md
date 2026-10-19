# Naming Chooser (Addendum C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A model picks, per object, the best of the four candidate names (today's, Gemma, Qwen3.6, Cosmos-Reason2) or says none fits (→ today's name for that object); the best of three choosers is chosen automatically against the existing marks, checked by the user on 50 fresh objects, then renames every d2 object.

**Architecture:** The choose logic (candidate list, prompt c1, reply parsing, one row per object with the per-object V0 fallback) lives in `rover_vlm.descriptions` next to the naming code; `label_descriptions.py --stage choose` runs it against a vLLM endpoint like the name stage. `naming_variants.sbatch` gains a choose mode. CPU stages write today's names for every object and the fresh 50-object sample; the audit script scores chooser files with the existing name scoring. Chooser outputs are written as `names_<set>_C<k>_n2.jsonl`, so the existing `rename-texts --variant C<k>` consumes them unchanged.

**Tech Stack:** Python 3.12, Pillow, pytest; bash + SLURM; vLLM 0.21 (`.venv-cosmos3`); `uv run` for everything.

**Spec:** `docs/superpowers/specs/2026-10-15-object-naming-design.md`, Addendum C (with the sections it builds on).

## Global Constraints

- Branch `feat/turn-events`. Never `git add -A`, never `git stash`; stage only the files a task names. `README.md` (the user's uncommitted "Prompt framings" row) and `scripts/evaluate.py` are the user's: README edits are committed index-only (Task 6).
- `uv run` only; `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` before tests. Never load a model on the login node; ask before every `sbatch`; at most 3 SLURM jobs.
- d1 files are read-only (`*_d1.jsonl`, `data/prepared_habitat_instr_v2/`, `data/prepared_habitat_instr_v3/`). Temp files in the session scratchpad.
- Choosers: **C1 = `Qwen/Qwen3.6-35B-A3B-FP8`, C2 = `nvidia/Cosmos-Reason2-32B`, C3 = `google/gemma-3-12b-it`**; prompt **c1**; `--max-tokens 8192`; namer re-runs V1–V3 with n2, `--max-tokens 8192`, default concurrency **32**.
- Candidates: V0 (today's name, gate names read last), V1, V2, V3 — distinct case-insensitive, only names that passed, shuffled with a seed stable per object, lettered a–d, no model labels, plus "none".
- **"none" or a failed parse → that object's final name is V0, written into the chooser row (`passed: true`, `name_source: ["V0_fallback"]`)**; `passed: false` only when there is neither a pick nor a V0 name.
- One distinct candidate → no call, it is the pick (`single: true`); no candidate → no call, final name V0.
- Gate number: **final-name accuracy** on the 98 marked random objects (the final name is among the ticked names; "none is right" / Done-with-nothing is wrong for every chooser); ceiling 81/90 non-opening. At most one prompt revision (c2) after the scores.
- Fresh check: **50 train objects not in the 119, plain random (population-weighted), seed 1**; one name per object (the final name); report accuracy with a 95% interval next to the comparison estimate; no hard gate.
- Artifact-db doc ids allow only letters, digits and `_ - . ~ : @ +`.

## Review Focus

1. **A reply that is not one valid letter** (a letter past the list, two letters, the name itself, an empty answer). Expected: a parse failure, one retry, then the V0 fallback, counted in the summary. Test in Task 1.
2. **An object missing from some candidate files** (V2 left it blank, a steering object absent from the train files, no V0 name). Expected: candidates from what exists; with no pick and no V0 the row has `passed: false`. Tests in Tasks 1 and 2.
3. **A chooser run that dies midway.** Expected: request failures are error rows, `--resume` retries them, and `rename-texts` refuses the file until it is complete (already built). Test in Task 2.
4. **`--candidates` without V0.** Expected: the stage refuses before any call (the per-object fallback is impossible). Test in Task 2.
5. **The check folder also holds `names_V0.jsonl`** (written by `naming-sample`). Expected: the check page and its score use only the chooser's file. Test in Task 5.

---

## File map

| File | Change |
|---|---|
| `src/rover_vlm/descriptions.py` | `candidate_names`, `choose_prompt`, `parse_choice_reply`, `choose_object`; `summarize_d1(..., "choose")` |
| `scripts/label_descriptions.py` | `--stage choose`, `--candidates`, `--choose-prompt` |
| `scripts/prepare_habitat_instr.py` | `today_names`, stage `today-names`, `naming-sample --proportional --exclude` |
| `slurm/naming_variants.sbatch` | `MAX_TOKENS`, `CONCURRENCY` 32, `MODE=choose` with `CHOOSERS`, sets `check` |
| `scripts/audit_instruction_labels.py` + `.html` | `--names-glob`, chooser stats in the score, sanitized doc ids |
| `tests/test_naming_choose.py` (new) | Tasks 1–5 |
| `README.md` | run book (index-only) |

---

### Task 1: The choose core

**Files:**
- Modify: `src/rover_vlm/descriptions.py` (after `label_object`; `summarize_d1`)
- Test: `tests/test_naming_choose.py` (create)

**Interfaces:**
- Produces:
  - `candidate_names(oid: str, variant_names: dict[str, dict[str, str | None]]) -> list[dict]` — each `{"name": str, "from": [variant keys]}`; distinct by `" ".join(name.lower().split())`, variants visited in sorted order (first spelling kept), list shuffled with `random.Random("choose:" + oid)`.
  - `choose_prompt(need: str, names: list[str], version: str = "c1") -> str` — `ValueError` on another version.
  - `parse_choice_reply(raw: str, n: int) -> str | None` — `"a"`…(n-th letter), `"none"`, or `None`.
  - `choose_object(ask, obj: dict, candidates: list[dict], fallback: str | None, seed: int, image: bytes | None = None, choose_prompt: str = "c1") -> dict` — `ask(image, prompt, seed) -> (raw, finish)` as in `label_object`. Row: `{"id", "need", "name", "passed", "choice", "name_source": [..], "candidates": [names in shown order], "single", "choose_prompt", "rejects": [] | ["parse"], "attempts", "finish"}` (+ `"raw"` on a parse failure).
  - `summarize_d1(rows, "choose")` adds `"choices": {"picked", "none", "failed", "single", "no_candidates"}` and `"from": {variant: count}` (each entry of `name_source`).

- [ ] **Step 1: Write the failing tests** in `tests/test_naming_choose.py` (module docstring: "Naming chooser (spec 2026-10-15 Addendum C)."):
  - `test_candidate_names_distinct_and_stable`: `variant_names = {"V0": {"o": "Grey Sofa"}, "V1": {"o": "grey sofa"}, "V2": {"o": "white wall"}, "V3": {"o": None}}` → two candidates; the sofa one has `from == ["V0", "V1"]` and name `"Grey Sofa"`; the same call twice gives the same order; an object in no file → `[]`.
  - `test_choose_prompt_c1`: for `choose_prompt("wall", ["white wall", "kitchen island"])` (whitespace-normalized) — contains `"a. white wall"`, `"b. kitchen island"`, `"Pick the one name that best names the single thing that fills most of the outline"`, `"Look at what is inside the outline, not at what is next to it or behind it."`, `"If no name fits, answer \"none\"."`, the n2 wall hint `"something long that the rover drives alongside or stops in front of"`, and `'{"choice": "<letter>|none"}'`; `choose_prompt("wall", ["x"], "c9")` raises `ValueError`.
  - `test_parse_choice_reply` (Review Focus 1): with `n=3` — `<answer>{"choice": "b"}</answer>` → `"b"`; `"B"` → `"b"`; `"c. white wall"` → `"c"`; `"none"` and `"None of these"` → `"none"`; `"d"` → `None` (past the list); `"b, c"` → `None`; `"white wall"` → `None`; `""` and an unparseable reply → `None`.
  - `test_choose_object_pick_none_and_failure`: with a fake `ask` returning `"b"` → `name` is candidate b's name, `name_source` its `from`, `passed`, `attempts == 1`; returning `"none"` → `name == fallback`, `name_source == ["V0_fallback"]`, `passed`; returning garbage twice → `choice is None`, `rejects == ["parse"]`, `attempts == 2`, `name == fallback`; garbage with `fallback=None` → `passed is False` (Review Focus 2). Pass `image=b"x"` so no image file is read.
  - `test_choose_object_single_and_no_candidates_make_no_call`: one candidate → `ask` never called, `single is True`, `choice == "a"`, `attempts == 0`; no candidates → never called, `name == fallback`.
  - `test_summarize_choose`: rows of each kind → `choices == {"picked": 1, "none": 1, "failed": 1, "single": 1, "no_candidates": 1}` and `from` counts.

- [ ] **Step 2: Run** `uv run pytest tests/test_naming_choose.py -q` — Expected: FAIL (`ImportError: cannot import name 'candidate_names'`).

- [ ] **Step 3: Implement** the four functions and the `summarize_d1` branch. c1 is this text (exact copy; `{what}` is `_N2_WHAT.get(need, "something the rover drives past.")`, `{lines}` one `"<letter>. <name>"` per candidate):

```
{SCENE_INTRO}

The picture has two panels. Left: the camera view with one region outlined and everything
else dimmed. Right: a close-up of that region. The outlined region is {what}

Here are names other people gave the outlined thing:
{lines}

Pick the one name that best names the single thing that fills most of the outline. Look at what
is inside the outline, not at what is next to it or behind it. If the outline is mostly part of
the building, the right name says which part (a wall, a column, a door, a partition, a railing);
a furniture or appliance name is right only if the outline is clearly on it. If no name fits,
answer "none".

{_answer_block('{"choice": "<letter>|none"}')}
```

`parse_choice_reply`: read `answer_payload(parse_trace(raw)[1])["choice"]` (a bare `<answer>` string also counts); lower-case and strip; `none…` → `"none"`; a single letter within the first `n`, alone or followed by a non-letter, → that letter; anything else → `None`. `choose_object` asks up to twice (`seed`, `seed + 1`) with the image from `naming_image(obj["image"], obj["polygon"], obj["crop_box"] or obj["bbox"])` unless `image` is given.

- [ ] **Step 4: Run** `uv run pytest tests/test_naming_choose.py tests/test_descriptions.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rover_vlm/descriptions.py tests/test_naming_choose.py
git commit -m "feat: naming chooser core — candidates, prompt c1, reply parsing, per-object V0 fallback (Addendum C)"
```

---

### Task 2: `label_descriptions.py --stage choose`

**Files:**
- Modify: `scripts/label_descriptions.py`
- Test: `tests/test_naming_choose.py`

**Interfaces:**
- Consumes: Task 1's `candidate_names`, `choose_object`, `summarize_d1(..., "choose")`.
- Produces: CLI `--stage choose --objects F --candidates V0=PATH V1=PATH … --out F [--choose-prompt c1] [--resume]`; one row per row of `--objects`, each with `"chooser": <served model id>` added.

- [ ] **Step 1: Write the failing tests** — load the script with `importlib` (as `tests/test_naming_fix.py` loads its scripts); monkeypatch the module's `resolve_endpoint` (→ `"http://x/v1"`), `connect` (→ `(None, "fake-model")`) and `chat_with_image`; write a 16×16 PNG with Pillow for the objects' `image`; set `sys.argv` and call `main()`:
  - `test_choose_stage_writes_one_row_per_object`: three objects (two candidates, one candidate, no candidate), `chat_with_image` answering `{"choice": "a"}` → three rows; only the first called the model; every row has `chooser == "fake-model"`; the last row's `name` is its V0 name (Review Focus 2).
  - `test_choose_stage_resumes_after_errors` (Review Focus 3): `chat_with_image` raising on the first run → an `error` row; a second run with `--resume` and a working fake → the row is labelled and no error row remains among the latest rows.
  - `test_choose_stage_needs_v0` (Review Focus 4): `--candidates V1=... V2=...` → `SystemExit` naming V0, and the fake was never called.

- [ ] **Step 2: Run** `uv run pytest tests/test_naming_choose.py -q -k stage` — Expected: FAIL (`--stage` rejects `choose`).

- [ ] **Step 3: Implement**: `choose` in `--stage` choices; `--candidates` (`nargs="+"`, `VARIANT=PATH`, passed rows only → `variant_names`); `--choose-prompt` (`choices=["c1"]`, default `c1`); items = every row of `--objects` (no `--events`); refuse unless `V0` is among the candidates; `work` calls `choose_object(ask_image, it, candidate_names(oid, variant_names), variant_names["V0"].get(oid), args.seed, choose_prompt=args.choose_prompt)` and adds `"chooser": model`; the progress line shows `row["name"]` for `choose` as for `name`. Add a docstring example under the steering one:
  `uv run scripts/label_descriptions.py --stage choose --objects OBJ --candidates V0=… V1=… V2=… V3=… --out OUT --resume`.

- [ ] **Step 4: Run** `uv run pytest tests/test_naming_choose.py tests/test_naming_fix.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/label_descriptions.py tests/test_naming_choose.py
git commit -m "feat: label_descriptions --stage choose (Addendum C)"
```

---

### Task 3: Today's names for every object; the fresh 50-object sample

**Files:**
- Modify: `scripts/prepare_habitat_instr.py`
- Test: `tests/test_naming_choose.py`

**Interfaces:**
- Produces:
  - `today_names(labels: Path) -> dict[str, tuple[str, str]]` — `{object id: (name, "gemma" | "cosmos")}` with `stage_naming_sample`'s precedence (Gemma train, eval, steer, gate — gate last — over Cosmos train, eval, gate); `stage_naming_sample` uses it.
  - Stage `today-names [--labels --events-dir]` → `<labels>/names_{train,eval,steer}_V0.jsonl`, one row per object of `objects_train.jsonl` / `objects_eval.jsonl` / `steer/objects_eval.jsonl`: `{"id", "need", "name", "passed", "namer"}`.
  - `naming-sample --proportional` (plain random `--n` train objects with `--seed`, group `"random"`, no strata) and `--exclude FILE` (an `objects_sample.jsonl` whose ids are skipped).

- [ ] **Step 1: Write the failing tests**:
  - `test_today_names_stage_covers_every_object`: fixture label files with a gate/steer name conflict (gate wins) and a Cosmos-only object → three `_V0` files, one row per object, the right names and `namer`; an object named nowhere → `passed: False`, `name: None`.
  - `test_naming_sample_proportional_excludes`: 200 train objects (180 wall, 20 object), `--proportional --n 50 --seed 1 --exclude` a file listing 10 ids → 50 random rows, none of the excluded ids, reproducible with seed 1, kinds not forced to 45/45/10.

- [ ] **Step 2: Run** `uv run pytest tests/test_naming_choose.py -q -k "today or proportional"` — Expected: FAIL (`no attribute 'today_names'`).

- [ ] **Step 3: Implement** the helper, the stage (CLI + dispatch entry), the two `naming-sample` options; add the commands to the module docstring's naming-fix block.

- [ ] **Step 4: Run** `uv run pytest tests/test_naming_choose.py tests/test_naming_fix.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_habitat_instr.py tests/test_naming_choose.py
git commit -m "feat: today-names stage and naming-sample --proportional --exclude (Addendum C)"
```

---

### Task 4: The job — choose mode, token budget, concurrency

**Files:**
- Modify: `slurm/naming_variants.sbatch`
- Test: `tests/test_naming_choose.py`

**Interfaces:**
- Consumes: Task 2's CLI; Task 3's `names_<set>_V0.jsonl`.
- Produces: env `MODE` (`name` default | `choose`), `CHOOSERS` (default `"C1 C2 C3"`), `CHOOSE_PROMPT` (default `c1`), `MAX_TOKENS` (default 8192, passed as `--max-tokens` in both modes), `CONCURRENCY` (default 32); set `check` = `$D/naming_check/objects_sample.jsonl`.
  - Candidates: `sample` → `$D/naming_ab/names_V{0,1,2,3}.jsonl`; `check` → `$OUT/names_train_V0.jsonl` + `$OUT/names_train_V{1,2,3}_${NAME_PROMPT}.jsonl`; `train|eval|steer` → `$OUT/names_${s}_V0.jsonl` + `$OUT/names_${s}_V{1,2,3}_${NAME_PROMPT}.jsonl`.
  - Choose outputs: `sample` → `$D/naming_ab/choose_${c}.jsonl`; `check` → `$D/naming_check/choose_${c}.jsonl`; `train|eval|steer` → `$OUT/names_${s}_${c}_${NAME_PROMPT}.jsonl` (the shape `rename-texts --variant C<k>` reads).
  - Every input file of every set is checked before any model is served; a missing one ends the job with its path.

- [ ] **Step 1: Write the failing tests**:
  - `test_naming_variants_sbatch_choose_mode` (static): the three chooser model ids, `MODE="${MODE:-name}"`, `CHOOSERS="${CHOOSERS:-C1 C2 C3}"`, `MAX_TOKENS="${MAX_TOKENS:-8192}"`, `CONCURRENCY="${CONCURRENCY:-32}"`, `--max-tokens "$MAX_TOKENS"` in both modes, `--stage choose`, the output patterns above.
  - `test_naming_variants_sbatch_choose_dry_run`: run the script with `bash` on CPU against a fake `vllm` that serves `/health` from a directory (bound to 0.0.0.0, `--port` from its arguments), a fake `nvidia-smi`, and a fake `UV` that appends its arguments to a file; `MODE=choose CHOOSERS=C1 SETS="sample train"` with fixture input files under a tmp `DATA`/`OUT` → exit 0, and the recorded calls show `--stage choose`, `V0=…/naming_ab/names_V0.jsonl`, `--out …/names_train_C1_n2.jsonl`, `--max-tokens 8192`. A second run with one candidate file removed → non-zero exit naming that file and no `vllm` start. (The fake server answers its first health poll after ~10–20 s, as in the earlier CPU dry runs.)

- [ ] **Step 2: Run** `uv run pytest tests/test_naming_choose.py -q -k sbatch` — Expected: FAIL.

- [ ] **Step 3: Implement** (the `OUT` dir becomes overridable as `OUT="${OUT:-outputs/instr_labels_v3}"` for the dry run); update the header comment with the choose examples:
  `MODE=choose sbatch slurm/naming_variants.sbatch` (comparison on the sample) and `MODE=choose CHOOSERS=C1 SETS="train eval steer" sbatch slurm/naming_variants.sbatch` (rollout with the winner).

- [ ] **Step 4: Run** `uv run pytest tests/test_naming_choose.py tests/test_naming_fix.py tests/test_slurm_thread_caps.py -q && bash -n slurm/naming_variants.sbatch` — Expected: PASS, `bash -n` silent.

- [ ] **Step 5: Commit**

```bash
git add slurm/naming_variants.sbatch tests/test_naming_choose.py
git commit -m "feat: naming_variants.sbatch choose mode, 8192-token budget, concurrency 32 (Addendum C)"
```

---

### Task 5: Scoring choosers and the check page

**Files:**
- Modify: `scripts/audit_instruction_labels.py`, `scripts/audit_instruction_labels.html`
- Test: `tests/test_naming_choose.py`

**Interfaces:**
- Consumes: Task 1's `summarize_d1(rows, "choose")`.
- Produces: `_variant_names(sample_dir, glob="names_V*.jsonl") -> {key: {object id: name or None}}` and the CLI flag `--names-glob` (same default) for `--mode names` and `--score-marks`; the variant key is the file stem after its first `_` (`names_V2` → `V2`, `choose_C1` → `C1`); the score gains `"choosers": {key: summarize_d1(rows, "choose")["choices"] | {"from": …}}` for files whose rows carry `"choice"`.

- [ ] **Step 1: Write the failing tests**:
  - `test_score_choosers`: sample of 3 random objects, marks (one "none"), `choose_C1.jsonl` whose rows' `name` already holds the final name (one pick, one V0 fallback) → `random.C1.correct` as expected and `choosers.C1` present.
  - `test_names_glob_picks_only_the_chooser_file` (Review Focus 5): a folder with `names_V0.jsonl` and `choose_C1.jsonl` → `_variant_names(folder, "choose_*.jsonl") == {"C1": …}` and `names_items` on it shows one name per object (the chooser's).
  - `test_page_sanitizes_doc_ids`: the template contains the doc-id function `(id) => id.replace(/[^A-Za-z0-9_\-.~:@+]/g, "_")`, `persist` writes under it, and the `onSnapshot` loop keys marks by `oid` when present.

- [ ] **Step 2: Run** `uv run pytest tests/test_naming_choose.py -q -k "score_choosers or glob or doc_ids"` — Expected: FAIL.

- [ ] **Step 3: Implement**; then build the page from the real `naming_ab` folder and drive it once in headless Chromium (cached Playwright venv `~/.cache/uv/archive-v0/svHu3FhgoO3s-6xoDyzLv/bin/python -I`, `executable_path` = the cached `chromium_headless_shell-1234`): tick a name on an object whose id holds `,` (`00328-DACaFbApXUe_c038:gap1@-8.0,-1.3`) and check the mark is kept with its `oid`; no page errors.

- [ ] **Step 4: Run** `uv run pytest tests/test_naming_choose.py tests/test_naming_fix.py tests/test_d2_stages.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_instruction_labels.py scripts/audit_instruction_labels.html tests/test_naming_choose.py
git commit -m "feat: score choosers against the names marks, --names-glob, sanitized mark doc ids (Addendum C)"
```

---

### Task 6: README run book, full suite

- [ ] **Step 1:** In `README.md`, after the naming-fix block of "## Turn events and d2 (2026-10-13)", add a short "Naming chooser (Addendum C)" paragraph (why: the audit's best namer was 63%, a right name is among the four candidates 90% of the time) and the run order as commands: `today-names`; `MODE=choose sbatch …` (comparison); `--score-marks <marks> --names-glob "choose_C*.jsonl"`; `VARIANTS="V1 V2 V3" SETS="train eval steer" sbatch …`; `naming-sample --proportional --n 50 --seed 1 --exclude $D/naming_ab/objects_sample.jsonl --out-dir $D/naming_check`; `MODE=choose CHOOSERS=<winner> SETS=check sbatch …`; `--mode names --sample-dir $D/naming_check --names-glob "choose_*.jsonl"`; `MODE=choose CHOOSERS=<winner> SETS="train eval steer" sbatch …`; `rename-texts --variant <winner>`. Commit index-only exactly as in the previous plan's Task 6 (copy of `HEAD:README.md` in the scratchpad, same edit, `git hash-object -w` + `git update-index --cacheinfo`, verify the user's row stays unstaged).
- [ ] **Step 2: Run** `uv run pytest -q` — Expected: all pass (1225 before this plan + the new tests).
- [ ] **Step 3: Commit** `docs: README naming chooser run book`.

---

### Task 7: Run — comparison, full candidates (controller)

- [ ] **Step 1:** `uv run scripts/prepare_habitat_instr.py today-names`; report rows per set and how many have no name.
- [ ] **Step 2: Ask the user**, then submit the comparison `MODE=choose sbatch slurm/naming_variants.sbatch` and, chained after it (`--dependency=afterany:<id>`), the full candidates `VARIANTS="V1 V2 V3" SETS="train eval steer" sbatch slurm/naming_variants.sbatch`. Watch them in the background.
- [ ] **Step 3:** When the comparison ends: save nothing new (the marks are already in the scratchpad: `names_marks/`); run `--score-marks <names_marks> --names-glob "choose_C*.jsonl"`; report per chooser final-name accuracy (random, by kind, non-opening vs the 81/90 ceiling), flagged, none/failed/single rates. Winner = best random accuracy (tie → faster model). If the user wants the one allowed revision (c2), it is a new prompt version through Task 1's `choose_prompt`, test first.
- [ ] **Step 4:** When the full candidates job ends: check completeness per set (rows for every object, no error rows; re-submit with the same line if not — it resumes).

### Task 8: Run — fresh check, rollout (controller; stops for the user)

- [ ] **Step 1:** `naming-sample --proportional --n 50 --seed 1 --exclude $D/naming_ab/objects_sample.jsonl --out-dir $D/naming_check`; ask, then `MODE=choose CHOOSERS=<winner> SETS=check sbatch …`.
- [ ] **Step 2:** `--mode names --sample-dir $D/naming_check --names-glob "choose_*.jsonl" --out outputs/names_check.html`; publish with `capabilities: {db: {}}`; one `ArtifactData list` of `marks`; record the URL in memory. **STOP** for the user's marks.
- [ ] **Step 3:** Save the marks (`ArtifactData list … out_dir`), score with `--sample-dir $D/naming_check --names-glob "choose_*.jsonl"`, report accuracy with a 95% Wilson interval beside the comparison estimate. **STOP** for the user's decision.
- [ ] **Step 4 (after the user accepts):** ask, then `MODE=choose CHOOSERS=<winner> SETS="train eval steer" sbatch …`; then `rename-texts --variant <winner>` (report `rename_meta.json`), `steer` + `steer-select`, and publish G3′ and steering Gate 1′ as in the previous plan's Task 8. **STOP** for the user's marks, then — asking before each — `STAGE=steer` and `STAGE=full`.
