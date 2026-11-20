# Route Checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An image-based check (Qwen3.6 via vLLM) that gives every d3 route text a "probably wrong" score, validated against the user's audit marks, so hard frames can be left out of the d3 **training** set.

**Architecture:** Pure logic (prompt, reply parsing, features, score fit, AUC, removal curve, audit-page parsing) in a new `src/rover_vlm/route_check.py`. Images and manifests come from `scripts/audit_instruction_labels.py` (it owns the page rendering): a calibration manifest from the saved audit pages + marks, a production manifest rendered with `events_item(..., dashed=True)`. A new `--stage check` in `scripts/label_descriptions.py` asks the model; `slurm/check_routes.sbatch` serves Qwen3.6 on the full A100 and runs it. `scripts/route_check_report.py` scores, reports and writes the train drop list; `finalize-descriptions --drop-train-ids` applies it.

**Tech Stack:** Python 3.12 via `uv run`, numpy, Pillow, openai client against vLLM, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-route-checker-design.md` (commit d1fe32f)

## Global Constraints

- Filter **train only**; eval texts are checked but never dropped (eval reports may add a "checker-passed subset" line).
- The checker never judges left/right sides (round i1: Cosmos-Nano at chance on drawn-route sides); the prompt says sides are already correct.
- Model `Qwen/Qwen3.6-35B-A3B-FP8`, full A100 (`--gres=gpu:a100:1`), served like `slurm/naming_variants.sbatch` (`MAX_TOKENS` 8192, `CONCURRENCY` 32).
- Prompt version `k1` is fixed before any reply is read; at most one revision (`k2`).
- Validation bar before anything is dropped: out-of-fold AUC >= 0.75 overall and > 0.6 within each d3 stop kind with >= 20 calibration items.
- Every `sbatch` is submitted only after the user says yes. `uv run` only; cap threads (`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`); no model loading on the login node.
- All outputs under `outputs/route_check/` (gitignored); marks copied to `outputs/audit_marks/<slug>/`.

## Review Focus

- A Qwen3.6 reply that thinks first, wraps the JSON in a code fence, or adds prose after it must still parse; two unreadable replies give a row with `rejects: ["parse"]` (a result, not an `error`, so `--resume` does not loop on it). Test in Task 1.
- A text that names no object ("turn right then stop by the wall") has `names: []` and must score normally. Test in Task 1.
- `confidence` given as `"85%"`, `85.0`, or out of 0-100 is read and clamped; missing -> 50. Test in Task 1.
- The same route judged on two pages is two calibration items: manifest ids are `<slug>/<route_id>`, the place-grouped CV groups by place so they never straddle folds. Test in Task 2 (ids) and Task 4 (groups).
- A drop list that holds eval ids or ids not in train must not touch eval and must say how many ids matched. Test in Task 5.

---

### Task 1: Prompt, reply parsing and the per-route check call

**Files:**
- Create: `src/rover_vlm/route_check.py`
- Test: `tests/test_route_check.py`

**Interfaces:**
- Consumes: `rover_vlm.traces.parse_trace(text) -> (think, answer)`, `rover_vlm.traces.answer_payload(answer) -> dict | None`.
- Produces:
  - `CHECK_PROMPT_VERSION = "k1"`
  - `check_prompt(text: str) -> str`
  - `parse_check_reply(raw: str) -> dict | None` -> `{"names": [{"name": str, "fits": "yes"|"no"|"unsure", "is_object": "yes"|"no"|"unsure"}], "missing_obstacle": bool, "stop_ok": bool, "better_stop": str, "overall_true": bool, "confidence": int}`
  - `check_route(ask, item: dict, seed: int) -> dict` where `ask(image, prompt, seed) -> (raw, finish)` and `item` is a manifest row (`id`, `image`, `text`); returns `{"id", "prompt_version", "attempts", "finish", "rejects", "passed", **parsed}` (`passed` = parsed; on two failures `rejects: ["parse"]`, `passed: False`, `raw: [..]`).

- [ ] **Step 1: Write the failing tests** in `tests/test_route_check.py`:
  - `test_prompt_states_legend_and_never_sides`: `check_prompt("Go left of the sofa, then stop next to the lamp.")` contains the text verbatim, `"magenta"`, `"dashed"`, `"green"`, `"orange"`, `"middle of the room"`, and a sentence saying left/right sides are already correct and must not be judged (assert `"already correct" in p`).
  - `test_parse_plain_json`, `test_parse_after_think_and_code_fence` (`"<think>…</think>```json\n{…}\n``` trailing prose"` parses), `test_parse_bad_returns_none` (prose only -> `None`; JSON missing `overall_true` -> `None`).
  - `test_parse_empty_names_ok`: `"names": []` parses with `names == []`.
  - `test_confidence_forms`: `"85%"` -> 85, `85.0` -> 85, `140` -> 100, `-3` -> 0, missing -> 50.
  - `test_check_route_retries_then_parse_reject`: a fake `ask` returning prose twice -> row `rejects == ["parse"]`, `passed is False`, `attempts == 2`, `len(row["raw"]) == 2`; a fake returning prose then valid JSON -> `attempts == 2`, `passed is True`, seeds called are `(seed, seed + 1)`.
- [ ] **Step 2: Run** `uv run pytest tests/test_route_check.py -q` — expected: FAIL (module not found).
- [ ] **Step 3: Implement** the four names above in `src/rover_vlm/route_check.py` (module docstring: what it does + spec path). The prompt describes the drawing (magenta route, solid in view, dashed out of sight, a ring where it leaves the view; numbered outlines, stop object green, event objects orange), asks the JSON fields above in that order, says sides are already correct, and asks `stop_ok` as "is this a natural way to say where the route ends; would a person rather say e.g. 'in the middle of the room'?" Yes/no strings and booleans both accepted in parsing.
- [ ] **Step 4: Run** `uv run pytest tests/test_route_check.py -q` — expected: all PASS.
- [ ] **Step 5: Commit** `git add src/rover_vlm/route_check.py tests/test_route_check.py && git commit -m "feat: route checker prompt k1, reply parsing and per-route check call"`

### Task 2: Calibration and production manifests

**Files:**
- Modify: `src/rover_vlm/route_check.py` (add page parsing + mark mapping)
- Modify: `scripts/audit_instruction_labels.py` (two modes, docstring lines)
- Test: `tests/test_route_check.py`

**Interfaces:**
- Produces:
  - `audit_page_items(html: str) -> list[dict]` — the `items` of the page's `const DATA = {...};` payload.
  - `mark_wrong(verdict: str) -> bool | None` — `"ok"` -> False, `"unsure"`/`"?"`/missing -> None, anything else -> True.
  - `calibration_rows(slug: str, items: list[dict], marks: dict[str, dict], image_dir: Path) -> list[dict]` — skips unmarked, unsure and drop items (`instruction` starting `"DROPPED"`); writes each item's JPEG (decoded from its `img` data URI) to `image_dir/<slug>/<route_id>.jpg`; rows `{"id": f"{slug}/{route_id}", "route_id", "page": slug, "image": str(path), "text": item["instruction"], "wrong": bool, "verdict": str, "note": str}`.
  - `audit_instruction_labels.py --mode check-calibration --pages SLUG=HTML=MARKS_DIR ... --out DIR` -> `DIR/manifest.jsonl` + `DIR/images/`.
  - `audit_instruction_labels.py --mode check-manifest --events E --objects O --desc D [D ...] --out DIR` -> one row per passed desc row whose route is in E: `{"id": route_id, "route_id", "image", "text": description}`, image from `events_item(rec, row, objects, text, dashed=True)` saved as `DIR/images/<route_id>.jpg`.

- [ ] **Step 1: Write the failing tests**: `test_audit_page_items_reads_payload` (a tiny HTML string with `const DATA = {"items": [...]};\n` -> the items); `test_mark_wrong_mapping` (`ok`, `unsure`, `?`, `stop`, `name`, `keep`, `neither`); `test_calibration_rows_skip_and_ids` (3 items: one ok, one unsure, one `DROPPED…`; marks for all -> 1 row, id `"g3/00800-X_c000_L"`, image file exists and opens with Pillow); `test_same_route_two_pages_two_rows` (same route id under slugs `a` and `b` -> ids `a/…` and `b/…`).
- [ ] **Step 2: Run** `uv run pytest tests/test_route_check.py -q` — expected: the new tests FAIL.
- [ ] **Step 3: Implement** the three functions in `route_check.py` and the two modes in `audit_instruction_labels.py` (`check-manifest` reuses the d3 G3 loading path: labelset record by id, events by id, objects by id; only passed rows; prints counts).
- [ ] **Step 4: Run** `uv run pytest tests/test_route_check.py -q` — expected: PASS. Then build the calibration manifest (marks from the session scratchpad `allmarks/` copied first):

```bash
for s in d2_g3:1Z3ZCbkSxF3uKkKjGVS3Uw d2_g3n2:3zJnPpiLZ3FZY114HXSVa4 d3_g3:RivnqsN9hy4h6AjMRpfspg \
         d3_g3a2:XH9hkaEoy2PpLukPqNX7xF d3_g1:CLEPTAR6Wk8j79GLDXFrgk d3_g1a2:A8BygRV1dzGNCKHtMU9VCu; do
  mkdir -p outputs/audit_marks/${s%%:*} && cp $SCRATCH/allmarks/${s##*:}/marks/*.json outputs/audit_marks/${s%%:*}/; done
uv run scripts/audit_instruction_labels.py --mode check-calibration --out outputs/route_check/calibration \
  --pages d2_g3=outputs/instr_d2_audit.html=outputs/audit_marks/d2_g3 \
          d2_g3n2=outputs/instr_d2_audit_n2.html=outputs/audit_marks/d2_g3n2 \
          d3_g3=outputs/instr_d3_audit.html=outputs/audit_marks/d3_g3 \
          d3_g3a2=outputs/instr_d3_audit_a2.html=outputs/audit_marks/d3_g3a2 \
          d3_g1=outputs/stops_audit.html=outputs/audit_marks/d3_g1 \
          d3_g1a2=outputs/stops_audit_a2_plus.html=outputs/audit_marks/d3_g1a2
```
Expected: ~347 text-page rows (~61 wrong) + ~210 G1 rows; counts per page printed.
- [ ] **Step 5: Commit** `git add src/rover_vlm/route_check.py scripts/audit_instruction_labels.py tests/test_route_check.py && git commit -m "feat: route checker manifests from saved audit pages + marks and from d3 texts"`

### Task 3: `--stage check` and the GPU job

**Files:**
- Modify: `scripts/label_descriptions.py` (stage choice, `--manifest`, work branch, docstring), `src/rover_vlm/descriptions.py:793` (`summarize_d1` gains a `"check"` branch)
- Create: `slurm/check_routes.sbatch`
- Test: `tests/test_route_check.py`

**Interfaces:**
- Consumes: `check_route` (Task 1), manifest rows (Task 2).
- Produces: `label_descriptions.py --stage check --manifest M --out OUT [--resume]` writing one `check_route` row per manifest id; `summarize_d1(rows, "check")` adds `{"parse_failed": int, "overall_true_share": float}`; `MANIFESTS="calibration" sbatch slurm/check_routes.sbatch` (default) or `MANIFESTS="train eval"`, reading `outputs/route_check/<set>/manifest.jsonl`, writing `outputs/route_check/<set>/check_k1.jsonl`, two passes per set (`--resume`), the server stopped on exit.

- [ ] **Step 1: Write the failing test** `test_summarize_check`: rows `[{passed: True, overall_true: True, rejects: []}, {passed: False, rejects: ["parse"]}, {error: "x"}]` -> `errors == 1`, `parse_failed == 1`, `overall_true_share == 1.0`.
- [ ] **Step 2: Run** `uv run pytest tests/test_route_check.py::test_summarize_check -q` — expected: FAIL.
- [ ] **Step 3: Implement** the summary branch and the stage (items = manifest rows, key `"id"`; the progress line shows `overall_true`/`confidence`). Write `slurm/check_routes.sbatch` from `slurm/naming_variants.sbatch` (same `#SBATCH` header, `cosmos3_serve_lib.sh`, `MAX_TOKENS=8192`, `CONCURRENCY=32`, input check before serving, header comment with both submit lines).
- [ ] **Step 4: Run** `uv run pytest -q` (whole suite) — expected: all PASS; `bash -n slurm/check_routes.sbatch` clean; `uv run scripts/label_descriptions.py --stage check --manifest outputs/route_check/calibration/manifest.jsonl --out /tmp/x.jsonl --limit 0` prints `0 to label` without contacting a server.
- [ ] **Step 5: Commit** `git add scripts/label_descriptions.py src/rover_vlm/descriptions.py slurm/check_routes.sbatch tests/test_route_check.py && git commit -m "feat: --stage check (Qwen3.6 route checker) + slurm/check_routes.sbatch"`
- [ ] **Step 6: ASK the user, then submit** `MANIFESTS=calibration sbatch slurm/check_routes.sbatch`.

### Task 4: Score fit, AUC, removal curve and the report

**Files:**
- Modify: `src/rover_vlm/route_check.py`
- Create: `scripts/route_check_report.py`
- Test: `tests/test_route_check.py`

**Interfaces:**
- Produces:
  - `FEATURES = ("name_no", "object_no", "missing_obstacle", "stop_no", "overall_no", "p_false")`; `features(row: dict) -> np.ndarray | None` (None for a parse-rejected row; `p_false = 1 - c/100` if `overall_true` else `c/100`).
  - `fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray` (weights incl. bias; numpy Newton/IRLS, no sklearn).
  - `predict(w: np.ndarray, X: np.ndarray) -> np.ndarray`
  - `oof_scores(X, y, groups: list[str], k: int = 5, seed: int = 0) -> np.ndarray` (folds by group = place `scene_cXXX`).
  - `auc(scores, wrong) -> float` (ties averaged).
  - `removal_curve(scores, wrong, shares=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)) -> list[dict]` with `share, threshold, kept_accuracy, wrong_caught`.
  - `scripts/route_check_report.py --manifest M --check C [--events-dir data/prepared_habitat_instr_v5] --out DIR` -> `DIR/report.json` (overall / per page / per d3 stop kind / per verdict type AUC for the fitted OOF score and the `p_false` baseline, the curve, parse-failed count, weights fitted on all rows) + a printed table + the 20 highest-scored and 20 lowest-scored wrong items; `--apply C2 --weights DIR/report.json --threshold T --out-ids FILE` writes the ids of rows scoring >= T.

- [ ] **Step 1: Write the failing tests**: `test_auc_known` (`[0.1, 0.4, 0.35, 0.8]` vs `[F, F, T, T]` -> 0.75; all ties -> 0.5); `test_fit_separates` (synthetic X where one feature equals y -> fitted AUC > 0.95); `test_oof_groups_never_straddle` (rows of the same place land in one fold; assert via a fold-assignment helper `group_folds(groups, k, seed) -> list[int]` also exported); `test_removal_curve_monotone` (kept_accuracy non-decreasing as share grows on a perfectly ranked toy set; share 0.10 removes ceil(0.1 n) rows); `test_features_none_on_parse_reject`.
- [ ] **Step 2: Run** `uv run pytest tests/test_route_check.py -q` — expected: new tests FAIL.
- [ ] **Step 3: Implement** the functions and the report script (module docstring with both command lines). Stop kinds for d3 rows come from `events_{train,eval}.jsonl` via `route_events.stop_kind`.
- [ ] **Step 4: Run** `uv run pytest -q` — expected: PASS.
- [ ] **Step 5: Commit** `git add src/rover_vlm/route_check.py scripts/route_check_report.py tests/test_route_check.py && git commit -m "feat: route checker score fit, AUC, removal curve and report"`
- [ ] **Step 6 (after the calibration job):** run the report, publish it as an artifact (AUCs vs the bar, the curve, the top items with their images), and ask the user: pass / revise prompt once (`k2`) / stop.

### Task 5: Train drop list in `finalize-descriptions`

**Files:**
- Modify: `scripts/prepare_habitat_instr.py:892-930` (`stage_finalize_descriptions`) and its argparser (`finalize-descriptions` subparser), module docstring run line
- Test: `tests/test_route_check.py` (or the existing finalize test file if one covers `stage_finalize_descriptions`)

**Interfaces:**
- Produces: `finalize-descriptions --drop-train-ids FILE` — one route id per line; matching train records are removed before the shuffle; eval untouched; stats gain `"dropped_checker": n` for train and the run prints `drop list: <matched> of <lines> ids matched train; <k> ids are eval/unknown and were ignored`. A helper `drop_records(recs: list[dict], ids: set[str]) -> tuple[list[dict], int]` in `src/rover_vlm/route_check.py`.

- [ ] **Step 1: Write the failing tests**: `test_drop_records_train_only` (records `a, b, c`; ids `{b, zz}` -> kept `a, c`, matched 1); `test_drop_list_never_touches_eval` (call the finalize helper path with an eval id in the list on a tiny fixture -> eval output unchanged; if a full fixture is too heavy, assert `stage_finalize_descriptions` applies `drop_records` only to `splits["train"]["kept"]` via a monkeypatched `finalize_descriptions`).
- [ ] **Step 2: Run** `uv run pytest tests/test_route_check.py -q` — expected: FAIL.
- [ ] **Step 3: Implement** `drop_records` and the option.
- [ ] **Step 4: Run** `uv run pytest -q` — expected: PASS.
- [ ] **Step 5: Commit** `git add scripts/prepare_habitat_instr.py src/rover_vlm/route_check.py tests/test_route_check.py && git commit -m "feat: finalize-descriptions --drop-train-ids (route checker, train only)"`

### Task 6: Production run book (after job 97236 and a passed validation)

No new code; each step logged in `.superpowers/sdd/2026-11-10-stop-phrases-d3/progress.md`.

- [ ] **Step 1:** Build the manifests: `uv run scripts/audit_instruction_labels.py --mode check-manifest --events data/prepared_habitat_instr_v5/events_train.jsonl --objects data/prepared_habitat_instr_v5/objects_train.jsonl --desc outputs/instr_labels_v3/desc_train_mix_d3.jsonl --out outputs/route_check/train` and the same for eval. Expected: ~20,037 train / ~2,084 eval rows (minus failed texts).
- [ ] **Step 2: ASK, then submit** `MANIFESTS="train eval" sbatch --time=12:00:00 slurm/check_routes.sbatch`.
- [ ] **Step 3:** `route_check_report.py --apply` on train and eval with the calibration weights; publish the size-vs-accuracy curve on train; the user picks the threshold. The same threshold on eval writes `outputs/route_check/eval/flagged_ids.txt` (never dropped; the d3 eval report uses it for its extra "checker-passed subset" line).
- [ ] **Step 4:** `uv run scripts/prepare_habitat_instr.py finalize-descriptions --round d3 --desc-suffix _mix --drop-train-ids outputs/route_check/train/drop_ids.txt`; report train size before/after and the eval checker-passed subset size.
