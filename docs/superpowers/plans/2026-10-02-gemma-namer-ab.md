# Gemma namer A/B on the d1 label gate

**Spec:** `docs/superpowers/specs/2026-09-30-route-descriptions-design.md` (Addenda A, A2, A3 binding);
this plan adds "Addendum A4 (namer A/B)". Decided by the user 2026-10-02.

**Why.** d1 label audit of gate r2 (job 92136, https://claude.ai/artifact/BcifA65C2LJptFCmD9PeLW):
84 correct / 11 wrong name / 2 unnatural / 2 unsure / 1 wrong stop — below the 90% gate. No side
errors. The wrong names mostly name a small detail inside the outline instead of the outlined
thing ("metallic door handle", "light switch", "electrical outlet cover", "carved wooden chair
leg", "white door panel"); 6/11 are `between` events. Both unnatural ones come from the style
"a command that also says briefly why or where it leads" ("... to avoid the obstacle"). A
part-word reject list was scored offline and rejected (catches 3/11 wrong, drops 3/84 ok).
User chose: Gemma 3 12B (google/gemma-3-12b-it, cached) as the object namer, Cosmos3-Nano d2
still composes; plus a "whole thing" naming line and no "why it leads" style for d1.

## Global Constraints

- Naming prompt: add, right after the "1. Name what is inside the outline ..." instruction, the
  sentence: `Name the whole thing the outline covers, not a small part on it (a handle, a switch,
  a socket, a leg).` Same prompt for both namers.
- d1 compose styles: `D1_STYLES` in descriptions.py = the shared `STYLES` minus
  "a command that also says briefly why or where it leads", plus "a calm, step-by-step
  instruction"; d1 compose uses `D1_STYLES` (stable per route id as now). The shared `STYLES`
  (i4) is unchanged.
- `PROMPT_VERSION = "d3"`.
- Serving: `slurm/cosmos3_serve_lib.sh` passes the Cosmos-only vLLM flags
  (`--hf-overrides ...Cosmos3Reasoner...`, `--mm-encoder-tp-mode data`) only when `$MODEL`
  is a Cosmos3 model; every other flag and both functions keep their names and behaviour.
  Add `cosmos3_stop_server` (kill $SERVER_PID, wait for it, remove the endpoint file).
- `slurm/label_descriptions.sbatch`: `NAMER=cosmos|gemma` (default cosmos = today's behaviour,
  byte-identical outputs/paths). With `NAMER=gemma`: serve `google/gemma-3-12b-it` for the two
  name passes, stop it, then serve Cosmos3-Nano for the two compose passes. Gemma outputs use
  tag suffix `_gemma`: `names_<tag>_gemma_d1.jsonl`, `desc_<tag>_gemma_d1.jsonl` (gate:
  `names_gate_gemma_d1.jsonl`, `desc_gate_gemma_d1.jsonl`); Cosmos files are never touched.
  Both namers use the same `--max-tokens` (4096).
- `scripts/audit_instruction_labels.py --mode d1`: new `--ids FILE` (one route id per line);
  when given, the page shows exactly those passed routes (in file order) instead of a random
  sample, and prints how many ids were not passed in the --desc file.
- Pure Python/bash; no model is ever loaded on the login node. Tests: `tests/test_descriptions.py`,
  `tests/test_audit*.py` if present (else add to the closest test file); shell changes get a
  `bash -n` syntax check and a dry test of the MODEL branch (source the lib with a fake vllm
  binary on PATH/VENV that echoes its args) in a pytest using subprocess.
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest -q -x` (975 passed before).
- README.md: user-owned uncommitted row -> index-only commits; never stage scripts/evaluate.py.

## Task 1: naming line + d1 styles + prompt d3

Files: `src/rover_vlm/descriptions.py`, `tests/test_descriptions.py`.
Tests first: naming prompt contains the exact sentence for all needs; `D1_STYLES` excludes the
"why or where it leads" style, contains "a calm, step-by-step instruction", has 4 entries;
compose for any route id uses a style from D1_STYLES; shared STYLES unchanged; PROMPT_VERSION "d3".

## Task 2: Gemma serving + NAMER in the sbatch

Files: `slurm/cosmos3_serve_lib.sh`, `slurm/label_descriptions.sbatch`, a test file for the
shell (subprocess with a fake vllm), README d1 section (index-only: `NAMER=gemma`).
Default path (NAMER unset) must produce the same commands and file names as today.

## Task 3: audit --ids + spec Addendum A4

Files: `scripts/audit_instruction_labels.py`, its tests, spec Addendum A4 (the audit numbers
above, the decisions, the rejected part-word list), README line for `--ids`.
