# Design: instruction-conditioned path prediction

**Date:** 2026-09-24
**Status:** approved design, not yet implemented
**Branch:** `feat/instruction-paths` (off `feat/real-image-eval`, which carries rounds 3–4 and v7;
`main` does not)
**Builds on:** Habitat round 4 (`data/prepared_habitat_v4`, `--framing endpoints`)

## Context

Until now the language in this project has been an *output*: Cosmos3-Nano wrote a reasoning
trace justifying the ground-truth path, and Qwen learned to emit trace + path. This phase flips
it: the language becomes an *input*. The prompt carries a short route instruction ("go left
around the blue table") and the model must output the path that obeys it.

Round 4 already hands the model the route's entry point and goal, and hits a 0.011 median point
error. On most frames that pins the path completely, so an instruction would be redundant and a
model could learn to ignore it while scoring well. The experiment only means something on frames
where **the same image and the same endpoints admit more than one correct path**.

## Goal

Answer one question cheaply: **can Qwen3.5-2B be steered by a route instruction?** And a second
one: **do Cosmos-written landmark instructions add anything over a bare left/right template?**

### Non-goals (deferred until this result is in)
- Instructions on every round-4 frame (approach A).
- Frames whose two routes enter the image at different points (see "The entry-point leak").
- Instruction-only or goal-only prompts — the entry point and goal stay in the prompt.
- Real-image evaluation.
- Any change to round 4's data, splits or results.

## Source data (verified 2026-09-24)

`meta.json` lists candidate routes, each labelled `left` / `right` / `direct` with a `feasible`
flag; `fpv_paths.json` holds their image-space projections. Round 4's ground-truth path *is* one
of these candidates (`habitat_data.select_correct_path` takes `meta["label"]`).

Over the 10,552 samples in `habitat_generated`:
- 2,604 have a feasible `left` **and** a feasible `right` route to the same goal.
- Their goals always coincide (end-point distance 0 in every pair).

### The entry-point leak
In 57% of those pairs the two routes enter the frame 0.5–1.0 apart (typically one from each
side edge); in the other 43% they enter at the same point (distance ≈ 0). Where they differ, the
round-4 prompt's entry point already reveals the side, and the instruction is bypassable.
**Decision: keep only pairs whose entry points are within 0.05.**

### Resulting set (original frames only, no `_y*` yaw copies; round 4's scene-disjoint split)

| split | shared-entry pairs | records | `label` side left / right |
|---|---|---|---|
| train | 973 | 1,946 | 548 / 425 |
| eval | 125 | 250 | 75 / 50 |

**Revised at implementation (final review):** pairs whose two routes are within 0.05 Fréchet
of each other are also dropped (`routes_too_similar`, 183 pairs — one route drawn twice, where
following the instruction cannot be measured), and the start snap compares the full `[x, y, v]`
so the start's visibility flag cannot leak the side. Final set: **813 train / 102 eval pairs
(1,626 / 204 records)**; `label` side left/right 458/355 train, 59/43 eval.

Yaw-augmented frames are excluded because the candidate projections belong to the unrotated camera.

## 1. Dataset

Each kept frame yields **two records**: same image, same entry point and goal, one per side,
each with its own instruction and its own path.

- **Path:** `select_correct_path(fpv_paths, meta, index=None)` gains an optional `index`
  (default `meta["label"]`, so existing callers are unchanged). Both records go through the same
  clip + resample as round 4.
- **Prompt:** round 4's `endpoints_prompt` plus one line `Route instruction: "<text>"`.
- **Answer:** unchanged, `{"path": [[x, y, v], ...]}`.
- **Eval records** also carry `alt_path` (the other side's route, same processing) for scoring.
- **Two instruction sources, same frames and records, only the text differs:**
  - `instr_template` — fixed text from the candidate's side label, e.g. "Go around the obstacle
    on its left." No Cosmos.
  - `instr_cosmos` — Cosmos3-Nano's landmark-grounded description (section 2).
- **Entry point:** a separate `scripts/prepare_habitat_instr.py` (revised during planning: it
  derives from the already integrity-checked `data/prepared_habitat_v4` splits, the way
  `prepare_habitat_traces.py` derives from a prepared split, instead of adding a pair-producing
  mode to the manifest-driven `prepare_habitat.py`). Two stages:
  - `pairs` — build both records per kept frame (template text) + the Cosmos overlay images →
    `pairs_train.json`, `pairs_eval.json`, `overlays/`. This is the labelling input.
  - `finalize --instructions <train.jsonl> <eval.jsonl>` — keep pairs whose two records both
    passed, write `train_template.json`, `train_cosmos.json`, `eval_template.json`,
    `eval_cosmos.json` (identical ids across template/cosmos) and `meta.json`.
- **Asserts:** eval ∩ train = ∅ by scene; every pair has both records or neither.

## 2. Cosmos instruction labelling

New module `src/rover_vlm/instructions.py` holds prompts, parser and the reference detector —
kept apart from `traces.py`, whose leakage machinery is trace-specific. Labelling
runs from a new `scripts/label_instructions.py` (revised during planning: each record needs two
chained calls with two different images, which does not fit `label_traces.py`'s one-call
trace loop). The endpoint/resume/output-guard logic it shares with `label_traces.py` moves
into `src/rover_vlm/labelling_io.py`, used by both.

**Generation (one call per record).**
- Image: frame with **both** routes drawn — target bold, the other thin and faded — so Cosmos
  sees what distinguishes the target.
- Ask for one instruction a person would give a driver: 6–25 words, naming at least one object
  visible in the frame, never mentioning drawn lines, colours, numbers or coordinates.
- Carry over the v6/v7 lessons: no concrete nouns in the prompt's own examples; bounded scratch
  block; "exactly one JSON object" (`{"instruction": "..."}`); harvest the answer, not `<think>`.

**Verification (a second, independent call).**
- Image: both routes drawn identically, numbered 1 and 2 in random order (seeded per id).
- Ask which route the instruction describes → `{"choice": N}`.
- A record is kept if the choice is correct; **a pair is kept only if both of its records pass**,
  preserving left/right balance.

**Automatic rejection** (`instruction_rejects()`): references to the drawing ("the pink line",
"path 1", "the marked route"), coordinates, or length outside 6–25 words. ("No object named" was
dropped during planning — not detectable by pattern; the verifier and the side-only count cover it.) Also *counted, not rejected*:
instructions whose only content is a side word — a high share means Cosmos adds nothing.

**Gate before the full run.** 100 eval frames first. Pass = drawing references 0, pair survival
≥ 80%, and the user reads 20 instructions. Then the full set: ~1,100 frames × 2 records × 2 calls
≈ 4,400 calls, ~2 h against the existing `serve_cosmos3.sbatch` endpoint.

**Known weakness:** the same model writes and verifies. The real test is Qwen's pair success
on held-out frames; `nvidia/Cosmos-Reason2-8B` (cached) is available as an independent verifier
if self-agreement looks inflated.

**Revised after the first gate (2026-09-25, prompt i1 → i2).** The i1 gate failed at chance
level: pair survival 25% (0.5² = 25%), verifier accuracy 59%. A control run gave the verifier
the *template* instruction on the same check images: 63% accuracy, 37% pair survival — Cosmos3-Nano
cannot tell left from right on drawn routes (it picks "the curved one" for "go around"), and
48% of i1 instructions named no side while only 67% of the rest named the right one. Since the
side is known exactly, i2 hands it to Cosmos and asks only for the landmark; the Cosmos verify
call is dropped and replaced by text rules (`instruction_rejects(text, side)`: correct side word
present, opposite absent, something named, no drawing reference). Remaining unchecked risk: a
named landmark that is not in the frame.

## 3. Training

| run | data | question |
|---|---|---|
| `instr_template` | ~1,946 records, template text | steerable at all? |
| `instr_cosmos` | same records, Cosmos text | do landmark instructions add anything? |

- Fresh LoRA on base `Qwen/Qwen3.5-2B`, round-4 hyperparameters; ~3 epochs (~400 steps,
  ~1.3 h each on a 3g.40gb slice at round 4's measured 11.7 s/step).
- Not initialised from the round-4 adapter: it has learned that the endpoints determine the path
  and to always emit the shortest feasible side.
- The existing `slurm/train_habitat.sbatch <train-file> <output-dir> --epochs 3`, submitted
  twice (MaxJobs=3). Outputs `outputs/runs_instr_v1/<arm>/`. `train.py` and the loader are
  unchanged.

## 4. Evaluation

**Metrics** (new functions in `src/rover_vlm/eval.py`; `evaluate.py --task habitat` adds an `instruction`
block to `metrics.json` whenever the records carry `alt_path`, so no new task is needed):

| metric | definition | instruction-blind model scores |
|---|---|---|
| **pair success** (headline) | on *both* records of a frame, prediction is closer (Fréchet) to the instructed route than to `alt_path` | **0%** |
| follows instruction | same test, per record | 50% |
| point error / Fréchet | vs the instructed route | – |
| parse rate | as before | – |

An instruction-blind model returns the same path for both records of a pair, so it wins
exactly one of the two and its pair success is 0.

**Runs** (`slurm/eval_instruction.sbatch`, ~250 records × ~1.1 s ≈ 5 min each):

| model | template eval | Cosmos eval |
|---|---|---|
| `instr_template` | ✓ | ✓ |
| `instr_cosmos` | ✓ | ✓ |
| round-4 LoRA | | ✓ (reference, expect ≈ 0% pair success) |
| base Qwen3.5-2B | | ✓ (zero-shot instruction following) |

**Success criteria.**
1. `instr_template` pair success ≳ 70% → the mechanism works.
2. `instr_cosmos` ≥ `instr_template` on the Cosmos eval → landmark text earns its labelling
   cost. Otherwise a side label suffices and Cosmos is not worth it at scale.
3. Point error on the instructed route within ~2× round 4's 0.011 median → following the
   instruction does not cost geometry.

**Report.** `scripts/visualize_instruction_results.py` → `outputs/instruction_results.html`: the
2×2 table plus sample frames with both predictions overlaid (reusing `overlay.py`).

## Testing

pytest (CPU, the gate on the login node — no CPU smoke-train for the 2B):
- `select_correct_path(index=...)` equals the old behaviour when `index` is omitted.
- Pair builder: shared-entry filter, both-or-neither, scene-disjoint split, `alt_path` present.
- `instruction_rejects()` on hand-written positive/negative strings.
- Pair-success metric: an identical-prediction pair scores 0; a correctly split pair scores 1.
- Instruction prompt contains no concrete nouns outside the placeholder.
