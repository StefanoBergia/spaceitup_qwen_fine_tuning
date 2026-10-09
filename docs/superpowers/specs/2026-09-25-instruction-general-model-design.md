# Design: the general instruction-conditioned model (v2)

**Date:** 2026-09-25
**Status:** approved design, not yet implemented
**Branch:** `feat/instruction-paths` (continues v1)
**Builds on:** `2026-09-24-instruction-conditioned-paths-design.md` (v1) and its results: on 92
held-out two-route frames the Cosmos-trained adapter reached 79.3% pair success (template-trained
55.4% on Cosmos wording, round-4 model 0%). Instruction prompt i2: Cosmos3-Nano is handed the side
and names the landmark; a text check verifies it (Cosmos cannot judge left/right on drawn routes).

## Goal

One Qwen3.5-2B LoRA, trained on every round-4 frame, that
1. still does round 4's job with the plain prompt (no instruction), and
2. follows a route instruction when one is given.

Plus a larger held-out pair set and three robustness probes (blocked-side instruction,
hand-written wording, the instructed prompt on ordinary frames).

### Non-goals
- Real-image evaluation (deferred by the user).
- A template arm, subset-size sweeps, the 0.8B model.
- Reasoning traces in the output.
- New Habitat frames. **Note:** the larger eval set is mostly yaw-rotated copies of v1's
  viewpoints (282 pairs from 104 viewpoints vs 92 from 92), so it tests robustness to small
  camera rotation, not more independent evidence. Truly independent pairs need new frames.

## Source data (verified 2026-09-25)

Round 4's split, `data/prepared_habitat_v4` (scene-disjoint), **original and yaw-augmented frames
alike**: augmented frames are full re-renders with their own `fpv_paths.json` candidates (v1
wrongly assumed otherwise and skipped them).

| | frames | `direct` | pair frames (orig + aug) | viewpoints behind the pairs | single-route |
|---|---|---|---|---|---|
| train | 23,770 | 48 | 2,263 (813 + 1,450) | 823 | 21,459 |
| eval | 2,591 | 7 | 282 (102 + 180) | 104 | 2,302 |

Pair frames pass v1's `build_pair` filters (both sides feasible, entry ≤ 0.05, routes ≥ 0.05
Fréchet apart). Single-route = everything else that is not `direct` (including two-sided frames
that fail those filters: 2,250 + 179 train, 234 + 24 eval).

## 1. Dataset — `data/prepared_habitat_instr_v2/`

Per frame (round-4 prompt; answer `{"path": [...]}`):

| record | id | prompt | answer |
|---|---|---|---|
| plain | `<sample>` | round-4 prompt, byte-identical to `data/prepared_habitat_v4` | the correct route |
| instructed | `<sample>_L` / `_R` (its side) | round-4 + Cosmos instruction for the correct side | the correct route |
| opposite (pair frames) | the other `_L` / `_R` | round-4 + instruction for the other feasible side | that route, start snapped |

- `direct` frames: plain only.
- Pair frames: both instructed records carry `pair_id`, `alt_path`, `entry_gap` (v1 semantics).
- Single-route frames: the instructed record carries `alt_path` = the best other-side candidate
  (feasible or not; else any other drawable candidate) — used only for Cosmos's overlay — and
  **no** `pair_id`, so pair metrics ignore it. `habitat_meta.parent_sample_id` is kept for
  clustered intervals.

Counts: train 23,770 plain + 25,985 instructed/opposite = **49,755**; eval 2,591 + 2,866 = 5,457.

Eval files:
- `eval_plain.json` — the plain records (= round-4 eval).
- `eval_instructed.json` — the correct-side instructed record of every non-`direct` eval frame.
- `eval_pairs.json` — both instructed records of the 282 eval pair frames.
- `probe_wrong_side.json` — 300 single-route eval frames (fixed seed) whose *other* side is
  infeasible, rebuilt from that blocked candidate with an instruction for its side; `alt_path` =
  the feasible route. "follows" = obeyed the instruction into the blocked side.
- `probe_handwritten.json` — 30 eval pairs (original frames, fixed seed), both sides, with
  hand-written instructions from `configs/instructions_handwritten_v2.json`.

Entry point: a new `general` stage (build + overlays-in-memory + `finalize-general`) in
`scripts/prepare_habitat_instr.py`; v1's `pairs` / `finalize` stay untouched.

## 2. Labelling

- One unattended job `slurm/label_instructions.sbatch` (serve + client in one allocation,
  localhost, like `label_full_path.sbatch`; `--resume` everywhere; `--time 14:00:00`, 3g.40gb).
- Stage 0 gate: 200 single-route eval records; pass = accept ≥ 0.9 and drawing refs = 0
  (`label_instructions.py --gate --min-accept 0.9`; pair survival is not used for this gate).
- Full run: 25,985 train + 2,866 eval instructed records = 28,851 calls (~4 h at v1's rate).
- `--retry-rejected`: one retry with seed+1 for rejected records.
- Images inline: `render_generation_image` can return JPEG bytes; `chat_with_image` accepts bytes
  and sends a `data:image/jpeg;base64,...` URL. No overlay files for v2.
- Rule change: minimum instruction length 6 → **4 words** (77 of v1's 91 train rejects were short
  correct instructions like "Go left around the sofa."). Other i2 rules unchanged. Prompt stays i2.
- Records still rejected after the retry: frame keeps its plain record; for a pair frame both
  instructed records are dropped.

## 3. Training

- Fresh LoRA on `Qwen/Qwen3.5-2B`, round-4 hyperparameters, **1 epoch** over ~49.8k records
  (~3,110 steps; each image seen ~2× as in round 4's 2 epochs), `--time 14:00:00`.
- `train.py` gains `--save-steps N` (+ `save_total_limit=2`) and `--resume` (resume from the latest
  checkpoint in `--output-dir`). Defaults keep today's behaviour.
- Output `outputs/runs_instr_v2/general/`.

## 4. Evaluation

`slurm/eval_instruction_v2.sbatch` (skips finished cells), ~2 h:

| cell | model | set | pass criterion |
|---|---|---|---|
| 1 | general | `eval_plain` | median point error ≤ 0.013 and `win_vs_to_goal` ≥ 69% (round 4: 0.011 / 71.6%) |
| 2 | general | `eval_instructed` | not worse than cell 1 on median error and win rate |
| 3 | general | `eval_pairs` | pair success ≥ 75% and not significantly below cell 4 |
| 4 | v1 Cosmos adapter | `eval_pairs` | reference |
| 5 | general | `probe_wrong_side` | reported only (obey rate) |
| 6 | general | `probe_handwritten` | reported only |

Metrics: existing ones plus pair success split by `source` (original / augmented) and a bootstrap
interval that resamples **viewpoints** (`parent_sample_id`, else the sample id), not pairs.

Report: new page **"One Model, Both Prompts"** from `scripts/visualize_instruction_general.py`,
reusing the v1 page's chart/figure helpers (waypoint occlusion markers included). The v1 artifact
is left as the record of v1.

**Success:** cells 1–3 pass.

## Testing (pytest, CPU)

- `general` builder: record kinds and counts on synthetic samples; `direct` → plain only; plain
  record prompt byte-identical to `build_record(framing="endpoints")`; single-route record has
  `alt_path` and no `pair_id`; opposite record only on pair frames.
- Wrong-side probe record: built from the blocked candidate, `alt_path` is the feasible route.
- Clustered bootstrap groups by `parent_sample_id`.
- In-memory overlay returns JPEG bytes; `chat_with_image` builds a data URL for bytes.
- `instruction_rejects` with the 4-word minimum; `--retry-rejected` selects only rejected rows.
- `train.py` argument parsing for `--save-steps` / `--resume` (no model load).
