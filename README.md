# SPACEITUP — Rover Path VLM (Qwen3.5-2B fine-tuning)

Fine-tune Qwen3.5-2B (vision-language) to output a traversable path as waypoints from a
single forward-facing RGB image. **Current phase:** feasibility check on the open
[BAAI/ShareRobot](https://huggingface.co/datasets/BAAI/ShareRobot) `trajectory` subset
(6,870 samples: image + instruction → normalized `[[x,y],...]` waypoints) before building
the Habitat + Cosmos Reason 3 pipeline. See `CLAUDE.md` for full project context.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Everything lives on NFS, shared between the
login node and the GPU node (`thor`, SLURM `A100` partition).

```bash
uv sync            # creates .venv with Python 3.12 + all deps
```

## Repo layout

```
configs/          # YAML run configs (dataset size, LoRA rank, lr, per-GPU batch settings)
src/rover_vlm/    # importable package — shared data/model logic
scripts/          # thin CLI entrypoints (each has a docstring with usage)
slurm/            # sbatch files for GPU jobs on thor
data/             # (gitignored) downloaded dataset + prepared splits
outputs/          # (gitignored) checkpoints, logs, eval results, inspection reports
```

## Workflow

All CPU-only steps run on the login node; GPU steps are launched manually via
`sbatch slurm/<job>.sbatch`.

### 1. Download dataset + model (login node)

```bash
uv run scripts/download.py               # both dataset and model
uv run scripts/download.py --dataset-only
uv run scripts/download.py --model-only
```

Dataset lands in `data/sharerobot/trajectory/`; model weights go to the default
Hugging Face cache (`~/.cache/huggingface`, on NFS → visible from thor).

### 2. Inspect the dataset

```bash
uv run scripts/inspect_dataset.py        # report + overlay images → outputs/inspection/
```

### 3. Prepare splits

```bash
uv run scripts/prepare_data.py           # normalized conversations + nested subsets → data/prepared/
```

Held-out eval split is fixed across all runs; training subsets (500/1K/2K/5K/full) are
nested so scaling comparisons aren't confounded by subset composition.

### 4. Train (GPU)

Always validate the pipeline before submitting long jobs:

```bash
uv run scripts/train.py --smoke        # CPU, ~10 min: full pipeline on 4 samples
sbatch slurm/train_smoke.sbatch        # GPU, ~5 min: 20 steps on a 1g.20gb slice
```

**One-command option** — the entire experiment (all trainings + all evals + comparison)
as a single resumable job on one 3g.40gb slice (~4-6 h; resubmit to continue after a
failure or time limit — finished stages are skipped):

```bash
sbatch slurm/run_all.sbatch
```

Or run the stages individually — scaling runs (3g.40gb slice each; output dir derived
from the file name):

```bash
sbatch slurm/train.sbatch                                  # train_500 (default)
for f in train_1000 train_2000 train_5000 train_full; do
  sbatch slurm/train.sbatch data/prepared/$f.json
done
```

Adapters land in `outputs/runs/lora_<name>/adapter`. Logs: `outputs/slurm/`.

### 5. Evaluate (GPU) and compare (CPU)

```bash
sbatch slurm/eval.sbatch               # base model zero-shot only
sbatch slurm/eval.sbatch all           # base + every adapter under outputs/runs/
uv run scripts/compare_evals.py        # login node: table + scaling plot -> outputs/eval/
uv run scripts/visualize_predictions.py  # login node: interactive prediction explorer (HTML)
```

Every eval uses the same fixed split (`data/prepared/eval.json`). Per-model results go
to `outputs/eval/<tag>/{predictions,metrics}.json`; the comparison lands in
`outputs/eval/comparison.md` and `outputs/eval/scaling_curve.png`.

Metrics: parse rate, in-range rate, and — over parseable predictions — mean/median
resampled point error, endpoint error, and discrete Fréchet distance (all in
normalized [0,1] image coordinates).

`scripts/visualize_predictions.py` is the *qualitative* companion to the scaling plot:
it reads the same `predictions.json` files and writes a single self-contained
`outputs/eval/prediction_explorer.html` that overlays each model's predicted path (and
ground truth) on the source image, side by side, for a set of eval samples spread across
difficulty (ranked by base-model error). Open it in a browser or publish it as an
artifact. Tune with `--per-bucket` / `--max-image-px`.

## Habitat phase

A second, closer-to-target open dataset: Facebook Habitat–generated rover frames with
an A*-computed traversable path and per-waypoint visibility (`v` = 1 visible / 0
obstructed by the frame edge or an obstacle). Same conversation format and training/eval
code as the ShareRobot phase — only the data source and task prompt differ; the goal is
to sanity-check the pipeline on data that's structurally closer to the real rover task.

```bash
uv run scripts/prepare_habitat.py        # scans the mounted dataset -> data/prepared_habitat/
uv run scripts/inspect_habitat.py --num 16  # overlay path + visibility -> outputs/inspection_habitat/
```

`prepare_habitat.py` reads every sample dir under the mounted dataset, keeps
correct-path-in-FOV samples, and writes fixed-seed nested splits (`eval`, `train_500`,
`train_1000`, `train_2000`, `train_full`) plus `meta.json` to `data/prepared_habitat/`.
Images are referenced by their absolute NFS path (never copied).

**One-command option** — the entire habitat experiment (all trainings + all evals +
comparison) as a single resumable job on one 3g.40gb slice (~2-3 h; resubmit to continue
after a failure or time limit — finished stages are skipped):

```bash
sbatch slurm/run_all_habitat.sbatch
```

Or run the stages individually. Train (GPU) — same `scripts/train.py`, just pointed at the habitat splits:

```bash
sbatch slurm/train_habitat.sbatch                                   # train_full (default)
for f in train_500 train_1000 train_2000; do
  sbatch slurm/train_habitat.sbatch data/prepared_habitat/$f.json
done
```

Adapters land in `outputs/runs/habitat_<name>/adapter`.

Evaluate (GPU) and compare (CPU):

```bash
sbatch slurm/eval_habitat.sbatch habitat_base                                            # zero-shot baseline
sbatch slurm/eval_habitat.sbatch habitat_train_full outputs/runs/habitat_train_full/adapter  # per adapter, tag habitat_train_<size>
uv run scripts/compare_evals.py --eval-dir outputs/eval_habitat --meta data/prepared_habitat/meta.json
```

Results go to `outputs/eval_habitat/<tag>/{predictions,metrics}.json`, separate from the
ShareRobot `outputs/eval/` tree so the two phases' comparisons don't collide.

## Habitat classification phase

A second task on the *same* Habitat frames: each sample offers several candidate paths
(3 or 5), of which one is canonical and — in 46% of samples — more than one is
acceptable. The candidates are drawn on the frame with numbered badges and the model
answers `{"choice": N}`. This tests discrimination (does the model understand
traversability?) rather than generation.

```bash
uv run scripts/prepare_habitat_choice.py          # renders composites -> data/prepared_habitat_choice/
uv run scripts/inspect_habitat_choice.py --num 12 # overlays -> outputs/inspection_habitat_choice/
```

`prepare_habitat_choice.py` **requires `prepare_habitat.py` to have run first**: split
membership is copied from `data/prepared_habitat/` so both tasks use exactly the same
frames and can be compared sample by sample. Rendered composites go to
`data/prepared_habitat_choice/images/` (gitignored — the source dataset is never
written to); re-running skips images that already exist.

Run the whole experiment as one resumable job, then compare:

```bash
sbatch slurm/run_all_habitat_choice.sbatch
uv run scripts/compare_evals.py --task choice \
    --eval-dir outputs/eval_habitat_choice --meta data/prepared_habitat_choice/meta.json
```

Adapters land in `outputs/runs/habitat_choice_<name>/adapter`; results in
`outputs/eval_habitat_choice/<tag>/`.

Metrics: parse rate, valid-index rate, **strict accuracy** (vs. the canonical `label`)
and **accepted accuracy** (vs. the full `accepted` set — the headline number, since many
samples have several valid answers). Accuracies count an unparseable answer as wrong, so
they can't be inflated by dropping failures.

Build the results report (login node, after the evals):

```bash
uv run scripts/visualize_choice_results.py           # -> outputs/eval_habitat_choice/choice_results.html
uv run scripts/visualize_choice_results.py --per-bucket 6
```

Emits a single self-contained page (inline CSS/JS, base64 images — no external requests,
so it can be published as an artifact directly): headline tiles, accuracy vs. training-set
size against both chance baselines, the chosen-index distribution for base vs. LoRA vs.
ground truth, the metrics table, an error breakdown by decision margin, and a gallery of
real eval frames grouped by how ambiguous the decision was (mistakes included). The page
template lives in `scripts/_choice_report.html`.

> **Dashes are a reading aid, not training input.** `inspect_habitat_choice.py` and the
> report gallery re-render frames with occluded stretches **dashed** so you can see what
> passes behind an obstacle. The composites in `data/prepared_habitat_choice/images/` draw
> every path **solid** — the model gets no occlusion cue and must infer depth from the
> image. Pass `--solid` to either script to see exactly what the model receives.
> `render_choice_image()` defaults to solid for this reason; keep the defaults when
> generating data, or the training images change and the existing adapters no longer match.

> **Read accuracy against the right baseline.** Every sample contains a straight-line
> `direct` candidate that is correct in only 11 of 4,398 samples. A model that learns
> nothing but "never pick the straight line" scores ~0.49 accepted accuracy, versus ~0.36
> for uniform guessing. `meta.json` and the comparison table both report
> `chance_accepted_excluding_direct` — that ~0.49 line, not raw chance, is the bar to beat.

> **Note:** the interactive `scripts/visualize_predictions.py` explorer is ShareRobot-only
> — it expects 2-element `[x,y]` predictions and does not yet understand the habitat
> `{"path","goal"}` format. For qualitative habitat inspection use
> `scripts/inspect_habitat.py`; a habitat explorer is a deferred follow-up.

## Comparing base model sizes

Both habitat sweeps are parameterized by base model, so the same experiment can be
re-run on a smaller Qwen3.5 and the scaling curves overlaid. The model argument
suffixes the **output trees**; eval tags are deliberately left identical across
models, which is what makes the overlay possible.

```bash
sbatch slurm/run_all_habitat.sbatch 0.8b          # -> outputs/runs_0.8b/, outputs/eval_habitat_0.8b/
sbatch slurm/run_all_habitat_choice.sbatch 0.8b   # -> outputs/eval_habitat_choice_0.8b/
```

No argument (or `2b`) reproduces the original Qwen3.5-2B paths exactly. An unrecognized
model name exits non-zero rather than falling back to the default — a typo must never
quietly write into a completed tree.

Overlay the two models once both sweeps finish (output goes to `--eval-dir`; the
`--compare-dir` tree is read-only):

```bash
uv run scripts/compare_evals.py \
    --eval-dir outputs/eval_habitat_0.8b --label 0.8B \
    --compare-dir outputs/eval_habitat --compare-label 2B \
    --meta data/prepared_habitat/meta.json
```

In the overlaid plot, **colour is the metric and line style is the model** (solid =
`--eval-dir`, dashed = `--compare-dir`).

`scripts/train.py` and `scripts/evaluate.py` both take `--model-id` directly if you
want a size the sbatch wrappers don't list. `evaluate.py` records the resolved id in
`metrics.json` and `train.py` writes a `train_config.json` next to each adapter, so a
result tree says which model produced it instead of relying on directory naming.

> **A LoRA target that matches nothing is fatal, not a warning.** PEFT silently ignores
> unmatched `target_modules`, which would train fewer parameters than intended and only
> surface as a flat loss curve hours into a job. `train.py` checks every target against
> the loaded module tree and exits with the unmatched names. Qwen3.5-2B and Qwen3.5-0.8B
> both match `LORA_TARGETS_TEXT` in full (6 full-attention layers, 18 linear-attention,
> 24 MLPs), so no change is needed between those two.

## Reasoning-trace phase (Cosmos3-Nano)

Both Habitat tasks predict *what* to do but not *why*. This phase labels each sample with
a natural-language reasoning trace, generated by `nvidia/Cosmos3-Nano`, which will later be
trained into Qwen3.5 inside its native `<think></think>` block.

**Strategy: justify the ground truth.** Cosmos is shown the frame *and* the known-correct
answer and asked to explain why it holds. It never has to solve the task, so a trace can
never contradict its label.

That creates one hazard worth naming, because it silently destroys the dataset if missed.
At inference Qwen will **not** know the answer, so a trace like *"since candidate 1 is
given as correct…"* teaches exactly the wrong reflex — such a trace is worse than none.
`rover_vlm.traces.leakage_spans()` detects that phrasing, and **leakage rate is the gate
for the whole phase**: it must be 0 before labelling at scale.

### Setup (login node — needs network)

```bash
uv run scripts/download.py --cosmos     # ~35 GB into the shared HF cache
bash slurm/cosmos3_env.sh               # builds .venv-cosmos3 (vLLM, cu128)
```

A second venv is unavoidable: vLLM pins its own torch, which cannot coexist with this
repo's `torch==2.10.0+cu128`. Only the server lives there — the client is plain HTTP and
runs from the main `.venv`.

**The CUDA build is the fiddly part.** `uv`'s `--torch-backend=cu128` only picks *torch*'s
wheel index; it does nothing to vLLM's own precompiled extension. The vllm wheel on PyPI is
a CUDA 13 build, so a plain `uv pip install vllm==0.21.0` dies at import with
`ImportError: libcudart.so.13` even though torch resolved to cu128 correctly — thor's
driver is 570.158.01 and CUDA 13 needs ≥ 580. vLLM publishes per-CUDA builds as **GitHub
release assets**, and `cosmos3_env.sh` installs the `+cu129` one, which links
`libcudart.so.12` (already provided by the cu128 torch stack) and runs fine on a 12.8
driver under CUDA minor-version compatibility. There is no `+cu128` asset — cu129 is the
CUDA-12 variant.

### Serve, then iterate (server on thor, client on odin)

```bash
sbatch slurm/serve_cosmos3.sbatch        # writes outputs/cosmos3_endpoint.txt when READY
```

The job serves only the *reasoner* tower of the Mixture-of-Transformers checkpoint (via an
`--hf-overrides` architecture swap) and holds it for 8 h. Because odin reaches thor over
TCP, you iterate prompts from the login node against that one long-lived server instead of
paying a 35 GB model load per attempt:

```bash
# smoke test — 3 samples, full prompt and raw reply printed
uv run scripts/label_traces.py --task choice \
    --split data/prepared_habitat_choice_v2/eval.json --limit 3 --show

# an iteration round, after editing the prompts in src/rover_vlm/traces.py
uv run scripts/label_traces.py --task path \
    --split data/prepared_habitat_v2/eval.json --limit 30 --prompt-version v2
```

Each run writes `outputs/traces/<task>_<prompt-version>.jsonl` (flushed per line, so a
killed run keeps its partial output) and prints the gate:

```
  parsed a trace   30/30
  answer matches   30/30      # drift check: did it echo the sample it was shown?
  LEAKAGE           0/30      # must be 0
  median words     94         # target 60-150, trainable length
```

Bump `--prompt-version` each round so revisions stay comparable on the same sample ids.
`scancel` the serve job when done.

### What the prompt iteration actually found

Five rounds, each fixing a distinct failure. Worth reading before editing the prompts,
because most of these are invisible in the summary numbers:

1. **Harvest the answer, not `<think>`.** v1 scraped Cosmos's think block and every trace
   opened *"Okay, let's see. The user wants me to explain why candidate 1 is…"*. That block
   is its private deliberation about *our request*, so it will always talk about the user
   and the answer. The trace is now a `reasoning` field inside the answer JSON, and
   `<think>` is discarded.
2. **Never put concrete nouns in the prompt.** The rules once illustrated specificity with
   *"the floor between the sofa and the kitchen counter"* — and 2 of 12 traces then
   described a kitchen counter that was not in the frame. A labelling prompt must not
   supply vocabulary for the scene; `tests/test_traces.py` now asserts it doesn't.
3. **Don't make the model echo large ground truth.** Re-emitting the full waypoint list
   cost ~200 tokens, halved throughput, and tipped Cosmos into pretty-printed fenced JSON
   that dropped the `<answer>` tags entirely. It echoes just the goal now — same drift
   check, 1% of the cost.
4. **Bound the scratch block.** Told to "think as long as you need", Cosmos occasionally
   spent 3,000+ words and hit the token cap before ever closing `</think>`, losing the
   sample. Healthy runs use ~200-400 scratch words.
5. **Demand one JSON object explicitly.** Otherwise it writes the reasoning as bare prose
   and the echoed answer — the only check that it looked at the right sample — is gone.

The parser is deliberately forgiving of all of this: missing `<answer>` tags, fenced JSON,
and bare prose are all salvaged, because at full-dataset scale a formatting slip must not
silently delete a good label. `answer_payload()` keeps *unverifiable* separate from
*wrong*, so a formatting slip is never scored as the model drifting off-sample.

### Where things live

| Piece | File |
|---|---|
| Prompts, `<think>` parsing, leakage detection | `src/rover_vlm/traces.py` |
| Labelling client (`--limit N --show` is the smoke test) | `scripts/label_traces.py` |
| vLLM venv builder (login node) | `slurm/cosmos3_env.sh` |
| Serving job | `slurm/serve_cosmos3.sbatch` |
| CPU tests | `tests/test_traces.py` |

Full-dataset labelling is deliberately **not** part of this phase — it starts once a prompt
version clears the gate above.

## Visualizations — where each one lives

Every report is a **single self-contained HTML file** (inline CSS/JS, base64 images,
zero network requests — required by the artifact CSP), regenerated from the eval
outputs on the login node. All are CPU-only; none need a GPU.

| Report | Regenerate with | Writes to |
|---|---|---|
| ShareRobot prediction explorer | `uv run scripts/visualize_predictions.py` | `outputs/eval/prediction_explorer.html` |
| Habitat path + visibility results | `uv run scripts/visualize_habitat_results.py` | `outputs/eval_habitat/habitat_results.html` |
| Habitat classification results | `uv run scripts/visualize_choice_results.py` | `outputs/eval_habitat_choice/choice_results.html` |
| 2B vs 0.8B comparison (both tasks) | `uv run scripts/visualize_model_comparison.py` | `outputs/model_comparison.html` |

Published artifacts (private to the owner; republish the same file path to update in
place, or pass the URL as `url=` from another session):

- ShareRobot explorer — https://claude.ai/code/artifact/71daf6f4-76c8-47bf-8825-540d32347f9b
- Habitat path + visibility — https://claude.ai/code/artifact/4fdd1859-b6d9-4e14-b20d-5b41b24a9574
- Habitat classification — https://claude.ai/code/artifact/a5c5b899-1148-4f41-8c7a-4325558d39cf
- 2B vs 0.8B comparison — https://claude.ai/code/artifact/1195c008-dc27-482c-953f-4b017965f89e

Useful flags:

```bash
uv run scripts/visualize_choice_results.py --per-bucket 6      # more gallery samples
uv run scripts/visualize_choice_results.py --solid             # training composites verbatim
uv run scripts/visualize_model_comparison.py --per-side 5      # more disagreement frames
uv run scripts/visualize_model_comparison.py --a-label 2B --b-label 0.8B
```

`visualize_model_comparison.py` is the only report that tests its own claims: both
models are scored on identical frames, so it runs McNemar's exact test (classification)
and a paired bootstrap (regression) via `src/rover_vlm/compare.py` and labels each gap
significant or not, rather than leaving significance to be guessed from bar heights.

Every report is regenerable from the repo. The two Habitat generators take `--eval-dir`
/ `--label`, so the same page can be rebuilt for either base model:

```bash
uv run scripts/visualize_habitat_results.py \
    --eval-dir outputs/eval_habitat_0.8b --label Qwen3.5-0.8B
```

Shared drawing code lives in `src/rover_vlm/overlay.py` (white-underlaid strokes;
waypoint fill encodes visibility, so colour stays free to mean "which model"), and the
goal-visibility class-imbalance maths in `rover_vlm.eval.goal_visibility_confusion`.
