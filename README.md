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

## Re-running everything on a grown dataset (`VERSION`)

The Habitat generator keeps producing samples, so the dataset grows between rounds
(4,398 sample dirs at the first prep → 9,688 on 2026-07-21). Re-preparing reshuffles the
fixed-seed splits over a different record list, so **a new prep is a new experiment**:
its eval set is not the old eval set, and its numbers belong in their own tables.

`VERSION` keeps the rounds apart. It suffixes the data dir *and* both output trees
together, so a re-prep can never land in a completed run's results; `SIZES` picks which
training subsets to run. Empty `VERSION` (the default) reproduces the original paths
exactly.

Round 2 (`_v2`, ~9.1K samples, eval grown to 1,000, `train_full` only) — prep on the
login node:

```bash
uv run scripts/prepare_habitat.py \
    --out-dir data/prepared_habitat_v2 --eval-size 1000 --train-sizes ""

mkdir -p data/prepared_habitat_choice_v2
ln -s ../prepared_habitat_choice/images data/prepared_habitat_choice_v2/images

uv run scripts/prepare_habitat_choice.py \
    --src-splits data/prepared_habitat_v2 --out-dir data/prepared_habitat_choice_v2
```

The symlink reuses the composites already rendered for round 1 — rendering is keyed by
sample id and skips files that exist, so only the new frames cost anything. It only ever
*adds* to the shared images dir, leaving round 1 valid.

Then all four experiments (2B / 0.8B × regression / classification) in **one job**:

```bash
sbatch slurm/run_all_experiments.sbatch      # ~6-9 h serial on a 3g.40gb slice
```

It runs the two per-task scripts four times with `VERSION=_v2 SIZES=train_full`, then
builds both 2B-vs-0.8B overlays. Every stage is skipped when its adapter or
`metrics.json` already exists, so a job that dies or times out is resumed by simply
resubmitting. Results land in `outputs/runs_v2{,_0.8b}/` and
`outputs/eval_habitat{,_choice}_v2{,_0.8b}/`.

> With a single training size, `scaling_curve.png` is one marker per model — the real
> deliverable this round is `comparison.md` (base vs. `train_full`, per model).

> **Splits from different rounds are not interchangeable.** Each round is internally
> clean (`prepare_habitat.py` asserts eval ∩ train = ∅), but rounds overlap each other:
> 39% of the v2 eval set was in round 1's training data, and 89% of round 1's eval set is
> in v2's training data. So a round-1 adapter must never be scored on the v2 eval split,
> and vice versa. This rules out the obvious "score the old adapter on the new eval set
> to isolate the effect of more data" experiment — cross-round numbers are *different
> test sets, each valid on its own*, not a controlled comparison.

## Results — round 1 vs. round 2

Round 1: 3,660 train / 500 eval (jobs 87428, 87657, 87773, 87794).
Round 2 (`_v2`): 8,140 train / 1,000 eval (job 87995, all four experiments, 6h40m).

> **Read the round-to-round deltas with care.** The two rounds use *different eval sets*,
> so a delta mixes "more training data" with "different test frames". For the part of
> the difference that is actually attributable to data volume, see the doubly-held-out
> analysis below.

**Path + visibility regression**

| Model | Round | Train | parse | mean pt err | Fréchet | vis. acc | goal err | goal vis. acc |
|---|---|---|---|---|---|---|---|---|
| 2B base | 1 | — | 0.954 | 0.438 | 0.714 | 0.689 | 0.402 | 0.243 |
| 2B base | v2 | — | 0.965 | 0.432 | 0.698 | 0.695 | 0.372 | 0.225 |
| 2B + LoRA | 1 | 3,660 | 1.000 | 0.123 | 0.228 | 0.831 | 0.033 | 0.750 |
| **2B + LoRA** | **v2** | **8,140** | 1.000 | **0.102** | **0.187** | **0.877** | **0.025** | **0.867** |
| 0.8B base | 1 | — | 0.026 | 0.794 | 1.009 | 0.454 | 0.729 | 0.615 |
| 0.8B base | v2 | — | 0.021 | 0.862 | 1.176 | 0.567 | 0.934 | 0.762 |
| 0.8B + LoRA | 1 | 3,660 | 1.000 | 0.131 | 0.245 | 0.784 | 0.029 | 0.688 |
| **0.8B + LoRA** | **v2** | **8,140** | 1.000 | **0.108** | **0.198** | **0.869** | **0.026** | **0.860** |

**Path classification** (chance 0.275 strict / 0.363 accepted / **0.499 excluding the `direct` shortcut**)

| Model | Round | Train | strict | accepted | picked direct |
|---|---|---|---|---|---|
| 2B base | 1 | — | 0.256 | 0.322 | 0.364 |
| 2B base | v2 | — | 0.221 | 0.282 | 0.366 |
| 2B + LoRA | 1 | 3,660 | 0.872 | 0.898 | 0.000 |
| **2B + LoRA** | **v2** | **8,140** | **0.887** | **0.917** | 0.002 |
| 0.8B base | 1 | — | 0.240 | 0.302 | 0.330 |
| 0.8B base | v2 | — | 0.277 | 0.336 | 0.299 |
| 0.8B + LoRA | 1 | 3,660 | 0.832 | 0.876 | 0.000 |
| **0.8B + LoRA** | **v2** | **8,140** | **0.861** | **0.892** | 0.002 |

The base-model rows are the sanity check on the two eval sets: with no training involved,
a round's numbers should only move by sampling noise, and they mostly do (2B regression
parse 0.954 → 0.965, mean error 0.438 → 0.432). The 0.8B base rows swing more, but they
are computed from ~2% of frames and mean nothing — see the survivorship note below.

### What did the extra data actually buy?

Settled by three comparisons with different biases (report:
`uv run scripts/visualize_crossround.py`, and `scripts/compare_crossround.py` for the
numbers alone). **Path regression improved; classification did not measurably.**

Regression, paired on 553 frames absent from round 1's pool — **10 of 10 comparisons
favour v2 significantly**, both model sizes. Waypoint error 0.1189 → 0.1031 (2B),
visibility 0.8432 → 0.8743, goal visibility p=2.9e-07. The exposure asymmetry does *not*
confound this: the round-1 model does not degrade in unfamiliar rooms at all (it scores
slightly better on the 553 than on its own eval), so path geometry transfers across
scenes and the gain is attributable to data.

Classification is the opposite, and the raw test is misleading:

| | raw paired gap | − exposure artifact | = attributable to data | independent estimate |
|---|---|---|---|---|
| 2B strict | +5.61 pts | 4.56 | **+1.05** | +1.5 (ns) |
| 2B accepted | +4.70 pts | 2.82 | **+1.88** | +1.9 (ns) |
| 0.8B strict | +3.07 pts | 0.20 | **+2.88** | +2.9 (ns) |
| 0.8B accepted | +2.35 pts | 0.80 | **+1.55** | +1.6 (ns) |

The round-1 **2B** model loses 4.56 points of strict accuracy purely from being tested in
rooms it never trained in, which accounts for ~80% of its apparent deficit. Discounting
that leaves +1.05, essentially identical to the independent estimate from each model's own
eval split. Two methods with unrelated biases landing on the same value is the strongest
evidence available here.

> A finding worth carrying forward: the **2B is far more scene-dependent than the 0.8B**
> for classification (−4.56 vs −0.20 points on unfamiliar rooms). The larger model leans
> on room-specific memorization the smaller one cannot afford. Every rover deployment room
> is unseen, so this is worth measuring directly — a **scene-disjoint** split (group by
> scene before shuffling in `make_splits`) would make it visible instead of latent.

#### The three methods

No single comparison settles it, so all three are on the report. They agree everywhere
except where the exposure asymmetry bites — which is itself the evidence that the
asymmetry, not the data, drives the classification result.

| | frames | pairing | exposure bias | needs GPU |
|---|---|---|---|---|
| **paired-553** | v2 eval absent from round 1's pool | paired | v2 favoured (0% vs 99% scene exposure) | yes — `slurm/eval_crossround.sbatch` |
| **matched-55** | held out by *both* rounds | paired | none (100%/100%) | no — existing predictions |
| **unpaired** | each model on its own round's split | unpaired | none | no |

The **matched-55** set is unbiased but only powered for large effects: it confirms the
regression gains (2B mean error 0.0961 vs 0.1200, CI excludes zero) and returns ns on
classification with discordant counts of 4-vs-2 — too few to resolve a 2-point difference.
That is a power limit, not evidence of no effect.

The **unpaired** comparison is licensed by the difficulty control above (the untrained base
model scores identically on both eval splits, every CI crossing zero), and it is what the
exposure-discounted paired-553 numbers are checked against.

Splitting a round's own data (`SIZES="train_2000 train_full"`) remains the only way to get
a scaling curve free of all of this — which is what the round-1 sweep did.

### Reading the numbers

Both models saturate the output format after LoRA (parse 1.000), from a 2B base at 0.965
and a 0.8B base at 0.021 — the small model cannot produce the format zero-shot at all.

**2B vs 0.8B, paired tests on the same 1,000 frames** (`src/rover_vlm/compare.py`):

| Metric | 2B | 0.8B | test | verdict |
|---|---|---|---|---|
| mean point error | 0.1020 | 0.1076 | bootstrap CI [-0.0126, +0.0012] | not significant |
| Fréchet | 0.1875 | 0.1975 | CI [-0.0227, +0.0023] | not significant |
| waypoint visibility acc | 0.8774 | 0.8692 | CI [-0.0015, +0.0175] | not significant |
| goal point error | 0.0252 | 0.0264 | CI [-0.0022, -0.0001] | significant, but ~1 px at 768 |
| goal visibility | — | — | McNemar p=0.573 | not significant |
| strict accuracy | 0.887 | 0.861 | McNemar p=0.0148 | **significant** |
| accepted accuracy | 0.917 | 0.892 | McNemar p=0.0088 | **significant** |

Read: **geometry converges, occlusion reasoning does not.** With enough data the 0.8B
matches the 2B on where the path goes; it stays behind on choosing which drawn path is
actually traversable. This reproduces round 1 on a doubled eval set — and without round
1's survivorship caveat, since both fine-tunes now parse 1.000 of the eval set, so the
error means cover all 1,000 frames rather than a self-selected subset. The caveat still
applies to **0.8B base**, which parses 2.1% — its error columns describe 21 frames and
the comparison page flags them rather than plotting them as a peer.

Two other things worth knowing: **2B base scores *below* chance on classification**
(0.221 vs 0.275 strict, picking the straight-line `direct` candidate 36.6% of the time) —
it is actively drawn to the shortcut, not guessing. And both fine-tunes drop
`picked_direct` to 0.002 while clearing the 0.499 no-direct floor by ~0.39, so the
accuracy is real discrimination rather than shortcut avoidance.

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

### Full-dataset labelling (path regression, v6)

v6 cleared the gate, so the whole path-regression dataset gets labelled with it — 8,140
`train_full` + 1,000 `eval` = **9,140 samples**:

```bash
sbatch slurm/label_full_path.sbatch
```

One job serves the model *and* runs the client against localhost, so there is no second
terminal to keep alive — the 2026-07-22 server was started under `srun` and died with its
terminal. It asks for `gpu:3g.40gb:1`, not the full A100: the reasoner only loads 16.65 GiB.

Output lands in `outputs/traces_full/path_v6_{train_full,eval}.jsonl`, deliberately *not* in
`outputs/traces/`, so the prompt-iteration runs the report is built from stay untouched.

**The job is safe to re-submit.** Every client call passes `--resume`, which skips finished
samples and retries failed ones, so a timeout or `scancel` costs only what was in flight.
`label_traces.py` also refuses to truncate a non-empty output file unless you pass
`--resume` or `--overwrite`.

Runtime: sequential labelling measured 5–10 s/sample, i.e. 13–25 h. `--concurrency 8`
(the default in the sbatch) overlaps requests — verified at 7.9× client-side against a mock
endpoint, though the real ceiling is vLLM's batching on one MIG slice, so budget 2–4 h plus
~10 min of model load. `--time` is 12 h, well past that.

The choice task is **not** included: this run is path regression only.

#### The run, and the leak it exposed

Job 88327 **completed** — 9,140 samples in 3h37m at ~43/min, 0 request errors, 17 no-trace,
17 truncated, median 94 words. But it surfaced a failure the 20–50-sample prompt-iteration
gate could not resolve: **~5% of path traces leak the answer** by referring to a route the
rover was handed — *"I continue along the planned path"*, *"trust the pre-planned
trajectory"*, *"hiding the ground truth for several waypoints"*.

This is prompt-caused and specific to the path task. Its prompt states *"The route it should
take … is: {json}"* and asks the model to justify *"this route"*, so the model adopts the
framing of a pre-existing plan. The choice task, which hands no route, has **0%** — that
asymmetry is the diagnosis. At inference Qwen has no planned route, so a trace that defers to
one teaches the wrong reflex, which is exactly what the leakage gate exists to prevent.

The gate missed it because `leakage_spans()` targeted meta-framing (*"the correct answer"*,
*"as stated"*), not handed-route vocabulary — so *"path v6: 0/20 leakage"* was a **detector
blind spot, not a clean result** (re-scanning the smoke runs with the fixed detector finds
1/20 on path v5 and v6). The lesson: a ~5% rate cannot be resolved at n=20–50, and the gate
must include handed-route patterns for any task that hands the answer shape over.

#### Filtering (`scripts/filter_traces.py`)

`leakage_spans()` was extended with the handed-route family (clear-only boundary: *"the
planned/pre-planned/designated/known path"*, *"ground truth"* — but **not** bare *"intended
path"*, which reads as the rover's own intent). Then:

```bash
uv run scripts/filter_traces.py        # both path_v6 splits
```

reads the raw `outputs/traces_full/*.jsonl`, re-scores leakage with the current detector, and
writes a sibling `*.filtered.jsonl` **non-destructively** — every row preserved, annotated
with `keep` / `drop_reason` / re-scored `leakage`. Drop priority: error → no-trace →
truncated → leak → drift (verifiable JSON whose goal ≠ ground truth). Salvaged prose lacking
a JSON envelope is *kept*. Measured result:

| split | n | kept | dropped (leak / no-trace / drift) |
|---|---|---|---|
| train_full | 8,140 | **7,768** (95.4%) | 348 / 16 / 8 |
| eval | 1,000 | **953** (95.3%) | 46 / 1 / 0 |

The kept set is leak-free by the detector that defines the gate (the script asserts it), and
7,768 clean training traces is well past phase-1's ~2K sweet spot. The dropped ~400 leave
gaps; recovering them with a v7 prompt that forbids pre-existing-route language, re-labelling
just the dropped ids (`label_traces.py --ids … --resume`), is a documented future option, not
done here.

> ℹ️ **The path source images moved — and the prepared splits were repointed.** On
> 2026-07-27 the FPV renders the splits reference at
> `/nfs/projects/spaceitup/rover_navigation/data/habitat_generated/dataset/<scene>/samples/…`
> were reorganised (by another user) into a `train/`+`val/` layout — the `dataset/` tree is
> gone. The labelling run read them before the change, so the traces are unaffected.
> `scripts/repoint_habitat_paths.py` rewrote every prepared JSON's image paths
> `…/dataset/` → `…/train/` (all resolve there; "both" cases are byte-identical copies),
> backing originals up to `data/_prepared_backup_pre_move_<stamp>/` and asserting 0 missing
> after — so training resolves images again. It is dry-run by default; `--apply` to write.
> Not touched: `habitat_data.py`'s `DATASET_ROOT` (the scan root for a *fresh* prepare run) —
> with the source now split across `train/`+`val/`, how a new prepare should treat that split
> is a design decision, not a mechanical repoint.

`uv run scripts/visualize_traces.py` renders every revision side by side: gate metrics per
version (each column carrying a definition of what it does and does not mean), then one card
per frame showing **the prompt that was sent beside the trace it produced**, for every
version that covered that frame, and finally the literal diff between consecutive prompts.

Prompts are re-rendered per frame through each version's own recovered code rather than
reusing the stored string — a recovered prompt belongs to the one sample it was rendered
for, and pasting it onto another card would show the wrong ground truth.

Classification frames are re-rendered from the source dataset with `dashed=True`, so a
candidate's occluded stretches break into dashes and you can see whether a route the trace
calls blocked really does pass behind something. As in `visualize_choice_results.py` this
is a **reading aid for the report only** — the training composites stay solid, because
dashing hands the model an occlusion cue it is supposed to infer. `--solid` shows the
composites verbatim; `--dataset-root` points at the Habitat samples.

The prompt text itself is not in git: `src/rover_vlm/traces.py` was first committed already
at v6, and the run records store the version tag but not the prompt body. It is instead
recovered from the session transcript by `scripts/recover_trace_prompts.py`, which replays
every Write, Edit and Bash patch in order and snapshots the file at the moment each
labelling run was launched:

```bash
uv run scripts/recover_trace_prompts.py \
    --transcript ~/.claude/projects/<project>/<session-id>.jsonl
```

It writes `scripts/trace_prompt_history.json` — the durable artifact, since the script only
works while the transcript survives. Two guards keep the output honest: the replay must
reproduce the current `traces.py` byte for byte or nothing is written, and
`visualize_traces.py` refuses to build if the recovered latest prompt no longer matches
what the code renders today. Snapshots are keyed on **when a run happened**, not on
`PROMPT_VERSION` — the constant lagged behind the runs (it jumped `v3` → `v5`, and the v4
runs were tagged with the CLI flag while the module still said v3).

The recovered prompts corrected the record: the comment block in `traces.py` credits v3
with removing the illustrative furniture example, but the diffs show that landed at **v4**,
one run later — so the v2 *and* v3 labels were both produced with `"the sofa and the
kitchen counter"` still in the prompt.

#### Training on the traces (`<think>` block)

The kept traces are wired into training as Qwen3.5's native reasoning. Two steps:

```bash
# 1. Merge kept traces into the prepared split (CPU, no GPU)
uv run scripts/prepare_habitat_traces.py
#    -> data/prepared_habitat_v2/train_full_traced.json   (7,768 records)

# 2. Fine-tune BOTH sizes (2B + 0.8B) and eval each, in one GPU job (recommended).
#    Sequential on one 3g.40gb slice, idempotent/resumable; evals pass --enable-thinking.
sbatch slurm/train_traced_both.sbatch
#    -> outputs/runs_v2_traced{,_0.8b}/ and outputs/eval_habitat_v2_traced{,_0.8b}/

#    Or a single size directly. train.py is task-agnostic — it trains on whatever the
#    train-file contains; the collator picks up the "reasoning" field automatically:
sbatch slurm/train_habitat.sbatch \
    data/prepared_habitat_v2/train_full_traced.json outputs/runs/habitat_traced
# 3. and eval WITH --enable-thinking so the model reasons before answering:
uv run scripts/evaluate.py --task habitat --enable-thinking \
    --adapter outputs/runs/habitat_traced/adapter --tag habitat_train_full_traced \
    --eval-file data/prepared_habitat_v2/eval.json --max-new-tokens 384
```

`prepare_habitat_traces.py` joins each prepared record to its kept trace **by id** and
attaches the reasoning as a top-level `"reasoning"` field. The training label stays the
**authoritative** ground-truth `{"path":…,"goal":…}` — the trace row's own echoed answer is
never used, only its reasoning prose. Records with no kept trace (leaked/no-trace/drift) are
**dropped by default** (a reasoning fine-tune wants every sample to reason); `--keep-untraced`
instead trains them answer-only.

`TrajectoryCollator` renders the reasoning through the processor's official
`reasoning_content` channel — it lands inside `<think>…</think>` before the answer — and
moves the label mask boundary to the `enable_thinking=True` generation prompt (which ends at
an open `<think>\n`). The trained region is therefore **exactly** what the model must produce
at inference: the reasoning, then `</think>`, then the JSON answer. A record without
`"reasoning"` renders an empty `<think></think>` and trains answer-only, identical to the
pre-trace pipeline — so plain and traced runs share one code path. The collator asserts the
generation prompt is a true token prefix of the full sequence, failing loud if a future
chat-template change ever slips the mask boundary (a silent mistrain otherwise).

Because the base template defaults thinking **off** (an empty closed `<think></think>` in the
generation prompt), eval of a trace-trained adapter **must** pass `--enable-thinking`, or the
model is handed a pre-closed think block and never reasons. `parse_path_answer` already skips
reasoning prose and scans for the `{"path",…,"goal"}` object, so no parser change is needed;
give generation enough room (`--max-new-tokens 384`) for ~100 words of trace plus the answer.

#### Traced results report (`scripts/visualize_traced_comparison.py`)

```bash
uv run scripts/visualize_traced_comparison.py   # -> outputs/eval_habitat_v2_traced/traced_comparison.html
```

One self-contained HTML page covering both the **2B-vs-0.8B** comparison and **per-sample
reasoning inspection**, built from the traced eval trees, the **plain (no-reasoning) LoRA**
trees (`outputs/eval_habitat_v2{,_0.8b}`), and `eval.json` (for the `id → image` lookup). It
shows: headline tiles (traced vs plain per size), a metrics table with six rows —
**base / plain SFT / traced** for each size — and a hover-definition on every column, paired-
bootstrap **tests** (plain-vs-traced per size, then 2B-vs-0.8B among the traced), and a gallery
— each frame with both sizes' predicted paths over the ground-truth corridor, next to **the
`<think>` reasoning each size generated**, its waypoints, its per-sample metrics, and the plain
model's error on that same frame. Frames are grouped into easy/medium/hard terciles by the 2B
traced error. A new script because the `habitat_train_full_traced` tag is off the size-scaling
curve (`compare_evals.py` skips it) and no existing report surfaces the generated reasoning; it
reuses `rover_vlm.overlay` (path/goal drawing) and `rover_vlm.compare.paired_bootstrap`.

**Key result (job 88716): the reasoning trace *hurt* accuracy.** The honest baseline is plain
SFT, not zero-shot base. Traced beats base hugely (2B median waypoint error 0.069 vs 0.465), but
loses to plain SFT on the same eval split — median waypoint error plain **0.044** vs traced
0.069 (2B), plain **0.046** vs 0.082 (0.8B); goal-visibility accuracy plain 0.867 vs traced
0.768 (2B). Paired bootstrap on mean waypoint error confirms it: plain significantly better for
both sizes (2B diff −0.024, CI [−0.032, −0.016]; 0.8B diff −0.027, CI [−0.036, −0.019]), and on
Fréchet and per-waypoint visibility too. Not a data-size artifact (7,768 traced vs 8,140 plain,
4.6% fewer, can't explain a ~25% error rise) — the likely cause is the inference-time reasoning
adding autoregressive drift on a pure-geometry output. Among the traced adapters, 2B still edges
0.8B (diff −0.009, CI excludes 0) — the sizes converge, echoing the plain-regression rounds.

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
| Cross-round: what more data bought | `uv run scripts/visualize_crossround.py` | `outputs/crossround_report.html` |
| Trace prompt: how v6 was arrived at | `uv run scripts/visualize_traces.py` | `outputs/traces_report.html` |
| Traced fine-tune: 2B vs 0.8B + reasoning | `uv run scripts/visualize_traced_comparison.py` | `outputs/eval_habitat_v2_traced/traced_comparison.html` |

Published artifacts (private to the owner; republish the same file path to update in
place, or pass the URL as `url=` from another session):

Round 1 (3,660 train / 500 eval):

- ShareRobot explorer — https://claude.ai/code/artifact/71daf6f4-76c8-47bf-8825-540d32347f9b
- Habitat path + visibility — https://claude.ai/code/artifact/4fdd1859-b6d9-4e14-b20d-5b41b24a9574
- Habitat classification — https://claude.ai/code/artifact/a5c5b899-1148-4f41-8c7a-4325558d39cf
- 2B vs 0.8B comparison — https://claude.ai/code/artifact/1195c008-dc27-482c-953f-4b017965f89e

Round 2 (8,140 train / 1,000 eval) — separate URLs, since neither round supersedes the
other (different eval sets):

- 2B vs 0.8B comparison — https://claude.ai/code/artifact/10164bb8-b09e-4e19-a847-134020f6c716
- Path + visibility, 2B — https://claude.ai/code/artifact/2e3046cb-d244-487c-8900-ae61fde31aca
- Path + visibility, 0.8B — https://claude.ai/code/artifact/d7f0d069-d064-47d8-bdd3-00fb14ba7933
- Classification, 2B — https://claude.ai/code/artifact/830c3d90-3453-4c19-98f4-3f7d0acba12c
- Classification, 0.8B — https://claude.ai/code/artifact/9e3fbf7f-9d9c-4825-a2df-81635ccb4538
- **Cross-round — what 2.2× the data bought** — https://claude.ai/code/artifact/ebba6298-4f40-4e7b-a65c-88efa8533b78

Traced fine-tune (reasoning distilled into the path, job 88716):

- **Traced 2B vs 0.8B + per-sample reasoning** — https://claude.ai/code/artifact/97d7146c-1a7d-46e7-8427-7acb9fd7c4ec

Reasoning-trace phase (labelling runs, not model evals — no round applies):

- **How the trace prompt converged, v1 → v6** — https://claude.ai/code/artifact/c1c8c625-5140-453f-b398-48565db9de25

The generators build each page's `<title>` from the eval tree's own `model_id` and split
sizes, so an artifact names the model and round it actually came from rather than
inheriting a title from whichever run was published first.

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
