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
