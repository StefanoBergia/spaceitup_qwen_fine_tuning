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

Train (GPU) — same `scripts/train.py`, just pointed at the habitat splits:

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

> **Note:** the interactive `scripts/visualize_predictions.py` explorer is ShareRobot-only
> — it expects 2-element `[x,y]` predictions and does not yet understand the habitat
> `{"path","goal"}` format. For qualitative habitat inspection use
> `scripts/inspect_habitat.py`; a habitat explorer is a deferred follow-up.
