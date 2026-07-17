# Project: Rover Path VLM — Qwen3.5-2B Fine-Tuning

## Long-term goal
Fine-tune Qwen3.5-2B (vision-language) to take a single forward-facing RGB image from a 
rover camera and output a traversable path as a sequence of waypoints, with a reasoning 
trace preceding the answer. Training data will eventually come from Facebook Habitat 
(RGB frame + A*-computed ground-truth path between sampled points) plus Cosmos Reason 3 
generated reasoning traces.

## Current phase: feasibility check on an open dataset
Before building the full Habitat + Cosmos Reason 3 pipeline, I want to confirm the 
fine-tuning mechanics work end-to-end, and understand how much data is actually needed, 
using an existing open dataset with the same input/output structure as my real task 
(single RGB image in, ordered list of image-coordinate waypoints out) but a different 
task (robot manipulation trajectory, not rover path planning).

**Dataset: `BAAI/ShareRobot`, `trajectory/` subset** (Hugging Face; note that 
`BAAI/RoboBrain-LoRA-Trajectory` is the LoRA *checkpoint* trained on it, not the data). 
6,870 samples; raw schema is `trajectory.json` with pixel-coordinate waypoints + 
image dims — our `scripts/prepare_data.py` normalizes to [0,1] (3 decimals) and renders 
the conversation format below (up to 10 points, longer trajectories subsampled):

```json
{
  "id": 0,
  "image": ["path/to/frame_0.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are a robot using the joint control. The task is \"reach for the cloth\". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
    },
    { "from": "gpt", "value": "[[x1,y1],[x2,y2],...]" }
  ]
}
```

Do NOT build the Habitat data generation or Cosmos Reason 3 trace generation yet. Focus 
only on this open dataset until the fine-tuning loop is confirmed working and I 
understand the data-scaling behavior.

## Training stack
- Starting with plain **`transformers` + `peft`** (LoRA) for the fine-tuning loop — not 
  Unsloth, for now. Keep Unsloth as a fallback if training turns out too slow or 
  memory-heavy on my hardware.
- Reference for data format and hyperparameters: Unsloth's official 
  `Qwen3_5_(2B)_Vision` Colab notebook 
  (https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(2B)_Vision.ipynb). 
  It confirms Unsloth has a dedicated 2B vision fine-tuning path, and its LoRA target 
  modules, image/conversation formatting, and default hyperparameters (learning rate, 
  batch size, grad accumulation) are a good starting point to mirror in a `transformers`/
  `peft` implementation.

## What I want to learn from this phase
1. **Does fine-tuning work at all**: can I load Qwen3.5-2B, apply LoRA, and fine-tune on 
   this dataset without errors, producing valid, well-formatted waypoint-list outputs.
2. **Base model vs. fine-tuned comparison**: evaluate the *base* (non-fine-tuned) 
   Qwen3.5-2B on this task first, to establish a zero-shot baseline (does it already 
   produce roughly-plausible coordinate lists, or is it unusable out of the box).
3. **Performance vs. dataset size**: fine-tune separately on multiple training subset 
   sizes (e.g. 500 / 1K / 2K / 5K / full dataset) and compare each against the base 
   model and against each other, on a fixed held-out eval split. I want a clear 
   scaling curve, not just a single fine-tuned checkpoint.
4. Metrics to track per dataset size: output format validity rate (parses into a 
   correct coordinate list), waypoint-position error vs. ground truth (e.g. mean/median 
   distance), and qualitative inspection of a few samples per size.

## Deliverables for this phase
- A data loading/formatting script for RoboBrain-LoRA-Trajectory, with a clean way to 
  slice fixed-size training subsets (same eval split held out across all runs).
- A LoRA fine-tuning script for Qwen3.5-2B using `transformers` + `peft`, parameterized 
  by training set size, so I can re-run it across the sizes above.
- An eval script that runs the same held-out set through: the base model, and each 
  fine-tuned checkpoint — producing a comparison table/plot (dataset size vs. metrics).
- Enough understanding of the training loop and results that I know what to change when 
  I scale up to real Habitat + Cosmos Reason 3 data.

## Technical preferences / constraints
- Language: Python.
- Coordinate format: normalized coordinates matching Qwen's grounding convention 
  (consistent with the dataset's own normalized [x, y] format).
- LoRA, not full fine-tune, for this phase — prioritize iteration speed across multiple 
  dataset sizes over squeezing out max performance.
- Hardware: no local GPU. SLURM cluster, `A100` partition, node `thor` (1× full A100 + 
  4× 3g.40gb MIG + 4× 1g.20gb MIG). Default target: **3g.40gb MIG slice**. Jobs are 
  launched manually by the user (`sbatch`) — Claude writes scripts + sbatch files and 
  the user pastes back logs. Login node and thor share NFS.
- Weights: `Qwen/Qwen3.5-2B` from Hugging Face (ungated, ~4GB), cached in 
  `~/.cache/huggingface` (NFS, visible from thor). All downloads happen on the login node.
- Environment: **uv** (Python 3.12, `pyproject.toml` + `uv.lock`, `.venv/` in repo on 
  NFS). `uv sync` to set up; `uv run scripts/<script>.py` to run.

## Repo hygiene
This project will grow over multiple phases (open-dataset test → Habitat pipeline → 
Cosmos Reason 3 traces → full training), so structure it to stay clean from the start 
rather than accumulating scattered scripts and notebooks:
- Clear top-level structure (e.g. `data/`, `scripts/`, `configs/`, `outputs/`, `notebooks/` 
  if any) — agree on this layout before writing the first script.
- Every script/module gets a short docstring or header comment stating what it does and 
  how to run it.
- A single `README.md` kept up to date with: setup instructions, how to run each phase 
  (data prep, training, eval), and where outputs land.
- No one-off throwaway scripts left in the repo root — either fold into the proper 
  module/CLI or delete once done.
- Config (dataset size, LoRA rank, learning rate, etc.) via config files or CLI args, 
  not hardcoded values scattered across scripts.
- Checkpoints, logs, and eval outputs go into a gitignored `outputs/` (or similar), not 
  mixed in with source code.
- Before adding a new file, check whether it belongs in an existing module rather than 
  spawning a new one — ask me if it's unclear where something should live.

## Compute workflow — no local GPU
I don't have a GPU in this environment. All GPU work (data-format checks that need the 
model, training runs, eval runs) happens on a SLURM cluster (srun/sbatch), which I launch 
manually — you don't have direct access to submit or monitor jobs.

Implications for how we work:
- Any code that needs a GPU (model loading, training, inference) should be written as a 
  standalone script I can run via `srun`/`sbatch`, not assumed to run interactively in 
  this session.
- Write SLURM batch scripts (`sbatch` files) alongside each GPU-needing script, with 
  reasonable defaults (partition, GPU count, time limit) that I can adjust.
- CPU-only steps (repo setup, non-GPU data inspection/preprocessing, writing configs, 
  parsing results/logs after a run) can be done directly in this session.
- After I run a script on the cluster, I'll paste back logs/output/errors for you to 
  read and act on — don't assume you can verify a GPU step yourself.
- Structure scripts so failures are easy to diagnose from logs alone (clear print/log 
  statements, no silent failures) since iteration means "edit → I resubmit → paste logs back."

## Out of scope for now
- Habitat scene generation and A* path computation
- Cosmos Reason 3 reasoning trace generation
- Building my own dataset
- Multi-GPU/distributed training setup
- Unsloth (kept as a fallback, not the starting stack)

These come later, once the open-dataset feasibility and scaling results are in hand.