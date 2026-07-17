#!/bin/bash
# GPU smoke test: 50 samples, 20 steps on a 1g.20gb MIG slice (~5-10 min).
# Validates CUDA memory + gives real per-step timing before launching full runs.
#
#   sbatch slurm/train_smoke.sbatch
#
#SBATCH --job-name=qwen-lora-smoke
#SBATCH --partition=A100
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

#nvidia-smi | head -12

# Everything is pre-downloaded on NFS from the login node; never hit the network.
export HF_HUB_OFFLINE=1

~/.local/bin/uv run scripts/train.py \
    --train-file data/prepared/train_500.json \
    --max-samples 50 \
    --max-steps 20 \
    --batch-size 2 \
    --grad-accum 1 \
    --output-dir outputs/runs/gpu_smoke
