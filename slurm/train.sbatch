#!/bin/bash
# LoRA fine-tuning run on a 3g.40gb MIG slice.
#
#   sbatch slurm/train.sbatch [train-file] [output-dir] [extra train.py args...]
#
# Defaults: train-file = data/prepared/train_500.json,
#           output-dir = outputs/runs/lora_<train-file-stem>
#   e.g. sbatch slurm/train.sbatch                                    # -> outputs/runs/lora_train_500
#        sbatch slurm/train.sbatch data/prepared/train_5000.json     # -> outputs/runs/lora_train_5000
#        sbatch slurm/train.sbatch data/prepared/train_full.json outputs/runs/my_run --epochs 3
#
#SBATCH --job-name=qwen-lora
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

TRAIN_FILE="${1:-data/prepared/train_500.json}"
stem="$(basename "$TRAIN_FILE" .json)"
OUTPUT_DIR="${2:-outputs/runs/lora_${stem}}"
shift $(( $# >= 2 ? 2 : $# ))

echo "train file: $TRAIN_FILE"
echo "output dir: $OUTPUT_DIR"

#nvidia-smi | head -12
export HF_HUB_OFFLINE=1

~/.local/bin/uv run scripts/train.py \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
