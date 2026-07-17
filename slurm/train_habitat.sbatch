#!/bin/bash
# LoRA fine-tune on Habitat path+visibility data.
#   sbatch slurm/train_habitat.sbatch [train-file] [output-dir]
#SBATCH --job-name=qwen-habitat
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

TRAIN_FILE="${1:-data/prepared_habitat/train_full.json}"
stem=$(basename "$TRAIN_FILE" .json)
OUTPUT_DIR="${2:-outputs/runs/habitat_${stem}}"
shift $(( $# >= 2 ? 2 : $# )) || true

$UV run scripts/train.py --train-file "$TRAIN_FILE" --output-dir "$OUTPUT_DIR" "$@"
