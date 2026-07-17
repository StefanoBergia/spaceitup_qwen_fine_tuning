#!/bin/bash
# Evaluate base or base+adapter on the Habitat eval split.
#   sbatch slurm/eval_habitat.sbatch base
#   sbatch slurm/eval_habitat.sbatch habitat_train_full outputs/runs/habitat_train_full/adapter
#SBATCH --job-name=qwen-habitat-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

TAG="${1:-habitat_base}"
ADAPTER="${2:-}"
EVAL_FILE=data/prepared_habitat/eval.json
OUT_DIR=outputs/eval_habitat   # separate dir so ShareRobot evals stay intact
if [ -n "$ADAPTER" ]; then
    $UV run scripts/evaluate.py --task habitat --tag "$TAG" --adapter "$ADAPTER" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR"
else
    $UV run scripts/evaluate.py --task habitat --tag "$TAG" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR"
fi
