#!/bin/bash
# Same as eval_real.sbatch but on the RTX partition (2080 Ti, 11 GB, fp16 -- evaluate.py
# falls back from bf16 automatically). Smaller batch to fit the card.
#   sbatch slurm/eval_real_rtx.sbatch <SET> <TAG> [ADAPTER]
#SBATCH --job-name=qwen-real-eval-rtx
#SBATCH --partition=RTX
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
SET="${1:?usage: eval_real_rtx.sbatch <SET> <TAG> [ADAPTER]}"
TAG="${2:?usage: eval_real_rtx.sbatch <SET> <TAG> [ADAPTER]}"
ADAPTER="${3:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
EVAL_FILE="data/prepared_real/$SET/eval.json"
OUT_DIR="outputs/eval_real/$SET"
[ -f "$EVAL_FILE" ] || { echo "missing $EVAL_FILE"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "nvidia-smi failed on at least one allocated card (continuing; torch will use whichever card answers)"
ARGS=(--task habitat --tag "$TAG" --model-id "$MODEL_ID" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR" --batch-size "${BATCH:-4}")
[ -n "$ADAPTER" ] && ARGS+=(--adapter "$ADAPTER")
[ "${THINK:-0}" = "1" ] && ARGS+=(--enable-thinking --max-new-tokens 384)
echo "evaluate.py ${ARGS[*]}"
$UV run scripts/evaluate.py "${ARGS[@]}"
