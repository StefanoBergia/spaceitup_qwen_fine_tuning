#!/bin/bash
# Evaluate base or base+adapter on a real-image eval set (data/prepared_real/<SET>/eval.json).
#   sbatch slurm/eval_real.sbatch tum_pioneer base_2b
#   sbatch slurm/eval_real.sbatch tum_pioneer plain_2b outputs/runs_v2/habitat_train_full/adapter
#   THINK=1 sbatch slurm/eval_real.sbatch tum_pioneer traced_2b outputs/runs_v2_traced/habitat_train_full_traced/adapter
# Env: MODEL_ID (default Qwen/Qwen3.5-2B), THINK=1 for trace-trained adapters.
# Output: outputs/eval_real/<SET>/<TAG>/{predictions,metrics}.json
#SBATCH --job-name=qwen-real-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

SET="${1:?usage: eval_real.sbatch <SET> <TAG> [ADAPTER]}"
TAG="${2:?usage: eval_real.sbatch <SET> <TAG> [ADAPTER]}"
ADAPTER="${3:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
EVAL_FILE="data/prepared_real/$SET/eval.json"
OUT_DIR="outputs/eval_real/$SET"
[ -f "$EVAL_FILE" ] || { echo "missing $EVAL_FILE (run scripts/prepare_real_eval.py)"; exit 1; }

ARGS=(--task habitat --tag "$TAG" --model-id "$MODEL_ID" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR")
[ -n "$ADAPTER" ] && ARGS+=(--adapter "$ADAPTER")
[ "${THINK:-0}" = "1" ] && ARGS+=(--enable-thinking --max-new-tokens 384)
echo "evaluate.py ${ARGS[*]}"
$UV run scripts/evaluate.py "${ARGS[@]}"
