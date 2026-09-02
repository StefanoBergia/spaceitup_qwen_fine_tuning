#!/bin/bash
# Score every Habitat-path model on one real-image eval set, in one job:
#   base 2B, plain 2B/0.8B (outputs/runs_v2*), traced 2B/0.8B (outputs/runs_v2_traced*).
#   sbatch slurm/run_all_real.sbatch tum_pioneer
#   sbatch slurm/run_all_real.sbatch gnd_gmu
# Idempotent: a stage whose metrics.json exists is skipped, so resubmit after a timeout.
# Output: outputs/eval_real/<SET>/<tag>/ ; then scripts/visualize_real_eval.py.
#SBATCH --job-name=qwen-real-all
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

SET="${1:?usage: run_all_real.sbatch <SET>}"
EVAL_FILE="data/prepared_real/$SET/eval.json"
OUT_DIR="outputs/eval_real/$SET"
[ -f "$EVAL_FILE" ] || { echo "missing $EVAL_FILE (run scripts/prepare_real_eval.py)"; exit 1; }

# tag | model id | adapter dir ("" = base) | thinking (0/1)
STAGES=(
  "base_2b|Qwen/Qwen3.5-2B||0"
  "plain_2b|Qwen/Qwen3.5-2B|outputs/runs_v2/habitat_train_full/adapter|0"
  "plain_0.8b|Qwen/Qwen3.5-0.8B|outputs/runs_v2_0.8b/habitat_train_full/adapter|0"
  "traced_2b|Qwen/Qwen3.5-2B|outputs/runs_v2_traced/habitat_train_full_traced/adapter|1"
  "traced_0.8b|Qwen/Qwen3.5-0.8B|outputs/runs_v2_traced_0.8b/habitat_train_full_traced/adapter|1"
)

for stage in "${STAGES[@]}"; do
  IFS='|' read -r TAG MODEL ADAPTER THINK <<<"$stage"
  echo "=== [$SET] $TAG  ($(date))"
  if [ -f "$OUT_DIR/$TAG/metrics.json" ]; then echo "    done, skipping"; continue; fi
  if [ -n "$ADAPTER" ] && [ ! -d "$ADAPTER" ]; then echo "    adapter missing: $ADAPTER -- skipping"; continue; fi
  ARGS=(--task habitat --tag "$TAG" --model-id "$MODEL" --eval-file "$EVAL_FILE" --out-dir "$OUT_DIR")
  [ -n "$ADAPTER" ] && ARGS+=(--adapter "$ADAPTER")
  [ "$THINK" = "1" ] && ARGS+=(--enable-thinking --max-new-tokens 384)
  $UV run scripts/evaluate.py "${ARGS[@]}"
done
echo "=== all stages finished ($(date))"
