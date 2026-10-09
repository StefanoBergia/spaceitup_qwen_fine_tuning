#!/bin/bash
# Score the 2B plain and 2B traced adapters on a benchmark-clip eval set, in one job.
#   sbatch slurm/eval_clips.sbatch real_clips
# The traced stage runs with --enable-thinking, so predictions.json keeps each frame's
# <think> reasoning in its `generated` field -- that is what
# scripts/render_clip_predictions.py burns into the video and writes to traces.jsonl.
#
# Idempotent: a stage whose metrics.json already exists is skipped, so resubmit freely.
# Sized from the measured rates on gnd_campus (502 samples): 1.07 s/sample plain,
# 1.75 s/sample traced -- ~70 min for a 1,450-frame clip set, so 3 h is roomy.
#
# Output: outputs/eval_real/<SET>/{plain_2b,traced_2b}/{predictions,metrics}.json
# Then, on the login node:
#   uv run scripts/render_clip_predictions.py --set <SET>
#SBATCH --job-name=qwen-clip-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

SET="${1:-real_clips}"
EVAL_FILE="data/prepared_real/$SET/eval.json"
OUT_DIR="outputs/eval_real/$SET"
[ -f "$EVAL_FILE" ] || { echo "missing $EVAL_FILE (run scripts/find_clips.py)"; exit 1; }
echo "=== $SET: $($UV run python -c "import json,sys;print(len(json.load(open('$EVAL_FILE'))))") samples"

# tag | model id | adapter dir | thinking (0/1)
STAGES=(
  "plain_2b|Qwen/Qwen3.5-2B|outputs/runs_v2/habitat_train_full/adapter|0"
  "traced_2b|Qwen/Qwen3.5-2B|outputs/runs_v2_traced/habitat_train_full_traced/adapter|1"
)

for stage in "${STAGES[@]}"; do
  IFS='|' read -r TAG MODEL ADAPTER THINK <<<"$stage"
  echo "=== [$SET] $TAG  ($(date))"
  if [ -f "$OUT_DIR/$TAG/metrics.json" ]; then echo "    done, skipping"; continue; fi
  [ -d "$ADAPTER" ] || { echo "    adapter missing: $ADAPTER -- skipping"; continue; }
  ARGS=(--task habitat --tag "$TAG" --model-id "$MODEL" --eval-file "$EVAL_FILE"
        --out-dir "$OUT_DIR" --adapter "$ADAPTER")
  [ "$THINK" = "1" ] && ARGS+=(--enable-thinking --max-new-tokens 384)
  $UV run scripts/evaluate.py "${ARGS[@]}"
done
echo "=== all stages finished ($(date))"
