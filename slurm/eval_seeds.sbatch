#!/bin/bash
# Sampling-consistency eval: draw N sampled answers per eval image (one seed each) for ONE
# model configuration, so rover_vlm.consistency can measure how much the output moves for
# the same image. Idempotent — a seed whose metrics.json exists is skipped, so resubmit to
# continue. Launch the four configs in parallel on four 3g.40gb slices (~2.5 h each at 5 seeds):
#   for c in plain2b plain0.8b traced2b traced0.8b; do sbatch slurm/eval_seeds.sbatch $c; done
# Override the seed list / decoding with env vars:
#   SEEDS="1 2 3" TEMPERATURE=0.7 sbatch slurm/eval_seeds.sbatch traced2b
# Then on the login node: uv run scripts/visualize_traced_comparison.py  (consistency section)
#SBATCH --job-name=qwen-eval-seeds
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

CONFIG="${1:?usage: sbatch slurm/eval_seeds.sbatch <plain2b|plain0.8b|traced2b|traced0.8b>}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
EVAL_FILE=data/prepared_habitat_v2/eval.json   # the split every v2 / traced run was scored on

# config -> model id, adapter, eval tree, tag, extra flags. Mirrors the dirs written by
# slurm/run_all_habitat.sbatch (plain, v2) and slurm/train_traced_both.sbatch (traced).
case "$CONFIG" in
  plain2b)    MODEL=Qwen/Qwen3.5-2B;   ADAPTER=outputs/runs_v2/habitat_train_full/adapter
              EVAL_DIR=outputs/eval_habitat_v2;             TAG=habitat_train_full;        EXTRA="" ;;
  plain0.8b)  MODEL=Qwen/Qwen3.5-0.8B; ADAPTER=outputs/runs_v2_0.8b/habitat_train_full/adapter
              EVAL_DIR=outputs/eval_habitat_v2_0.8b;        TAG=habitat_train_full;        EXTRA="" ;;
  traced2b)   MODEL=Qwen/Qwen3.5-2B;   ADAPTER=outputs/runs_v2_traced/habitat_train_full_traced/adapter
              EVAL_DIR=outputs/eval_habitat_v2_traced;      TAG=habitat_train_full_traced
              EXTRA="--enable-thinking --max-new-tokens 384" ;;
  traced0.8b) MODEL=Qwen/Qwen3.5-0.8B; ADAPTER=outputs/runs_v2_traced_0.8b/habitat_train_full_traced/adapter
              EVAL_DIR=outputs/eval_habitat_v2_traced_0.8b; TAG=habitat_train_full_traced
              EXTRA="--enable-thinking --max-new-tokens 384" ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

banner() { echo; echo "=================== $* ($(date +%H:%M:%S)) ==================="; }

[ -d "$ADAPTER" ] || { echo "adapter missing: $ADAPTER" >&2; exit 1; }
[ -f "$EVAL_FILE" ] || { echo "eval split missing: $EVAL_FILE" >&2; exit 1; }
banner "START $CONFIG — seeds [$SEEDS] T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"

for SEED in $SEEDS; do
    OUT="$EVAL_DIR/$TAG/seeds/seed$SEED"
    if [ -f "$OUT/metrics.json" ]; then
        banner "seed $SEED — metrics exist, skipping"
        continue
    fi
    banner "seed $SEED -> $OUT"
    $UV run scripts/evaluate.py --task habitat \
        --model-id "$MODEL" --adapter "$ADAPTER" --tag "$TAG" \
        --eval-file "$EVAL_FILE" --out-dir "$EVAL_DIR" \
        --seed "$SEED" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
        $EXTRA
done
banner "ALL SEEDS DONE for $CONFIG"
