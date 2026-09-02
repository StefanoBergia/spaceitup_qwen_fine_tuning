#!/bin/bash
# Sampling-consistency eval: draw N sampled answers per eval image (one seed each) for the
# plain / traced adapters at both sizes, so rover_vlm.consistency can measure how much the
# output moves for the same image. Idempotent — a seed whose metrics.json exists is skipped,
# so resubmit to continue after a failure or time limit.
#
#   sbatch slurm/eval_seeds.sbatch                       # all four configs, one job (~10 h at 5 seeds)
#   sbatch slurm/eval_seeds.sbatch traced2b              # one config (~2.5 h) — launch four in parallel
#   sbatch slurm/eval_seeds.sbatch plain2b traced2b      # any subset, in the given order
#   SEEDS="1 2 3" TEMPERATURE=0.7 sbatch slurm/eval_seeds.sbatch
#
# Then on the login node: uv run scripts/visualize_traced_comparison.py  (consistency section)
#SBATCH --job-name=qwen-eval-seeds
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

ALL_CONFIGS="plain2b plain0.8b traced2b traced0.8b"
CONFIGS="${*:-$ALL_CONFIGS}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
EVAL_FILE=data/prepared_habitat_v2/eval.json   # the split every v2 / traced run was scored on

banner() { echo; echo "=================== $* ($(date +%H:%M:%S)) ==================="; }

# One config: model id, adapter, eval tree, tag, extra flags — mirroring the dirs written by
# slurm/run_all_habitat.sbatch (plain, v2) and slurm/train_traced_both.sbatch (traced).
run_config() {
    local CONFIG="$1" MODEL ADAPTER EVAL_DIR TAG EXTRA
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
      *) echo "unknown config: $CONFIG (expected one of: $ALL_CONFIGS)" >&2; exit 2 ;;
    esac
    [ -d "$ADAPTER" ] || { echo "adapter missing: $ADAPTER" >&2; exit 1; }

    banner "[$CONFIG] START — seeds [$SEEDS]"
    for SEED in $SEEDS; do
        local OUT="$EVAL_DIR/$TAG/seeds/seed$SEED"
        if [ -f "$OUT/metrics.json" ]; then
            banner "[$CONFIG] seed $SEED — metrics exist, skipping"
            continue
        fi
        banner "[$CONFIG] seed $SEED -> $OUT"
        $UV run scripts/evaluate.py --task habitat \
            --model-id "$MODEL" --adapter "$ADAPTER" --tag "$TAG" \
            --eval-file "$EVAL_FILE" --out-dir "$EVAL_DIR" \
            --seed "$SEED" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
            $EXTRA
    done
    banner "[$CONFIG] DONE"
}

[ -f "$EVAL_FILE" ] || { echo "eval split missing: $EVAL_FILE" >&2; exit 1; }
banner "START — configs [$CONFIGS]  T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
for c in $CONFIGS; do
    run_config "$c"
done
banner "ALL CONFIGS DONE"
