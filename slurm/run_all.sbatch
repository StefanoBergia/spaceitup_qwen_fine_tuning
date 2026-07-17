#!/bin/bash
# Full scaling experiment in one job: train all subset sizes, evaluate base +
# every adapter on the fixed eval split, build the comparison table/plot.
#
#   sbatch slurm/run_all.sbatch
#
# Idempotent: completed stages are skipped (adapter / metrics.json already
# present), so if the job dies or hits the time limit, just resubmit it and
# it continues where it left off. Expected total: ~4-6 h on a 3g.40gb slice.
#
#SBATCH --job-name=qwen-lora-all
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

SIZES=(train_500 train_1000 train_2000 train_5000 train_full)

banner() { echo; echo "==================== $* ===================="; date; }

# ---- 1. training runs (smallest first: earliest signal) ----
for name in "${SIZES[@]}"; do
    run_dir="outputs/runs/lora_${name}"
    if [ -d "$run_dir/adapter" ]; then
        banner "train $name — already done, skipping"
        continue
    fi
    banner "train $name"
    $UV run scripts/train.py \
        --train-file "data/prepared/${name}.json" \
        --output-dir "$run_dir"
done

# ---- 2. evals: base model, then every adapter ----
if [ -f outputs/eval/base/metrics.json ]; then
    banner "eval base — already done, skipping"
else
    banner "eval base (zero-shot)"
    $UV run scripts/evaluate.py --tag base
fi

for name in "${SIZES[@]}"; do
    tag="lora_${name}"
    if [ -f "outputs/eval/${tag}/metrics.json" ]; then
        banner "eval $tag — already done, skipping"
        continue
    fi
    banner "eval $tag"
    $UV run scripts/evaluate.py --tag "$tag" --adapter "outputs/runs/${tag}/adapter"
done

# ---- 3. comparison table + scaling plot ----
banner "comparison"
$UV run scripts/compare_evals.py

banner "ALL DONE"
