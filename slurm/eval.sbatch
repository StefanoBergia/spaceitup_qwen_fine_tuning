#!/bin/bash
# Evaluate the base model and/or fine-tuned adapters on the fixed eval split.
#
#   sbatch slurm/eval.sbatch                # base model only (tag: base)
#   sbatch slurm/eval.sbatch all            # base + every adapter under outputs/runs/*/adapter
#   sbatch slurm/eval.sbatch outputs/runs/lora_train_500/adapter   # one adapter
#
#SBATCH --job-name=qwen-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

export HF_HUB_OFFLINE=1

run_eval() {  # run_eval <tag> [adapter]
    local tag="$1" adapter="${2:-}"
    if [ -n "$adapter" ]; then
        echo "== eval $tag ($adapter) =="
        ~/.local/bin/uv run scripts/evaluate.py --tag "$tag" --adapter "$adapter"
    else
        echo "== eval $tag (base model) =="
        ~/.local/bin/uv run scripts/evaluate.py --tag "$tag"
    fi
}

TARGET="${1:-base}"
case "$TARGET" in
    base)
        run_eval base
        ;;
    all)
        run_eval base
        for adapter in outputs/runs/*/adapter; do
            [ -d "$adapter" ] || continue
            run_name="$(basename "$(dirname "$adapter")")"
            run_eval "$run_name" "$adapter"
        done
        ;;
    *)
        run_name="$(basename "$(dirname "$TARGET")")"
        run_eval "$run_name" "$TARGET"
        ;;
esac
