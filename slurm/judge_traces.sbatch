#!/bin/bash
# VLM-as-judge scoring of the traced reasoning for BOTH sizes, in one job.
#
#   sbatch slurm/judge_traces.sbatch
#   MAX_SAMPLES=20 sbatch slurm/judge_traces.sbatch     # quick prompt-iteration pass
#
# Scores the student <think> reasoning against the image itself (reference-free) with a
# separate judge model (Cosmos-Reason2-8B, cached, transformers — no vLLM). Writes
# <eval-dir>/<tag>/judge.jsonl + judge_metrics.json alongside the predictions it judged.
# Resumable: rows already scored are skipped, so a timed-out job just resubmits.
#
# Cosmos-Reason2-8B is ~8B (fits a 3g.40gb slice in bf16). Set JUDGE=<hf-id> to try another
# judge (e.g. a non-Qwen model, to rule out self-preference toward the Qwen student).
#
#SBATCH --job-name=judge-traces
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
UV=~/.local/bin/uv

JUDGE="${JUDGE:-nvidia/Cosmos-Reason2-8B}"
EXTRA=""
[ -n "${MAX_SAMPLES:-}" ] && EXTRA="--max-samples $MAX_SAMPLES"

banner() { echo; echo "==================== $* ===================="; date; }

for dir in outputs/eval_habitat_v2_traced outputs/eval_habitat_v2_traced_0.8b; do
    if [ ! -f "$dir/habitat_train_full_traced/predictions.json" ]; then
        echo "ERROR: $dir/habitat_train_full_traced/predictions.json missing — run the traced evals first" >&2
        exit 1
    fi
    banner "judge $dir"
    $UV run scripts/judge_traces.py --eval-dir "$dir" \
        --tag habitat_train_full_traced --model-id "$JUDGE" $EXTRA
done

banner "BOTH JUDGED"
