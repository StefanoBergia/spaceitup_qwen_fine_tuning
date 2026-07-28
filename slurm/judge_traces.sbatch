#!/bin/bash
# LLM-as-judge scoring of reasoning traces AGAINST THE GROUND TRUTH, in one job.
#
#   sbatch slurm/judge_traces.sbatch
#   MAX_SAMPLES=20 sbatch slurm/judge_traces.sbatch          # quick prompt-iteration pass
#   TAGS="habitat_train_full_traced" sbatch slurm/judge_traces.sbatch   # skip the base check
#
# For BOTH sizes it judges the traced adapter AND the zero-shot base — the base run is the
# discrimination check: grading against known truth should score the base reasoning clearly
# below the traced one, otherwise the judge is not discriminating. Text-only (no image): the
# judge sees the ground-truth path direction + occlusion and the model's <think> reasoning,
# and grades agreement. Writes <eval-dir>/<tag>/judge.jsonl + judge_metrics.json, resumable.
#
# Cosmos-Reason2-8B (~8B, fits a 3g.40gb slice in bf16). Set JUDGE=<hf-id> for another judge
# (e.g. a non-Qwen model, to rule out self-preference toward the Qwen student).
#
#SBATCH --job-name=judge-traces
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
UV=~/.local/bin/uv

JUDGE="${JUDGE:-nvidia/Cosmos-Reason2-8B}"
read -r -a TAGS <<< "${TAGS:-habitat_train_full_traced habitat_base}"
EXTRA=""
[ -n "${MAX_SAMPLES:-}" ] && EXTRA="--max-samples $MAX_SAMPLES"

banner() { echo; echo "==================== $* ===================="; date; }

for dir in outputs/eval_habitat_v2_traced outputs/eval_habitat_v2_traced_0.8b; do
    for tag in "${TAGS[@]}"; do
        if [ ! -f "$dir/$tag/predictions.json" ]; then
            echo "ERROR: $dir/$tag/predictions.json missing — run the evals first" >&2
            exit 1
        fi
        banner "judge $dir  $tag"
        $UV run scripts/judge_traces.py --eval-dir "$dir" --tag "$tag" \
            --model-id "$JUDGE" $EXTRA
    done
done

banner "ALL JUDGED"
