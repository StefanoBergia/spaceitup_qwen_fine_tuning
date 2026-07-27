#!/bin/bash
# Fine-tune BOTH Qwen3.5 sizes on the trace-augmented Habitat path set, in ONE job.
#
#   2B   × path+visibility regression WITH reasoning traces (<think> block)
#   0.8B × path+visibility regression WITH reasoning traces (<think> block)
#
#   sbatch slurm/train_traced_both.sbatch
#
# Runs the two sizes sequentially on a single 3g.40gb MIG slice (2B first — the longer
# run — so a job that dies early still produced the harder result). For each size it
# trains on data/prepared_habitat_v2/train_full_traced.json (reasoning rendered inside
# Qwen's native <think> by the collator) then evals the base model and the trained
# adapter on the PLAIN eval split, both with --enable-thinking so the model actually
# reasons before answering (the base template defaults thinking OFF).
#
# Output trees carry a _traced marker AND a per-model suffix, so nothing here can
# overwrite the earlier plain-regression runs (outputs/runs_v2*, outputs/eval_habitat_v2*):
#   2B   -> outputs/runs_v2_traced/       outputs/eval_habitat_v2_traced/
#   0.8B -> outputs/runs_v2_traced_0.8b/  outputs/eval_habitat_v2_traced_0.8b/
# Eval tags are identical across the two models (habitat_base, habitat_train_full_traced),
# which is what lets a later cross-model overlay line them up.
#
# Idempotent and resumable: any stage whose adapter/ or metrics.json already exists is
# skipped, so if the job times out or dies, just resubmit and it continues.
#
# Prereq (login node, CPU): build the traced training file first —
#   uv run scripts/prepare_habitat_traces.py
# and make sure the image paths resolve (scripts/repoint_habitat_paths.py --apply if the
# Habitat source tree has moved).
#
# Rough estimate scaled from the Jul 20 v2 logs: 2B train_full ~2-3 h + 0.8B ~1-1.5 h,
# plus four short evals -> set a comfortable ceiling.
#
#SBATCH --job-name=qwen-habitat-traced-both
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

TRAIN_FILE="data/prepared_habitat_v2/train_full_traced.json"
EVAL_FILE="data/prepared_habitat_v2/eval.json"
# Thinking eval needs room for ~100 words of trace on top of the answer.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"

banner() { echo; echo "==================== $* ===================="; date; }

for f in "$TRAIN_FILE" "$EVAL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found — run 'uv run scripts/prepare_habitat_traces.py' first" >&2
        exit 1
    fi
done

# Train + eval one model size end to end. $1=model id  $2=output suffix  $3=label
run_model() {
    local id="$1" suffix="$2" label="$3"
    local run_dir="outputs/runs_v2_traced${suffix}/habitat_train_full_traced"
    local eval_dir="outputs/eval_habitat_v2_traced${suffix}"

    banner "[$label] TRAIN on traced data"
    if [ -d "$run_dir/adapter" ]; then
        banner "[$label] train — adapter exists, skipping"
    else
        $UV run scripts/train.py \
            --model-id "$id" \
            --train-file "$TRAIN_FILE" \
            --output-dir "$run_dir"
    fi

    banner "[$label] EVAL base (zero-shot, thinking on)"
    if [ -f "$eval_dir/habitat_base/metrics.json" ]; then
        banner "[$label] eval base — metrics exist, skipping"
    else
        $UV run scripts/evaluate.py --task habitat --enable-thinking \
            --model-id "$id" --tag habitat_base \
            --eval-file "$EVAL_FILE" --out-dir "$eval_dir" \
            --max-new-tokens "$MAX_NEW_TOKENS"
    fi

    banner "[$label] EVAL traced adapter"
    if [ -f "$eval_dir/habitat_train_full_traced/metrics.json" ]; then
        banner "[$label] eval adapter — metrics exist, skipping"
    else
        $UV run scripts/evaluate.py --task habitat --enable-thinking \
            --model-id "$id" --tag habitat_train_full_traced \
            --adapter "$run_dir/adapter" \
            --eval-file "$EVAL_FILE" --out-dir "$eval_dir" \
            --max-new-tokens "$MAX_NEW_TOKENS"
    fi
}

banner "START — 2B then 0.8B, traced path regression"
run_model Qwen/Qwen3.5-2B   ""      2B
run_model Qwen/Qwen3.5-0.8B "_0.8b" 0.8B
banner "BOTH SIZES DONE"
