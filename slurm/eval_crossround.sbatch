#!/bin/bash
# Score the ROUND-1 adapters on the cross-round eval subset, so a round-1 model and a v2
# model can be compared on identical frames.
#
#   sbatch slurm/eval_crossround.sbatch
#
# Only the round-1 adapters need GPU time. The v2 models were already evaluated on the
# full v2 eval split, and the subset is taken from those existing predictions — that is
# both free and more faithful than re-running, since a 553-frame run would batch
# differently from the 1,000-frame run that produced the published numbers.
#
# Prereq (login node): uv run scripts/prepare_crossround_eval.py
#
# What the subset is: the 553 frames of the v2 eval split that were absent from round 1's
# pool entirely. The other 447 are excluded — 392 were round-1 training data, and 55 are
# the matched-exposure group analysed separately from existing predictions.
#
# READ THE RESULT CAREFULLY: on these frames the round-1 model has 0% scene exposure and
# the v2 model ~99% (the generator adds whole new scenes). A v2 win therefore measures
# more data *plus* the new-room coverage that came with it — it is not a like-for-like
# generalization test. See README, "What did the extra data actually buy?".
#
# ~45-60 min total on a 3g.40gb slice (4 evals x 553 frames).
#
#SBATCH --job-name=qwen-crossround-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

export HF_HUB_OFFLINE=1
UV=~/.local/bin/uv

REG_DATA=data/prepared_habitat_crossround
CHOICE_DATA=data/prepared_habitat_choice_crossround

banner() { echo; echo "==================== $* ===================="; date; }

for f in "$REG_DATA/eval.json" "$CHOICE_DATA/eval.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found — run 'uv run scripts/prepare_crossround_eval.py' first" >&2
        exit 1
    fi
done

# task | model id | round-1 adapter | eval file | out dir | tag
RUNS=(
  "habitat|Qwen/Qwen3.5-2B|outputs/runs/habitat_train_full/adapter|$REG_DATA|outputs/eval_crossround|r1_2b_reg"
  "habitat|Qwen/Qwen3.5-0.8B|outputs/runs_0.8b/habitat_train_full/adapter|$REG_DATA|outputs/eval_crossround|r1_0.8b_reg"
  "choice|Qwen/Qwen3.5-2B|outputs/runs/habitat_choice_train_full/adapter|$CHOICE_DATA|outputs/eval_crossround|r1_2b_choice"
  "choice|Qwen/Qwen3.5-0.8B|outputs/runs_0.8b/habitat_choice_train_full/adapter|$CHOICE_DATA|outputs/eval_crossround|r1_0.8b_choice"
)

for spec in "${RUNS[@]}"; do
    IFS='|' read -r task model adapter data out tag <<< "$spec"
    if [ -f "$out/$tag/metrics.json" ]; then
        banner "$tag — already done, skipping"
        continue
    fi
    banner "$tag  ($model on $(basename "$data"))"
    $UV run scripts/evaluate.py --task "$task" --tag "$tag" \
        --model-id "$model" --adapter "$adapter" \
        --eval-file "$data/eval.json" --out-dir "$out"
done

banner "ALL DONE — now run: uv run scripts/compare_crossround.py"
