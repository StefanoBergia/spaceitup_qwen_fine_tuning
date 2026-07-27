#!/bin/bash
# All four Habitat experiments in ONE job, on the enlarged (~9.1K sample) dataset:
#
#   2B   × path+visibility regression      2B   × path classification
#   0.8B × path+visibility regression      0.8B × path classification
#
#   sbatch slurm/run_all_experiments.sbatch          # VERSION=_v2, train_full only
#   VERSION=_v3 SIZES="train_2000 train_full" sbatch slurm/run_all_experiments.sbatch
#
# This is a thin wrapper: it runs the two per-task scripts four times with VERSION/SIZES
# set, so all the train/eval/compare logic (and its per-stage idempotency) lives in one
# place. Their #SBATCH lines are inert comments when invoked as plain bash.
#
# Idempotent and resumable end to end: every stage is skipped when its adapter/ or
# metrics.json already exists, so if the job dies or times out mid-way, just resubmit.
#
# Prereq (login node, CPU) — build the v2 splits first:
#   uv run scripts/prepare_habitat.py --out-dir data/prepared_habitat_v2 \
#       --eval-size 1000 --train-sizes ""
#   mkdir -p data/prepared_habitat_choice_v2
#   ln -s ../prepared_habitat_choice/images data/prepared_habitat_choice_v2/images
#   uv run scripts/prepare_habitat_choice.py \
#       --src-splits data/prepared_habitat_v2 --out-dir data/prepared_habitat_choice_v2
#
# Estimate scaled from the Jul 17/20 logs: ~6-9 h serial on a 3g.40gb slice.
#
#SBATCH --job-name=qwen-habitat-v2-all
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

export VERSION="${VERSION:-_v2}"
export SIZES="${SIZES:-train_full}"
UV=~/.local/bin/uv

stamp() { echo; echo "############################################################"; \
          echo "### $*"; date; echo "############################################################"; }

stamp "START — VERSION=$VERSION SIZES=$SIZES"

# Longest experiment first, so a job that dies early still produced the hardest result.
stamp "1/4  regression · Qwen3.5-2B"
bash slurm/run_all_habitat.sbatch 2b

stamp "2/4  regression · Qwen3.5-0.8B"
bash slurm/run_all_habitat.sbatch 0.8b

stamp "3/4  classification · Qwen3.5-2B"
bash slurm/run_all_habitat_choice.sbatch 2b

stamp "4/4  classification · Qwen3.5-0.8B"
bash slurm/run_all_habitat_choice.sbatch 0.8b

# ---- cross-model overlay per task (2B vs 0.8B on identical eval tags) ----
# Written into the 0.8B tree; --compare-dir is read-only. Not gated on idempotency:
# it is cheap, CPU-only, and must reflect whatever the four runs above just produced.
stamp "overlay · regression 2B vs 0.8B"
$UV run scripts/compare_evals.py \
    --eval-dir "outputs/eval_habitat${VERSION}_0.8b" --label 0.8B \
    --compare-dir "outputs/eval_habitat${VERSION}" --compare-label 2B \
    --meta "data/prepared_habitat${VERSION}/meta.json" \
    --title "Qwen3.5 LoRA on Habitat path + visibility — 2B vs 0.8B"

stamp "overlay · classification 2B vs 0.8B"
$UV run scripts/compare_evals.py --task choice \
    --eval-dir "outputs/eval_habitat_choice${VERSION}_0.8b" --label 0.8B \
    --compare-dir "outputs/eval_habitat_choice${VERSION}" --compare-label 2B \
    --meta "data/prepared_habitat_choice${VERSION}/meta.json"

stamp "ALL FOUR EXPERIMENTS DONE"
