#!/bin/bash
# Full Habitat path-CLASSIFICATION experiment in one job: train all subset sizes,
# evaluate base + every adapter on the fixed eval split, build the comparison
# table/plot (choice metrics).
#
#   sbatch slurm/run_all_habitat_choice.sbatch          # Qwen3.5-2B (default)
#   sbatch slurm/run_all_habitat_choice.sbatch 0.8b     # Qwen3.5-0.8B -> *_0.8b trees
#
# Two env vars pick the dataset version and which subset sizes to train, e.g. the
# enlarged (~9.1K sample) dataset with a single full-size run:
#
#   VERSION=_v2 SIZES=train_full sbatch slurm/run_all_habitat_choice.sbatch 2b
#
# The model argument suffixes the output trees, so each base model's results are
# self-contained and no run can overwrite another's. Eval tags stay identical
# across models, which is what lets compare_evals.py --compare-dir overlay them.
#
# Idempotent: completed stages are skipped (adapter / metrics.json already present),
# so if the job dies or hits the time limit, just resubmit and it continues where it
# left off. Expected total: ~2-3 h on a 3g.40gb slice.
#
# Prereq: run `uv run scripts/prepare_habitat_choice.py` on the login node first
# (which itself needs prepare_habitat.py, since split membership is mirrored from it).
#
#SBATCH --job-name=qwen-habitat-choice
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

# ---- model selection: unknown values are fatal, never a silent fallback to 2B,
#      which would quietly write into another model's (possibly complete) tree ----
case "${1:-2b}" in
    2b)   MODEL_ID=Qwen/Qwen3.5-2B;   SUFFIX="";      LABEL=2B ;;
    0.8b) MODEL_ID=Qwen/Qwen3.5-0.8B; SUFFIX="_0.8b"; LABEL=0.8B ;;
    *)    echo "ERROR: unknown model '$1' (expected: 2b, 0.8b)" >&2; exit 2 ;;
esac

# ---- dataset version: VERSION suffixes the data dir *and* both output trees together,
#      so a re-prep on more data can never land in the previous run's results ----
VERSION="${VERSION:-}"
DATA="data/prepared_habitat_choice$VERSION"
EVAL_DIR="outputs/eval_habitat_choice${VERSION}${SUFFIX}"  # separate from the regression eval trees
RUNS_DIR="outputs/runs${VERSION}${SUFFIX}"
read -r -a SIZES <<< "${SIZES:-train_500 train_1000 train_2000 train_full}"

banner() { echo; echo "==================== $* ===================="; date; }

banner "model=$MODEL_ID runs=$RUNS_DIR evals=$EVAL_DIR"

if [ ! -f "$DATA/eval.json" ]; then
    echo "ERROR: $DATA/eval.json not found — run 'uv run scripts/prepare_habitat_choice.py' first" >&2
    exit 1
fi

# ---- 1. training runs (smallest first: earliest signal) ----
for name in "${SIZES[@]}"; do
    run_dir="$RUNS_DIR/habitat_choice_${name}"
    if [ -d "$run_dir/adapter" ]; then
        banner "train $name — already done, skipping"
        continue
    fi
    banner "train $name"
    $UV run scripts/train.py \
        --model-id "$MODEL_ID" \
        --train-file "$DATA/${name}.json" \
        --output-dir "$run_dir"
done

# ---- 2. evals: base model, then every adapter (choice task mode) ----
if [ -f "$EVAL_DIR/habitat_choice_base/metrics.json" ]; then
    banner "eval habitat_choice_base — already done, skipping"
else
    banner "eval habitat_choice_base (zero-shot)"
    $UV run scripts/evaluate.py --task choice --tag habitat_choice_base \
        --model-id "$MODEL_ID" \
        --eval-file "$DATA/eval.json" --out-dir "$EVAL_DIR"
fi

for name in "${SIZES[@]}"; do
    tag="habitat_choice_${name}"
    if [ -f "$EVAL_DIR/${tag}/metrics.json" ]; then
        banner "eval $tag — already done, skipping"
        continue
    fi
    banner "eval $tag"
    $UV run scripts/evaluate.py --task choice --tag "$tag" \
        --model-id "$MODEL_ID" \
        --adapter "$RUNS_DIR/${tag}/adapter" \
        --eval-file "$DATA/eval.json" --out-dir "$EVAL_DIR"
done

# ---- 3. comparison table + scaling plot (choice metrics) ----
banner "comparison"
$UV run scripts/compare_evals.py --task choice --eval-dir "$EVAL_DIR" --meta "$DATA/meta.json" \
    --label "$LABEL"

banner "ALL DONE"
