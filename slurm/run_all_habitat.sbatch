#!/bin/bash
# Full Habitat path+visibility experiment in one job: train all subset sizes,
# evaluate base + every adapter on the fixed eval split, build the comparison
# table/plot (with visibility metrics).
#
#   sbatch slurm/run_all_habitat.sbatch          # Qwen3.5-2B (default)
#   sbatch slurm/run_all_habitat.sbatch 0.8b     # Qwen3.5-0.8B -> *_0.8b trees
#
# The model argument suffixes the output trees, so each base model's results are
# self-contained and no run can overwrite another's. Eval tags stay identical
# across models, which is what lets compare_evals.py --compare-dir overlay them.
#
# Idempotent: completed stages are skipped (adapter / metrics.json already
# present), so if the job dies or hits the time limit, just resubmit it and it
# continues where it left off. Expected total: ~2-3 h on a 3g.40gb slice
# (train_full is 3,660 samples).
#
# Prereq: run `uv run scripts/prepare_habitat.py` on the login node first so
# data/prepared_habitat/ exists.
#
#SBATCH --job-name=qwen-habitat-all
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

DATA=data/prepared_habitat
EVAL_DIR=outputs/eval_habitat$SUFFIX      # separate from the ShareRobot outputs/eval tree
RUNS_DIR=outputs/runs$SUFFIX
SIZES=(train_500 train_1000 train_2000 train_full)

banner() { echo; echo "==================== $* ===================="; date; }

banner "model=$MODEL_ID runs=$RUNS_DIR evals=$EVAL_DIR"

if [ ! -f "$DATA/eval.json" ]; then
    echo "ERROR: $DATA/eval.json not found — run 'uv run scripts/prepare_habitat.py' first" >&2
    exit 1
fi

# ---- 1. training runs (smallest first: earliest signal) ----
for name in "${SIZES[@]}"; do
    run_dir="$RUNS_DIR/habitat_${name}"
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

# ---- 2. evals: base model, then every adapter (habitat task mode) ----
if [ -f "$EVAL_DIR/habitat_base/metrics.json" ]; then
    banner "eval habitat_base — already done, skipping"
else
    banner "eval habitat_base (zero-shot)"
    $UV run scripts/evaluate.py --task habitat --tag habitat_base \
        --model-id "$MODEL_ID" \
        --eval-file "$DATA/eval.json" --out-dir "$EVAL_DIR"
fi

for name in "${SIZES[@]}"; do
    tag="habitat_${name}"
    if [ -f "$EVAL_DIR/${tag}/metrics.json" ]; then
        banner "eval $tag — already done, skipping"
        continue
    fi
    banner "eval $tag"
    $UV run scripts/evaluate.py --task habitat --tag "$tag" \
        --model-id "$MODEL_ID" \
        --adapter "$RUNS_DIR/${tag}/adapter" \
        --eval-file "$DATA/eval.json" --out-dir "$EVAL_DIR"
done

# ---- 3. comparison table + scaling plot (visibility-aware) ----
banner "comparison"
$UV run scripts/compare_evals.py --eval-dir "$EVAL_DIR" --meta "$DATA/meta.json" \
    --label "$LABEL" \
    --title "Qwen3.5-$LABEL LoRA on Habitat path + visibility — performance vs. training set size"

banner "ALL DONE"
