#!/bin/bash
# d1 eval cells, skipping finished ones (safe to re-submit):
#     sbatch slurm/eval_descriptions.sbatch                         # adapter outputs/runs_desc/d1/adapter
#     ADAPTER=<dir> TAG=<tag> sbatch slurm/eval_descriptions.sbatch
#     SETS=steer sbatch slurm/eval_descriptions.sbatch              # the 4 steering stop-pair sets (d1)
#     SETS=steer_goal ADAPTER=outputs/runs_v4/habitat_train_full/adapter TAG=r4 sbatch slurm/eval_descriptions.sbatch
#     DATA=data/prepared_habitat_instr_v4 OUT_DIR=outputs/eval_desc_d2 ADAPTER=outputs/runs_desc/d2/adapter TAG=d2 sbatch slurm/eval_descriptions.sbatch
# metrics.json gains a "description" block. Then, on the login node:
#     uv run scripts/render_desc_videos.py --set eval_desc_pairs --tag d1
#SBATCH --job-name=qwen-desc-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
D="${DATA:-data/prepared_habitat_instr_v3}"
OUT="${OUT_DIR:-outputs/eval_desc}"
TAG="${TAG:-d1}"
ADAPTER="${ADAPTER:-outputs/runs_desc/d1/adapter}"
run() {  # run <set>
    local set=$1
    if [ -f "$OUT/$set/$TAG/metrics.json" ]; then echo "skip $TAG/$set (done)"; return; fi
    if [ ! -f "$D/$set.json" ]; then echo "SKIP $TAG/$set: no $D/$set.json"; return; fi
    if [ ! -d "$ADAPTER" ]; then echo "SKIP $TAG/$set: no adapter at $ADAPTER"; return; fi
    echo "=== $TAG on $set ==="
    $UV run scripts/evaluate.py --task habitat --tag "$TAG" --eval-file "$D/$set.json" \
        --out-dir "$OUT/$set" --adapter "$ADAPTER"
}
SETS="${SETS:-desc}"
STEER_GOAL="eval_steer_stop_template_goal eval_steer_stop_composed_goal"
case "$SETS" in
    desc)       LIST="eval_desc_entry eval_desc_goal eval_desc_pairs eval_desc_shuffled" ;;
    steer)      LIST="eval_steer_stop_template_entry eval_steer_stop_composed_entry $STEER_GOAL" ;;
    steer_goal) LIST="$STEER_GOAL" ;;
    *) echo "SETS must be desc, steer or steer_goal, got '$SETS'"; exit 1 ;;
esac
for s in $LIST; do run "$s"; done
echo "all cells done: uv run scripts/render_desc_videos.py --set <set> --tag $TAG; uv run scripts/visualize_steering.py"
