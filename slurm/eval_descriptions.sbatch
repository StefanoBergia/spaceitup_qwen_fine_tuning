#!/bin/bash
# d1 eval cells, skipping finished ones (safe to re-submit):
#     sbatch slurm/eval_descriptions.sbatch                         # adapter outputs/runs_desc/d1/adapter
#     ADAPTER=<dir> TAG=<tag> sbatch slurm/eval_descriptions.sbatch
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
D=data/prepared_habitat_instr_v3
OUT=outputs/eval_desc
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
for s in eval_desc_entry eval_desc_goal eval_desc_pairs eval_desc_shuffled; do run "$s"; done
echo "all cells done: uv run scripts/render_desc_videos.py --set eval_desc_pairs --tag $TAG"
