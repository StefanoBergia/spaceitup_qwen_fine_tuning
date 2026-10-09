#!/bin/bash
# v2 eval cells, skipping finished ones (safe to re-submit):
#   general   on eval_plain / eval_instructed / eval_pairs / probe_wrong_side / probe_handwritten
#   v1_cosmos on eval_pairs (reference: the v1 Cosmos-trained adapter)
#     sbatch slurm/eval_instruction_v2.sbatch
# ~5,700 records at ~1.1 s each; metrics.json gains "instruction" (pairs) and "wrong_side"
# (probe) blocks where the records carry them.
#SBATCH --job-name=qwen-instr-eval-v2
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
D=data/prepared_habitat_instr_v2
OUT=outputs/eval_instr_v2
GEN=outputs/runs_instr_v2/general/adapter
run() {  # run <tag> <set> <adapter>
    local tag=$1 set=$2 adapter=$3
    if [ -f "$OUT/$set/$tag/metrics.json" ]; then echo "skip $tag/$set (done)"; return; fi
    if [ ! -f "$D/$set.json" ]; then echo "SKIP $tag/$set: no $D/$set.json"; return; fi
    if [ ! -d "$adapter" ]; then echo "SKIP $tag/$set: no adapter at $adapter"; return; fi
    echo "=== $tag on $set ==="
    $UV run scripts/evaluate.py --task habitat --tag "$tag" --eval-file "$D/$set.json" \
        --out-dir "$OUT/$set" --adapter "$adapter"
}
run general   eval_plain        $GEN
run general   eval_instructed   $GEN
run general   eval_pairs        $GEN
run v1_cosmos eval_pairs        outputs/runs_instr_v1/cosmos/adapter
run general   probe_wrong_side  $GEN
run general   probe_handwritten $GEN
echo "all cells done: uv run scripts/visualize_instruction_general.py"
