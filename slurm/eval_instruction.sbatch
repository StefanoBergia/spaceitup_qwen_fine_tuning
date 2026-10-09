#!/bin/bash
# Evaluate every cell of the instruction-conditioned comparison, skipping finished ones.
#   sbatch slurm/eval_instruction.sbatch
# Cells: {instr_template, instr_cosmos} x {template, cosmos} eval sets, plus the round-4
# LoRA and the base model on the cosmos set. ~5 min per cell on a 3g.40gb slice.
#SBATCH --job-name=qwen-instr-eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
UV=~/.local/bin/uv
D=data/prepared_habitat_instr_v1
OUT=outputs/eval_instr_v1
RUNS=outputs/runs_instr_v1

run() {  # run <tag> <set> [adapter]
    local tag=$1 set=$2 adapter=${3:-}
    if [ -f "$OUT/$set/$tag/metrics.json" ]; then echo "skip $tag/$set (done)"; return; fi
    if [ -n "$adapter" ] && [ ! -d "$adapter" ]; then echo "SKIP $tag/$set: no adapter at $adapter"; return; fi
    echo "=== $tag on $set ==="
    $UV run scripts/evaluate.py --task habitat --tag "$tag" --eval-file "$D/eval_$set.json" \
        --out-dir "$OUT/$set" ${adapter:+--adapter "$adapter"}
}
run instr_template template $RUNS/template/adapter
run instr_template cosmos   $RUNS/template/adapter
run instr_cosmos   template $RUNS/cosmos/adapter
run instr_cosmos   cosmos   $RUNS/cosmos/adapter
run r4_lora        cosmos   outputs/runs_v4/habitat_train_full/adapter
run base           cosmos
echo "all cells done: uv run scripts/visualize_instruction_results.py"
