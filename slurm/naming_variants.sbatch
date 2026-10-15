#!/bin/bash
#SBATCH --job-name=naming-variants
#SBATCH --partition=A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
# Naming fix (spec docs/superpowers/specs/2026-10-15-object-naming-design.md): name a set of objects with the naming
# variants, one model at a time on the full A100 (V2 is 35 GB FP8, V3 63 GB bf16 -- neither fits a 3g.40gb slice).
#
#     sbatch slurm/naming_variants.sbatch                                   # comparison: V1 V2 V3 on naming_ab sample
#     VARIANTS=V2 SETS="train eval steer" sbatch slurm/naming_variants.sbatch  # rollout with the winner (V2 here)
#
# A model that fails to start is logged and skipped; the others still run. Every call passes --resume, so a
# re-submit continues where a TIMEOUT stopped it.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v3
source slurm/cosmos3_serve_lib.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
UV="${UV:-$HOME/.local/bin/uv}"
D="${DATA:-data/prepared_habitat_instr_v4}"
OUT=outputs/instr_labels_v3
VARIANTS="${VARIANTS:-V1 V2 V3}"
SETS="${SETS:-sample}"
NAME_PROMPT="${NAME_PROMPT:-n2}"
CONCURRENCY="${CONCURRENCY:-8}"

model_of() {
    case "$1" in
        V1) echo google/gemma-3-12b-it ;;
        V2) echo Qwen/Qwen3.6-35B-A3B-FP8 ;;
        V3) echo nvidia/Cosmos-Reason2-32B ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}
objects_of() {   # objects_of <set>
    case "$1" in
        sample) echo "$D/naming_ab/objects_sample.jsonl" ;;
        train) echo "$D/objects_train.jsonl" ;;
        eval) echo "$D/objects_eval.jsonl" ;;
        steer) echo "$D/steer/objects_eval.jsonl" ;;
        *) echo "unknown set $1" >&2; return 1 ;;
    esac
}
out_of() {   # out_of <set> <variant>
    local s=$1 v=$2
    if [ "$s" = sample ]; then echo "$D/naming_ab/names_${v}.jsonl"; else echo "$OUT/names_${s}_${v}_${NAME_PROMPT}.jsonl"; fi
}

trap 'cosmos3_stop_server' EXIT
echo "VARIANTS=$VARIANTS SETS=$SETS NAME_PROMPT=$NAME_PROMPT"
for v in $VARIANTS; do
    MODEL="$(model_of "$v")"
    echo; echo "=============== $v: $MODEL ==============="
    cosmos3_stop_server
    cosmos3_start_server
    if ! ( cosmos3_wait_healthy ); then
        echo "SKIP $v: $MODEL did not start"
        cosmos3_stop_server
        continue
    fi
    for s in $SETS; do
        objs="$(objects_of "$s")"
        [ -f "$objs" ] || { echo "missing $objs"; exit 1; }
        args=(--stage name --all-objects --objects "$objs" --out "$(out_of "$s" "$v")" --name-prompt "$NAME_PROMPT"
              --endpoint "$BASE/v1" --concurrency "$CONCURRENCY")
        echo; echo "--- $v $s (pass 1) ---"
        "$UV" run scripts/label_descriptions.py "${args[@]}" --resume
        echo; echo "--- $v $s (pass 2, retries errors) ---"
        "$UV" run scripts/label_descriptions.py "${args[@]}" --resume
    done
    cosmos3_stop_server
done
echo "naming variants done"
