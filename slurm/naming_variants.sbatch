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
# MODE=choose (Addendum C): a chooser model picks one of the candidate names per object (V0 = today's names, the
# fallback, + V1 V2 V3), or "none" -> V0. Needs prepare_habitat_instr.py today-names for the train/eval/steer sets.
#
#     MODE=choose sbatch slurm/naming_variants.sbatch                                   # C1 C2 C3 on the naming_ab sample
#     MODE=choose CHOOSERS=C1 SETS=check sbatch slurm/naming_variants.sbatch            # the fresh 50-object check
#     MODE=choose CHOOSERS=C1 SETS="train eval steer" sbatch slurm/naming_variants.sbatch   # rollout with the winner
#
# Every input of every set is checked before a model is served. A model that fails to start is logged and skipped; a
# labelling pass that fails (e.g. the server died mid-run) is logged as WARN; the others still run. Every call passes
# --resume, so a re-submit continues where a TIMEOUT or a failure stopped it (rename-texts refuses a file with gaps).
# Long runs: pass a longer limit at submission (sbatch --time=3-00:00:00 ...; the partition allows 7 days).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v3
source slurm/cosmos3_serve_lib.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
UV="${UV:-$HOME/.local/bin/uv}"
D="${DATA:-data/prepared_habitat_instr_v4}"
OUT="${OUT:-outputs/instr_labels_v3}"
MODE="${MODE:-name}"
VARIANTS="${VARIANTS:-V1 V2 V3}"
CHOOSERS="${CHOOSERS:-C1 C2 C3}"
SETS="${SETS:-sample}"
NAME_PROMPT="${NAME_PROMPT:-n2}"
CHOOSE_PROMPT="${CHOOSE_PROMPT:-c1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"     # V2 lost 7 of 119 sample names to thinking cut at 4096 (job 93495)
CONCURRENCY="${CONCURRENCY:-32}"
case "$MODE" in name) MODELS="$VARIANTS" ;; choose) MODELS="$CHOOSERS" ;; *) echo "MODE must be name or choose"; exit 1 ;; esac

model_of() {
    case "$1" in
        V1|C3) echo google/gemma-3-12b-it ;;
        V2|C1) echo Qwen/Qwen3.6-35B-A3B-FP8 ;;
        V3|C2) echo nvidia/Cosmos-Reason2-32B ;;
        *) echo "unknown variant or chooser $1" >&2; return 1 ;;
    esac
}
objects_of() {   # objects_of <set>
    case "$1" in
        sample) echo "$D/naming_ab/objects_sample.jsonl" ;;
        check) echo "$D/naming_check/objects_sample.jsonl" ;;
        train) echo "$D/objects_train.jsonl" ;;
        eval) echo "$D/objects_eval.jsonl" ;;
        steer) echo "$D/steer/objects_eval.jsonl" ;;
        *) echo "unknown set $1" >&2; return 1 ;;
    esac
}
candidates_of() {   # candidates_of <set>: the V0..V3 name files the chooser reads, as VARIANT=PATH
    local s=$1
    case "$s" in
        sample) for v in V0 V1 V2 V3; do echo "$v=$D/naming_ab/names_${v}.jsonl"; done ;;
        check) echo "V0=$OUT/names_train_V0.jsonl"; for v in V1 V2 V3; do echo "$v=$OUT/names_train_${v}_${NAME_PROMPT}.jsonl"; done ;;
        *) echo "V0=$OUT/names_${s}_V0.jsonl"; for v in V1 V2 V3; do echo "$v=$OUT/names_${s}_${v}_${NAME_PROMPT}.jsonl"; done ;;
    esac
}
out_of() {   # out_of <set> <variant or chooser>
    local s=$1 v=$2 c=$2
    if [ "$MODE" = choose ]; then      # rollout files take the winner-file name rename-texts --variant C<k> reads
        case "$s" in
            sample) echo "$D/naming_ab/choose_${c}.jsonl" ;;
            check) echo "$D/naming_check/choose_${c}.jsonl" ;;
            *) echo "$OUT/names_${s}_${c}_${NAME_PROMPT}.jsonl" ;;
        esac
    elif [ "$s" = sample ]; then echo "$D/naming_ab/names_${v}.jsonl"
    else echo "$OUT/names_${s}_${v}_${NAME_PROMPT}.jsonl"; fi
}
run_passes() {   # run_passes <variant> <set> <label_descriptions args...>: two --resume passes, a failure is logged
    local v=$1 s=$2; shift 2
    echo; echo "--- $v $s (pass 1) ---"
    "$UV" run scripts/label_descriptions.py "$@" --resume || echo "WARN $v $s pass 1 failed (exit $?)"
    echo; echo "--- $v $s (pass 2, retries errors) ---"
    "$UV" run scripts/label_descriptions.py "$@" --resume || echo "WARN $v $s pass 2 failed (exit $?)"
}

missing=()
for s in $SETS; do
    objs="$(objects_of "$s")"
    [ -f "$objs" ] || missing+=("$objs")
    if [ "$MODE" = choose ]; then
        while read -r c; do [ -f "${c#*=}" ] || missing+=("${c#*=}"); done < <(candidates_of "$s")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing input: %s\n' "${missing[@]}"
    exit 1
fi

trap 'cosmos3_stop_server' EXIT
echo "MODE=$MODE MODELS=$MODELS SETS=$SETS NAME_PROMPT=$NAME_PROMPT CHOOSE_PROMPT=$CHOOSE_PROMPT MAX_TOKENS=$MAX_TOKENS"
for v in $MODELS; do
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
        common=(--objects "$(objects_of "$s")" --out "$(out_of "$s" "$v")" --endpoint "$BASE/v1"
                --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS")
        if [ "$MODE" = choose ]; then
            mapfile -t cands < <(candidates_of "$s")
            run_passes "$v" "$s" --stage choose "${common[@]}" --candidates "${cands[@]}" --choose-prompt "$CHOOSE_PROMPT"
        else
            run_passes "$v" "$s" --stage name --all-objects "${common[@]}" --name-prompt "$NAME_PROMPT"
        fi
    done
    cosmos3_stop_server
done
echo "naming variants done"
