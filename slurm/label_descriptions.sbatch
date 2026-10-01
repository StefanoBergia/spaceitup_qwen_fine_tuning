#!/bin/bash
# Label d1 route descriptions with Cosmos3-Nano, unattended: serve + both stages in one job.
#
#     sbatch slurm/label_descriptions.sbatch              # gate: 300 eval routes, then stop for the audit
#     STAGE=full sbatch slurm/label_descriptions.sbatch   # eval + train (resumes after a TIMEOUT)
#
# Needs data/prepared_habitat_instr_v2/{events,objects}_{eval,train}.jsonl
# (uv run scripts/prepare_habitat_instr.py events). Spec:
# docs/superpowers/specs/2026-09-30-route-descriptions-design.md. Safe to re-submit: every
# call passes --resume. After a prompt fix, re-run the gate from scratch:
#     OVERWRITE=1 sbatch slurm/label_descriptions.sbatch
# Afterwards, on the login node:
#     uv run scripts/prepare_habitat_instr.py finalize-descriptions
#SBATCH --job-name=cosmos3-label-desc
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v3
source slurm/cosmos3_serve_lib.sh
D="${DATA:-data/prepared_habitat_instr_v2}"
OUT=outputs/instr_labels_v3
CONCURRENCY="${CONCURRENCY:-8}"
UV="${UV:-$HOME/.local/bin/uv}"
STAGE="${STAGE:-gate}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
for s in eval train; do
    for f in events objects; do
        [ -f "$D/${f}_$s.jsonl" ] || { echo "missing $D/${f}_$s.jsonl — run prepare_habitat_instr.py events"; exit 1; }
    done
done

cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy

label() {  # label <split> <tag> [--limit N] [--gate]
    local s=$1 tag=$2; shift 2
    local first=--resume gate=() limit=()
    [ -n "${OVERWRITE:-}" ] && first=--overwrite
    while [ $# -gt 0 ]; do
        case "$1" in
            --gate) gate=(--gate) ;;
            --limit) limit=(--limit "$2"); shift ;;
        esac
        shift
    done
    # name pass 1 may leave request-error rows; pass 2 (--resume) retries them, and only it gates
    echo; echo "=============== name: $tag (pass 1) ==============="
    "$UV" run scripts/label_descriptions.py --stage name --events "$D/events_$s.jsonl" \
        --objects "$D/objects_$s.jsonl" --out "$OUT/names_${tag}_d1.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" "$first" "${limit[@]}"
    echo; echo "=============== name: $tag (pass 2, retries errors) ==============="
    "$UV" run scripts/label_descriptions.py --stage name --events "$D/events_$s.jsonl" \
        --objects "$D/objects_$s.jsonl" --out "$OUT/names_${tag}_d1.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume "${limit[@]}" "${gate[@]}"
    echo; echo "=============== compose: $tag ==============="
    "$UV" run scripts/label_descriptions.py --stage compose --events "$D/events_$s.jsonl" \
        --names "$OUT/names_${tag}_d1.jsonl" --out "$OUT/desc_${tag}_d1.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" "$first" "${limit[@]}" "${gate[@]}"
}

if [ "$STAGE" = "gate" ]; then
    # set -e: a failed gate (exit 2) ends the job; the EXIT trap stops the server
    label eval gate --limit 300 --gate
    echo "gate labelled — audit it, then: STAGE=full sbatch slurm/label_descriptions.sbatch"
    exit 0
fi
label eval eval
label train train
echo; echo "=============== done ==============="
wc -l "$OUT"/*_d1.jsonl
