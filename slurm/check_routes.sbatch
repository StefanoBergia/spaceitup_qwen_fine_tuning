#!/bin/bash
#SBATCH --job-name=check-routes
#SBATCH --partition=A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
# Route checker (spec docs/superpowers/specs/2026-11-16-route-checker-design.md): Qwen3.6-35B-A3B-FP8 (35 GB FP8, the
# full A100 like slurm/naming_variants.sbatch) judges each manifest row's image + text with prompt k1. Manifests come
# from scripts/audit_instruction_labels.py --mode check-calibration / check-manifest (CPU, login node).
#
#     sbatch slurm/check_routes.sbatch                                              # the calibration set (~550 marked items)
#     MANIFESTS="train eval" sbatch --time=12:00:00 slurm/check_routes.sbatch       # every d3 text (after a passed calibration)
#
# Reads outputs/route_check/<set>/manifest.jsonl, writes outputs/route_check/<set>/check_k1.jsonl. Every call passes
# --resume (two passes; the second retries request errors), so a re-submit continues after a TIMEOUT. Every manifest
# is checked before the model is served.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
source slurm/cosmos3_serve_lib.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
UV="${UV:-$HOME/.local/bin/uv}"
R="${ROOT:-outputs/route_check}"
MANIFESTS="${MANIFESTS:-calibration}"
MODEL="${CHECK_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"     # Qwen3.6 thinks first; 4096 cut chooser replies (job 93495)
CONCURRENCY="${CONCURRENCY:-32}"

missing=()
for s in $MANIFESTS; do [ -f "$R/$s/manifest.jsonl" ] || missing+=("$R/$s/manifest.jsonl"); done
if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing input: %s\n' "${missing[@]}"
    exit 1
fi

trap 'cosmos3_stop_server' EXIT
echo "MANIFESTS=$MANIFESTS MODEL=$MODEL MAX_TOKENS=$MAX_TOKENS CONCURRENCY=$CONCURRENCY"
cosmos3_start_server
cosmos3_wait_healthy
for s in $MANIFESTS; do
    args=(--stage check --manifest "$R/$s/manifest.jsonl" --out "$R/$s/check_k1.jsonl" --endpoint "$BASE/v1"
          --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS")
    echo; echo "=============== check: $s (pass 1) ==============="
    "$UV" run scripts/label_descriptions.py "${args[@]}" --resume || echo "WARN $s pass 1 failed (exit $?)"
    echo; echo "=============== check: $s (pass 2, retries errors) ==============="
    "$UV" run scripts/label_descriptions.py "${args[@]}" --resume || echo "WARN $s pass 2 failed (exit $?)"
done
echo "route check done: uv run scripts/route_check_report.py --manifest $R/calibration/manifest.jsonl --check $R/calibration/check_k1.jsonl --out $R/calibration"
