#!/bin/bash
# Label the v2 instruction set with Cosmos3-Nano, unattended: serve + client in one job.
#
#     sbatch slurm/label_instructions.sbatch
#
# One allocation serves the model and drives the client over localhost, so nothing depends
# on a terminal or a session staying alive (on 2026-09-24 a READY server idled 8 h because
# the session watching it had ended). Stage 0 is the gate: GATE_N single-route eval records
# must reach accept rate >= GATE_MIN_ACCEPT with no drawing references, or the job stops
# before the full run. Then eval and train labelsets, each followed by one retry of the
# rejected rows with seed + 1. Safe to re-submit: every call passes --resume.
#
# ~28.9k records (25,985 train + 2,866 eval). v1 ran 125 records/min on a full A100 at
# concurrency 8; a 3g.40gb slice may be slower, hence the 14 h limit.
#
# Afterwards, on the login node:
#     uv run scripts/prepare_habitat_instr.py finalize-general --instructions \
#         outputs/instr_labels_v2/train_i2.jsonl outputs/instr_labels_v2/eval_i2.jsonl
#SBATCH --job-name=cosmos3-label-instr
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/instr_labels_v2
source slurm/cosmos3_serve_lib.sh
D="${DATA:-data/prepared_habitat_instr_v2}"
OUT=outputs/instr_labels_v2
CONCURRENCY="${CONCURRENCY:-8}"
GATE_N="${GATE_N:-200}"
GATE_MIN_ACCEPT="${GATE_MIN_ACCEPT:-0.9}"
UV="${UV:-$HOME/.local/bin/uv}"
# unbuffered so the log shows progress; thread caps for the client's image rendering
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
[ -f "$D/labelset_eval.json" ] || { echo "missing $D/labelset_eval.json — run prepare_habitat_instr.py general first"; exit 1; }

cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy

if [ "$GATE_N" -gt 0 ]; then
    echo; echo "=============== gate: $GATE_N single-route eval records ==============="
    "$UV" run python - "$D/labelset_eval.json" "$OUT/gate_split.json" "$GATE_N" <<'PY'
import json, sys
recs = [r for r in json.load(open(sys.argv[1]))
        if r["habitat_meta"]["kind"] == "instructed" and not r["habitat_meta"].get("pair_id")]
json.dump(recs[: int(sys.argv[3])], open(sys.argv[2], "w"))
PY
    # set -e: a failed gate (exit 2) ends the job here; the EXIT trap stops the server
    "$UV" run scripts/label_instructions.py --split "$OUT/gate_split.json" --out "$OUT/gate_i2.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume \
        --gate --min-accept "$GATE_MIN_ACCEPT" --min-pair-survival 0
    echo "gate PASSED"
fi

for s in eval train; do
    echo; echo "=============== labelling $s ==============="
    "$UV" run scripts/label_instructions.py --split "$D/labelset_$s.json" --out "$OUT/${s}_i2.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume
    echo; echo "=============== retry rejected: $s ==============="
    "$UV" run scripts/label_instructions.py --split "$D/labelset_$s.json" --out "$OUT/${s}_i2.jsonl" \
        --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume --retry-rejected
done
echo; echo "=============== done ==============="
wc -l "$OUT"/*_i2.jsonl
