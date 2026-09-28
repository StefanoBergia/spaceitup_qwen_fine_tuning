#!/bin/bash
# Label the v2 instruction set with Cosmos3-Nano, unattended: serve + client in one job.
#
#     sbatch slurm/label_instructions.sbatch                  # i4 gate only (default), then audit
#     STAGE=full sbatch slurm/label_instructions.sbatch       # i4 eval + train (~8 h)
#     PROMPT=i3 STAGE=full sbatch slurm/label_instructions.sbatch   # the old v2 prompt
#
# i4 (2026-09-28, docs/superpowers/specs/2026-09-28-grounded-landmarks-design.md) needs
# data/prepared_habitat_instr_v2/landmarks_{eval,train}.jsonl (prepare_habitat_instr.py
# landmarks). Its gate is the first 300 eval records with a target; the job stops
# after the gate so a human can audit the labels before the full run.
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
#         outputs/instr_labels_v2/train_i4.jsonl outputs/instr_labels_v2/eval_i4.jsonl
#SBATCH --job-name=cosmos3-label-instr
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
# Telegram notification through the cluster's SLURM mail plugin (start, end, failure — a
# failed gate exits non-zero, so it arrives as a failure)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=telegram:422627024
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
PROMPT="${PROMPT:-i4}"
# STAGE=gate: only the gate set, then stop (i4: a human audits it before the full run)
# STAGE=full: skip the gate, label eval + train
STAGE="${STAGE:-gate}"
# unbuffered so the log shows progress; thread caps for the client's image rendering
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
[ -f "$D/labelset_eval.json" ] || { echo "missing $D/labelset_eval.json — run prepare_habitat_instr.py general first"; exit 1; }
if [ "$PROMPT" = "i4" ]; then
    for s in eval train; do
        [ -f "$D/landmarks_$s.jsonl" ] || { echo "missing $D/landmarks_$s.jsonl — run prepare_habitat_instr.py landmarks"; exit 1; }
    done
fi
# --landmarks only for i4
lm() { [ "$PROMPT" = "i4" ] && echo "--landmarks $D/landmarks_$1.jsonl" || true; }

cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy

if [ "$STAGE" = "gate" ] && [ "$PROMPT" = "i4" ]; then
    # the first 300 eval records that have a target: after the route_bends_both_ways drop
    # (2026-09-28) almost no pair keeps both sides, so a pair-based gate would label nothing
    echo; echo "=============== gate (i4): 300 eval records with a landmark target ==============="
    "$UV" run python - "$D/labelset_eval.json" "$D/landmarks_eval.jsonl" "$OUT/gate_split_i4.json" <<'PY'
import json, sys
recs = json.load(open(sys.argv[1]))
has = {json.loads(l)["id"] for l in open(sys.argv[2]) if json.loads(l)["target"] != "none"}
keep = [r for r in recs if r["id"] in has][:300]
json.dump(keep, open(sys.argv[3], "w"))
print(len(keep), "gate records")
PY
    "$UV" run scripts/label_instructions.py --prompt i4 --split "$OUT/gate_split_i4.json" $(lm eval) \
        --out "$OUT/gate_i4.jsonl" --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume
    echo "gate labelled — audit it, then: STAGE=full sbatch slurm/label_instructions.sbatch"
    exit 0
fi

if [ "$STAGE" = "gate" ] && [ "$GATE_N" -gt 0 ]; then
    echo; echo "=============== gate ($PROMPT): $GATE_N single-route eval records ==============="
    "$UV" run python - "$D/labelset_eval.json" "$OUT/gate_split.json" "$GATE_N" <<'PY'
import json, sys
recs = [r for r in json.load(open(sys.argv[1]))
        if r["habitat_meta"]["kind"] == "instructed" and not r["habitat_meta"].get("pair_id")]
json.dump(recs[: int(sys.argv[3])], open(sys.argv[2], "w"))
PY
    # set -e: a failed gate (exit 2) ends the job here; the EXIT trap stops the server
    "$UV" run scripts/label_instructions.py --prompt "$PROMPT" --split "$OUT/gate_split.json" \
        --out "$OUT/gate_$PROMPT.jsonl" --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume \
        --gate --min-accept "$GATE_MIN_ACCEPT" --min-pair-survival 0
    echo "gate PASSED"
fi

for s in eval train; do
    echo; echo "=============== labelling $s ==============="
    "$UV" run scripts/label_instructions.py --prompt "$PROMPT" --split "$D/labelset_$s.json" $(lm $s) \
        --out "$OUT/${s}_${PROMPT}.jsonl" --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume
    echo; echo "=============== retry rejected: $s ==============="
    "$UV" run scripts/label_instructions.py --prompt "$PROMPT" --split "$D/labelset_$s.json" $(lm $s) \
        --out "$OUT/${s}_${PROMPT}.jsonl" --endpoint "$BASE/v1" --concurrency "$CONCURRENCY" --resume --retry-rejected
done
echo; echo "=============== done ==============="
wc -l "$OUT"/*_"$PROMPT".jsonl
