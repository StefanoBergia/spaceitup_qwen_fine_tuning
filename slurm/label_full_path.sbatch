#!/bin/bash
# Label the WHOLE path-regression dataset with the current prompt, unattended.
#
#     sbatch slurm/label_full_path.sbatch                       # v7 on data/prepared_habitat_v4
#     DATA=data/prepared_habitat_v2 VERSION=v6 sbatch slurm/label_full_path.sbatch   # the 2026-07 run
#
# One job that both serves the model and drives the client, so there is no second terminal
# to keep alive and nothing to coordinate — the 2026-07-22 server was started under srun
# and died with its terminal. The client talks to localhost, so it does not depend on
# outputs/cosmos3_endpoint.txt being current either.
#
# Covers both splits of $DATA. Round 4 (data/prepared_habitat_v4, prompt v7): 23,770
# train_full + 2,591 eval = 26,361 samples. Writes outputs/traces_full/path_$VERSION_<split>.jsonl,
# deliberately NOT into outputs/traces/, so the prompt-iteration runs the report is built
# from stay untouched. Round 2 (v2, prompt v6) was 9,140 samples and is kept side by side.
#
# STAGE 0 IS THE GATE. Before the full run the job labels GATE_N (default 200) eval samples
# into outputs/traces/path_$VERSION_gate.jsonl and scores them with scripts/filter_traces.py
# under the current leakage detector. Leak rate above GATE_MAX_LEAK (default 0.01) or keep
# rate below GATE_MIN_KEEP (default 0.95) aborts the job before a single full-run sample is
# spent — the v6 full run leaked ~5% and the n=20 smoke gate could not see it. GATE_N=0
# skips the gate (only for a prompt that has already passed it).
#
# SAFE TO RE-SUBMIT. Every client call passes --resume: finished samples are skipped and
# failed ones retried, so a timeout, a preemption or a scancel costs only the samples that
# were in flight. Just sbatch it again.
#
# Sizing, measured 2026-07-22: the reasoner loads 16.65 GiB of the 30.35 GB transformer/,
# so a 3g.40gb MIG slice is ample and schedules far more easily than the full A100. (An
# earlier comment here inferred a 40 GB card from vLLM's free-KV number — that was wrong;
# nvidia-smi reports 81920 MiB. thor is shared, so *free* memory varies run to run.)
#
# Runtime: the 9,140-sample v6 run took 3h37m at --concurrency 8 (~43 samples/min), so
# 26,361 samples is ~10-11 h plus ~10 min of model load. --time is set well past that
# because being killed at 95% and resuming is pure waste.
#SBATCH --job-name=cosmos3-label-path
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/traces_full

source slurm/cosmos3_serve_lib.sh

DATA="${DATA:-data/prepared_habitat_v4}"
VERSION="${VERSION:-v7}"
OUT_DIR=outputs/traces_full
CONCURRENCY="${CONCURRENCY:-8}"
GATE_N="${GATE_N:-200}"
GATE_MAX_LEAK="${GATE_MAX_LEAK:-0.01}"
GATE_MIN_KEEP="${GATE_MIN_KEEP:-0.95}"
UV="${UV:-$HOME/.local/bin/uv}"

# Python block-buffers stdout when it is a file rather than a tty, so without this the
# per-sample progress sits in a 4-8 KB buffer and the log looks frozen for minutes at a
# time — and anything still buffered is lost outright if the job is killed. The JSONL is
# flushed per line regardless, so only the human-readable progress was ever at risk.
export PYTHONUNBUFFERED=1

cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy

# Client runs from the repo venv, not .venv-cosmos3 — it is a plain HTTP caller and must
# not drag in vLLM's torch pin.
if [ "$GATE_N" -gt 0 ]; then
    echo
    echo "=============== gate: $GATE_N eval samples, prompt $VERSION ==============="
    "$UV" run scripts/label_traces.py \
        --task path \
        --split "$DATA/eval.json" \
        --limit "$GATE_N" \
        --endpoint "$BASE/v1" \
        --out-dir outputs/traces \
        --prompt-version "$VERSION" \
        --out-name "path_${VERSION}_gate" \
        --concurrency "$CONCURRENCY" \
        --resume
    # set -e: a failed gate exits here, the EXIT trap stops the server, the full run never starts.
    "$UV" run scripts/filter_traces.py "outputs/traces/path_${VERSION}_gate.jsonl" \
        --max-leak-rate "$GATE_MAX_LEAK" --min-keep-rate "$GATE_MIN_KEEP"
    echo "gate PASSED"
fi

for split in train_full eval; do
    echo
    echo "=============== labelling $split ==============="
    "$UV" run scripts/label_traces.py \
        --task path \
        --split "$DATA/$split.json" \
        --endpoint "$BASE/v1" \
        --out-dir "$OUT_DIR" \
        --prompt-version "$VERSION" \
        --out-name "path_${VERSION}_$split" \
        --concurrency "$CONCURRENCY" \
        --resume
done

echo
echo "=============== done ==============="
wc -l "$OUT_DIR"/path_"$VERSION"_*.jsonl
