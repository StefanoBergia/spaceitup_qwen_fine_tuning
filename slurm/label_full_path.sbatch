#!/bin/bash
# Label the WHOLE path-regression dataset with the v6 prompt, unattended.
#
#     sbatch slurm/label_full_path.sbatch
#
# One job that both serves the model and drives the client, so there is no second terminal
# to keep alive and nothing to coordinate — the 2026-07-22 server was started under srun
# and died with its terminal. The client talks to localhost, so it does not depend on
# outputs/cosmos3_endpoint.txt being current either.
#
# Covers both splits of data/prepared_habitat_v2 — 8,140 train_full + 1,000 eval = 9,140
# samples. Writes outputs/traces_full/path_v6_<split>.jsonl, deliberately NOT into
# outputs/traces/, so the prompt-iteration runs the report is built from stay untouched.
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
# Runtime: single-threaded labelling measured 5-10 s/sample, i.e. 13-25 h for 9,140. At
# --concurrency 8 expect roughly 2-4 h plus ~10 min of model load. --time is set well past
# that because being killed at 95% and resuming is pure waste.
#SBATCH --job-name=cosmos3-label-path
#SBATCH --partition=A100
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/traces_full

source slurm/cosmos3_serve_lib.sh

DATA=data/prepared_habitat_v2
OUT_DIR=outputs/traces_full
CONCURRENCY="${CONCURRENCY:-8}"
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
for split in train_full eval; do
    echo
    echo "=============== labelling $split ==============="
    "$UV" run scripts/label_traces.py \
        --task path \
        --split "$DATA/$split.json" \
        --endpoint "$BASE/v1" \
        --out-dir "$OUT_DIR" \
        --out-name "path_v6_$split" \
        --concurrency "$CONCURRENCY" \
        --resume
done

echo
echo "=============== done ==============="
wc -l "$OUT_DIR"/path_v6_*.jsonl
