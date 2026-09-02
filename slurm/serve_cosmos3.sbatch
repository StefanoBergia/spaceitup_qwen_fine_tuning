#!/bin/bash
# Serve the Cosmos3-Nano *reasoner* (image+text -> text) as an OpenAI-compatible endpoint.
#
#     sbatch slurm/serve_cosmos3.sbatch
#
# Use this for prompt iteration: it publishes outputs/cosmos3_endpoint.txt once /health is
# up and then holds the server for the job's lifetime, so you drive it from odin (which
# reaches thor over TCP) instead of paying a 35 GB model load per attempt:
#     uv run scripts/label_traces.py --task choice --split <split.json> --limit 3 --show
#
# For the full-dataset run use slurm/label_full_path.sbatch instead — it starts the same
# server and runs the client inside the one job, so nothing depends on a terminal staying
# open. Startup itself lives in slurm/cosmos3_serve_lib.sh, shared by both.
#
# Stop it with `scancel <jobid>` when you are done.
#
# MEASURED 2026-07-22: the reasoner override loads only 16.65 GiB of the 30.35 GB
# transformer/ — the diffusion tower really does stay unloaded. That fits a 3g.40gb MIG
# slice with room to spare, which is what the full-dataset job asks for. This one keeps the
# full card only because interactive iteration benefits from the extra KV cache; drop it to
# gpu:3g.40gb:1 if the A100 is contended.
#
# (An earlier note here concluded from vLLM's "16.34 GiB free for KV" that thor's a100 is a
# 40 GB card. It is not — nvidia-smi reports 81920 MiB. thor is shared, so *free* memory
# varies between runs; read capacity from nvidia-smi, never from vLLM's KV number.)
#SBATCH --job-name=cosmos3-serve
#SBATCH --partition=A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm

source slurm/cosmos3_serve_lib.sh

cosmos3_start_server
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT
cosmos3_wait_healthy

wait $SERVER_PID
