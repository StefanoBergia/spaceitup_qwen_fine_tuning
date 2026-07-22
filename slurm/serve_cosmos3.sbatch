#!/bin/bash
# Serve the Cosmos3-Nano *reasoner* (image+text -> text) as an OpenAI-compatible endpoint.
#
#     sbatch slurm/serve_cosmos3.sbatch
#
# Publishes outputs/cosmos3_endpoint.txt once /health is up, then holds the server for the
# job's lifetime. The login node (odin) can reach thor over TCP, so you iterate prompts
# from odin against this one long-lived server instead of paying a 35 GB model load per
# attempt:
#     uv run scripts/label_traces.py --task choice --split <split.json> --limit 3 --show
#
# Stop it with `scancel <jobid>` when you are done — it holds the full A100.
#
# MEASURED 2026-07-22: the reasoner override loads only 16.65 GiB of the 30.35 GB
# transformer/ — the diffusion tower really does stay unloaded.
#
# Also measured: thor's `gpu:a100:1` is a **40 GB** A100, not 80 GB. vLLM reported 16.34 GiB
# free for KV after weights at --gpu-memory-utilization 0.92, which only balances for a
# 40 GB card (0.92*40 - 16.65 - ~3.9 activations). That means the full card and a 3g.40gb
# MIG slice have essentially the same memory, so there is no memory argument for holding
# the full card: once this serves cleanly, switch --gres to gpu:3g.40gb:1 and the job will
# both schedule more easily and stop blocking the full A100. Left as a100 for now so the
# first successful run is not changing two things at once.
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

MODEL=nvidia/Cosmos3-Nano
PORT="${PORT:-8000}"

# The checkpoint advertises a 262,144-token context (the 256K reasoning window). vLLM sizes
# the KV cache so a single max-length request fits, which wants 36 GiB against the ~16 GiB
# actually free after weights — startup then fails outright. We need nothing like that: the
# labelling prompt is ~700 tokens, the image ~1k, and generation is capped at 4096, so ~6k
# is the real worst case. 16384 doubles that and leaves KV for several concurrent requests,
# which matters when the full-dataset run turns on concurrency.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ENDPOINT_FILE=outputs/cosmos3_endpoint.txt
VENV=.venv-cosmos3

export HF_HUB_OFFLINE=1          # weights come from the shared NFS cache, never the network
export VLLM_LOGGING_LEVEL=INFO

# A100 is SM80, which FlashInfer claims to support, so vLLM picks its top-k/top-p sampler
# by default — and that kernel is JIT-compiled on first use, needing `ninja` and `nvcc` at
# runtime. Neither exists here, so startup died in the profiling run with
#     FileNotFoundError: No such file or directory: 'ninja'
# 0 selects the PyTorch-native sampler instead (vllm/v1/sample/ops/topk_topp_sampler.py).
# Sampling cost is noise next to generating 4k reasoning tokens, and this keeps runtime
# compilation out of the picture entirely — cf. the Triton/Python.h trap in CLAUDE.md.
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ ! -x "$VENV/bin/vllm" ]; then
    echo "ERROR: $VENV missing. Run 'bash slurm/cosmos3_env.sh' on the login node first." >&2
    exit 1
fi

# Stale file from a previous job would make the client talk to a dead host.
rm -f "$ENDPOINT_FILE"
BASE="http://$(hostname):$PORT"
echo "node=$(hostname) port=$PORT model=$MODEL"
nvidia-smi

# --hf-overrides loads only the reasoner tower out of the shared MoT checkpoint.
# --allowed-local-media-path / lets the client pass file:// URIs, which matters because the
# prepared splits reference images by absolute NFS path and we do not want to base64 them.
"$VENV/bin/vllm" serve "$MODEL" \
    --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
    --tensor-parallel-size 1 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --allowed-local-media-path / \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!
trap 'rm -f "$ENDPOINT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT

# Model load is slow (35 GB off NFS); poll rather than guess a sleep. Bail out early if the
# server dies so the failure shows up here instead of as a 40-minute timeout.
echo "waiting for $BASE/health ..."
for i in $(seq 1 240); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: vllm exited during startup — see the traceback above." >&2
        wait $SERVER_PID
        exit 1
    fi
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
        echo "$BASE/v1" > "$ENDPOINT_FILE"
        echo "READY after ~$((i * 10))s -> $ENDPOINT_FILE ($BASE/v1)"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv
        break
    fi
    sleep 10
done

if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "ERROR: server did not become healthy within 40 minutes." >&2
    exit 1
fi

wait $SERVER_PID
