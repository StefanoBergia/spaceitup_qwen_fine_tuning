# Shared Cosmos3-Nano serving helpers, sourced by the sbatch files that need the reasoner.
#
#   source slurm/cosmos3_serve_lib.sh
#   cosmos3_start_server          # backgrounds vllm, sets SERVER_PID and BASE
#   cosmos3_wait_healthy          # blocks until /health answers, writes the endpoint file
#
# Kept out of the sbatch files themselves because two jobs need identical startup: the
# long-lived interactive server (serve_cosmos3.sbatch) and the unattended full-dataset run
# (label_full_path.sbatch). The flags below are load-bearing and were each paid for once —
# duplicating them invites fixing a startup bug in one copy only.

MODEL="${MODEL:-nvidia/Cosmos3-Nano}"
PORT="${PORT:-8000}"
VENV="${VENV:-.venv-cosmos3}"
ENDPOINT_FILE="${ENDPOINT_FILE:-outputs/cosmos3_endpoint.txt}"

# The checkpoint advertises a 262,144-token context (the 256K reasoning window). vLLM sizes
# the KV cache so a single max-length request fits, which wants 36 GiB and fails at startup.
# We need nothing like that: prompt ~700 tokens, image ~1k, generation capped at 8192, so
# ~10k is the real worst case. 16384 covers it and leaves KV for several concurrent
# requests, which is what makes --concurrency worth anything on the full run.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

export HF_HUB_OFFLINE=1          # weights come from the shared NFS cache, never the network
export VLLM_LOGGING_LEVEL=INFO

# A100 is SM80, which FlashInfer claims to support, so vLLM picks its top-k/top-p sampler
# by default — and that kernel is JIT-compiled on first use, needing `ninja` and `nvcc` at
# runtime. Neither exists here, so startup died in the profiling run with
#     FileNotFoundError: No such file or directory: 'ninja'
# 0 selects the PyTorch-native sampler instead (vllm/v1/sample/ops/topk_topp_sampler.py).
# Sampling cost is noise next to generating thousands of reasoning tokens, and this keeps
# runtime compilation out of the picture — cf. the Triton/Python.h trap in CLAUDE.md.
export VLLM_USE_FLASHINFER_SAMPLER=0

cosmos3_start_server() {
    if [ ! -x "$VENV/bin/vllm" ]; then
        echo "ERROR: $VENV missing. Run 'bash slurm/cosmos3_env.sh' on the login node first." >&2
        exit 1
    fi
    # A stale file from a previous job would point the client at a dead host.
    rm -f "$ENDPOINT_FILE"
    BASE="http://$(hostname):$PORT"
    echo "node=$(hostname) port=$PORT model=$MODEL max_model_len=$MAX_MODEL_LEN"
    nvidia-smi

    # --hf-overrides loads only the reasoner tower out of the shared MoT checkpoint.
    # --allowed-local-media-path / lets the client pass file:// URIs, which matters because
    # the prepared splits reference images by absolute NFS path.
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
}

cosmos3_wait_healthy() {
    # Model load is slow (35 GB off NFS); poll rather than guess a sleep. Bail out early if
    # the server dies so the failure shows up here rather than as a 40-minute timeout.
    echo "waiting for $BASE/health ..."
    for i in $(seq 1 240); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vllm exited during startup — see the traceback above." >&2
            wait "$SERVER_PID"
            exit 1
        fi
        if curl -sf "$BASE/health" >/dev/null 2>&1; then
            echo "$BASE/v1" > "$ENDPOINT_FILE"
            echo "READY after ~$((i * 10))s -> $ENDPOINT_FILE ($BASE/v1)"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv
            return 0
        fi
        sleep 10
    done
    echo "ERROR: server did not become healthy within 40 minutes." >&2
    exit 1
}
