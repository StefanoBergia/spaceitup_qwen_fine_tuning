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
# thor is shared: a fixed 8000 collided with another server on 2026-09-28 (job 91926,
# "Address already in use"). Unless PORT is given, take the first free port from a
# job-dependent start, checked with bash's /dev/tcp (no python: the repo's .python-version
# breaks bare python3 here).
cosmos3_free_port() {
    local p=$((8000 + ${SLURM_JOB_ID:-0} % 500))
    while (echo > "/dev/tcp/127.0.0.1/$p") 2>/dev/null; do
        p=$((p + 1))
    done
    echo "$p"
}
PORT="${PORT:-$(cosmos3_free_port)}"
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
# A server stuck at startup prints every thread's traceback on SIGABRT (cosmos3_dump_stuck).
export PYTHONFAULTHANDLER=1

# A100 is SM80, which FlashInfer claims to support, so vLLM picks its top-k/top-p sampler
# by default — and that kernel is JIT-compiled on first use, needing `ninja` and `nvcc` at
# runtime. Neither exists here, so startup died in the profiling run with
#     FileNotFoundError: No such file or directory: 'ninja'
# 0 selects the PyTorch-native sampler instead (vllm/v1/sample/ops/topk_topp_sampler.py).
# Sampling cost is noise next to generating thousands of reasoning tokens, and this keeps
# runtime compilation out of the picture — cf. the Triton/Python.h trap in CLAUDE.md.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Since thor's 2026-10-20 migration SLURM hands a MIG slice over as CUDA_VISIBLE_DEVICES=MIG-<uuid>, and vLLM 0.21
# int()s that variable (vllm/platforms/interface.py, device_id_to_physical_device_id): jobs 95027/95028 died at
# startup on 3g.40gb slices with "invalid literal for int() with base 10: 'MIG-...'". The job's cgroup exposes only
# its own slice, so CUDA's device 0 is that slice (the October jobs got a plain index). Only when nvidia-smi -L lists
# exactly that one slice is the UUID swapped for 0; anything else stops the job rather than guess a device.
cosmos3_mig_index() {
    case "${CUDA_VISIBLE_DEVICES:-}" in MIG-*) ;; *) return 0 ;; esac
    local seen
    seen=$(nvidia-smi -L | grep -o 'MIG-[0-9a-fA-F-]*' || true)
    if [ "$seen" != "$CUDA_VISIBLE_DEVICES" ]; then
        echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, but this job sees the MIG devices [${seen//$'\n'/ }]:" \
             "not exactly that one slice, so index 0 may be another one (vLLM cannot take the UUID itself)." >&2
        exit 1
    fi
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> 0 (the only MIG slice this job sees; vLLM needs an index)"
    export CUDA_VISIBLE_DEVICES=0
}

cosmos3_start_server() {
    if [ ! -x "$VENV/bin/vllm" ]; then
        echo "ERROR: $VENV missing. Run 'bash slurm/cosmos3_env.sh' on the login node first." >&2
        exit 1
    fi
    # A stale file from a previous job would point the client at a dead host.
    rm -f "$ENDPOINT_FILE"
    ulimit -c 0     # cosmos3_dump_stuck's SIGABRT must not write a 60 GB core file
    BASE="http://$(hostname):$PORT"
    echo "node=$(hostname) port=$PORT model=$MODEL max_model_len=$MAX_MODEL_LEN"
    nvidia-smi
    cosmos3_mig_index

    # --hf-overrides loads only the reasoner tower out of the shared MoT checkpoint, and
    # --mm-encoder-tp-mode data is the Cosmos vision tower's mode; both are Cosmos-only, so a
    # non-Cosmos model (e.g. google/gemma-3-12b-it) is served without them. $MODEL is read
    # here, not at source time, so one job can serve several models in turn.
    # --allowed-local-media-path / lets the client pass file:// URIs, which matters because
    # the prepared splits reference images by absolute NFS path.
    local cosmos_flags=()
    case "$MODEL" in
        *[Cc]osmos3*) cosmos_flags=(--hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'
                                    --mm-encoder-tp-mode data) ;;
    esac
    "$VENV/bin/vllm" serve "$MODEL" \
        ${cosmos_flags[@]+"${cosmos_flags[@]}"} \
        --tensor-parallel-size 1 \
        --async-scheduling \
        --allowed-local-media-path / \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --max-model-len "$MAX_MODEL_LEN" \
        --host 0.0.0.0 --port "$PORT" &
    SERVER_PID=$!
}

cosmos3_stop_server() {
    # Kill the server, wait for it to exit (frees the GPU memory), drop the endpoint file.
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
    rm -f "$ENDPOINT_FILE"
}

cosmos3_wait_healthy() {
    # Model load is slow (35 GB off NFS); poll rather than guess a sleep. Bail out early if
    # the server dies so the failure shows up here rather than as a 40-minute timeout.
    echo "waiting for $BASE/health ..."
    local polls="${STARTUP_POLLS:-240}" every="${STARTUP_POLL_S:-10}"
    for i in $(seq 1 "$polls"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vllm exited during startup — see the traceback above." >&2
            wait "$SERVER_PID"
            exit 1
        fi
        if curl -sf "$BASE/health" >/dev/null 2>&1; then
            echo "$BASE/v1" > "$ENDPOINT_FILE"
            echo "READY after ~$((i * every))s -> $ENDPOINT_FILE ($BASE/v1)"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv
            return 0
        fi
        sleep "$every"
    done
    echo "ERROR: server did not become healthy within $((polls * every / 60)) minutes; where it is stuck:" >&2
    cosmos3_dump_stuck
    exit 1
}

cosmos3_dump_stuck() {
    # Jobs 94222/94251: Cosmos-R2's server printed nothing at all for 40 minutes. Log each server process's
    # state and kernel wait channel (D = stuck in I/O, e.g. NFS) and its last open files, then SIGABRT it:
    # PYTHONFAULTHANDLER prints its Python tracebacks into this log.
    local p
    for p in $(pgrep -P "$SERVER_PID") "$SERVER_PID"; do
        ps -o pid=,stat=,wchan:32=,etime=,args= -p "$p" | cut -c1-200 || true    # it may exit meanwhile (set -e)
        ls -l "/proc/$p/fd" 2>/dev/null | grep -v -e "pipe:" -e "socket:" -e "/dev/" | tail -5 || true
        kill -ABRT "$p" 2>/dev/null || true
    done
    sleep 5
}
