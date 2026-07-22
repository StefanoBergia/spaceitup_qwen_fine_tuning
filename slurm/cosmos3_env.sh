#!/bin/bash
# Build .venv-cosmos3, the vLLM environment that serves the Cosmos3-Nano reasoner.
#
# Run on the LOGIN NODE (odin) — this needs network egress, and thor's jobs run with
# HF_HUB_OFFLINE=1:
#     bash slurm/cosmos3_env.sh
#
# Why a second venv: vLLM pins its own torch, which cannot coexist with this repo's
# torch==2.10.0+cu128 / transformers>=5.14. Only the server lives here; the client
# (scripts/label_traces.py) is plain HTTP and runs from the main .venv.
#
# WHY A GITHUB RELEASE WHEEL AND NOT PLAIN `vllm==0.21.0`:
# uv's --torch-backend flag only chooses *torch*'s wheel index. It has no effect on vLLM's
# own precompiled extension. The vllm wheel on PyPI is a CUDA 13 build — its _C.abi3.so is
# linked against libcudart.so.13 — so installing it here fails at import with
#     ImportError: libcudart.so.13: cannot open shared object file
# even though torch itself resolved correctly to cu128. vLLM publishes per-CUDA builds as
# GitHub release assets instead; for v0.21.0 the CUDA-12 variant is +cu129 (there is no
# cu128 asset). That build needs libcudart.so.12, which the cu128 torch stack already
# provides, and CUDA minor-version compatibility makes a 12.9 binary run fine on thor's
# 12.8 driver — unlike CUDA 13, which would need driver >= 580 against thor's 570.158.01.
#
# Idempotent: re-running an already-built env is a no-op you can use to verify it.

set -euo pipefail
cd "$(dirname "$0")/.."

UV=~/.local/bin/uv
VENV=.venv-cosmos3

# The model card pins 0.21.0; the cosmos cookbook says >=0.23.0. Both publish a +cu129
# asset, so to try the newer one:  VLLM_VERSION=0.23.0 bash slurm/cosmos3_env.sh
# (0.23.0's asset is tagged manylinux_2_28 rather than 2_34 — hence the two candidates.)
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
COSMOS_SPEC="vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
BASE="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}"

WHEEL=""
for TAG in manylinux_2_34 manylinux_2_28 manylinux_2_35; do
    URL="${BASE}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-${TAG}_x86_64.whl"
    if curl -sfIL -o /dev/null "$URL"; then WHEEL="$URL"; break; fi
done
if [ -z "$WHEEL" ]; then
    echo "ERROR: no +cu129 wheel found for vllm ${VLLM_VERSION}." >&2
    echo "Check the assets at https://github.com/vllm-project/vllm/releases/tag/v${VLLM_VERSION}" >&2
    exit 1
fi

echo "== building $VENV =="
echo "   vllm wheel: $WHEEL"
# Reuse an existing venv rather than rebuilding: `uv venv` errors on a populated dir, and
# re-creating means re-downloading a 457 MB wheel to reach the same state. RECREATE=1
# forces a clean rebuild when a resolution has genuinely gone bad.
if [ -d "$VENV" ] && [ -z "${RECREATE:-}" ]; then
    echo "   reusing existing $VENV (RECREATE=1 to rebuild from scratch)"
else
    $UV venv "$VENV" --python 3.12 --seed ${RECREATE:+--clear}
fi
$UV pip install --python "$VENV" --torch-backend=cu128 "$WHEEL" "$COSMOS_SPEC"

# Verification has to be GPU-free: odin has no NVIDIA driver, so `vllm --version` cannot
# work here (it dies on libcuda.so.1, or on "Failed to infer device type"). Check the
# things that are actually decidable on a login node — the CUDA *runtime* linkage is the
# one that was broken, and it is visible statically.
echo
echo "== verifying (login node — GPU checks happen in the serve job) =="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md, pathlib, subprocess, sys, vllm
for p in ("vllm", "torch", "vllm-cosmos3"):
    print(f"  {p:14s} {md.version(p)}")
so = pathlib.Path(vllm.__file__).parent / "_C.abi3.so"
needed = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
cudart = [l.split("[")[1].rstrip("]") for l in needed.splitlines() if "libcudart" in l]
print(f"  _C.abi3.so links {cudart}")
if cudart != ["libcudart.so.12"]:
    sys.exit(f"FAIL: expected libcudart.so.12, got {cudart} — wrong CUDA build for thor.")
import vllm_cosmos3  # noqa: F401
print("  vllm_cosmos3 import OK")
PY

echo
echo "OK. Next: sbatch slurm/serve_cosmos3.sbatch"
