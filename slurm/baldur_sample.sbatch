#!/bin/bash
#SBATCH --job-name=baldur-sample
#SBATCH --partition=RTX
#SBATCH --nodelist=baldur
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
# Sample job for baldur (RTX partition: 8x RTX 2080 Ti, 11 GB each, 64 CPUs, 375 GB RAM). Runs any repo script on one
# card through uv. Submit from the repo root; the script and its arguments follow the sbatch file:
#
#     sbatch slurm/baldur_sample.sbatch scripts/evaluate.py --task habitat --tag x --eval-file ... --batch-size 4
#     sbatch --gres=gpu:2080_ti:2 slurm/baldur_sample.sbatch scripts/...     # more cards: override on the command line
#
# What the 2080 Ti (Turing, compute capability 7.5) cannot do, so check before moving a thor job here:
#   - no bf16: load models in fp16 (scripts/evaluate.py falls back automatically; training in fp16 can overflow)
#   - no FlashAttention 2, no FP8 / Marlin kernels: vLLM FP8 models (Qwen3.6-35B-A3B-FP8, the route checker) do not run
#   - 11 GB per card: Qwen3.5-2B inference fits (batch ~4); 35 GB models do not, even split across cards
# GRES name is "2080_ti" since baldur's 2026-11 reconfig (older jobs used "2080ti", which no longer matches).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm
[ $# -ge 1 ] || { echo "usage: sbatch slurm/baldur_sample.sbatch <script.py> [args...]"; exit 1; }
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset VIRTUAL_ENV
UV="${UV:-$HOME/.local/bin/uv}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
    || echo "WARN nvidia-smi failed on at least one allocated card (continuing)"
"$UV" run python -c 'import torch; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", ok, *([torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0), "bf16", torch.cuda.is_bf16_supported()] if ok else []))'

echo "run: $*"
"$UV" run "$@"
echo "done $(date -Is)"
