#!/bin/bash
# HM3D semantic frames for the grounded-landmark name check (scripts/render_semantic_frames.py).
#
#     sbatch slurm/render_semantic_frames.sbatch
#
# Full A100 only: habitat's EGL renderer cannot use a MIG slice. ~6.3k frames in ~150
# scenes; resumable (existing .npz are skipped), so a timeout is fixed by re-submitting.
#SBATCH --job-name=semantic-frames
#SBATCH --partition=A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=telegram:422627024
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p outputs/slurm outputs/landmark_semantics
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/miniforge3/etc/profile.d/conda.sh
conda activate habitat
# the repo's .python-version makes pyenv's shim win over conda's `python`: call the env directly
"$CONDA_PREFIX/bin/python" -u scripts/render_semantic_frames.py "$@"
