"""Every GPU job the instruction plans hand to the user caps BLAS/OpenMP threads (CLAUDE.md)."""
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parent.parent / "slurm"


@pytest.mark.parametrize("name", ["train_habitat.sbatch", "label_instructions.sbatch",
                                  "eval_instruction_v2.sbatch", "eval_instruction.sbatch",
                                  "render_semantic_frames.sbatch"])
def test_thread_caps(name):
    text = (SLURM / name).read_text()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"{var}=" in text, f"{name} does not set {var}"
