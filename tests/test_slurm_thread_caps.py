"""Every GPU job the instruction plans hand to the user caps BLAS/OpenMP threads (CLAUDE.md)."""
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parent.parent / "slurm"


@pytest.mark.parametrize("name", ["train_habitat.sbatch", "label_instructions.sbatch",
                                  "eval_instruction_v2.sbatch", "eval_instruction.sbatch",
                                  "render_semantic_frames.sbatch", "label_descriptions.sbatch",
                                  "eval_descriptions.sbatch"])
def test_thread_caps(name):
    text = (SLURM / name).read_text()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"{var}=" in text, f"{name} does not set {var}"


def test_label_descriptions_has_a_steer_names_stage():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert '"$STAGE" = "steer_names"' in text and "--all-objects" in text
    assert "names_steer_gemma_d1.jsonl" in text and 'serve "$GEMMA_MODEL"' in text


def test_label_descriptions_has_a_steer_stage():
    text = (SLURM / "label_descriptions.sbatch").read_text()
    assert '"$STAGE" = "steer"' in text and "events_steer_eval.jsonl" in text
    assert "desc_steer_eval_d1.jsonl" in text and "names_primary_eval.jsonl" in text and "--fallback-names" in text
    assert "steer-propose" not in text
