import importlib.util
import sys
from pathlib import Path

import pytest


def _eval_mod():
    spec = importlib.util.spec_from_file_location(
        "evaluate", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    return module


def test_greedy_run_lands_in_tag_root():
    ev = _eval_mod()
    assert ev.run_out_dir(Path("outputs/eval"), "habitat_base", None) == \
        Path("outputs/eval/habitat_base")


def test_seeded_run_lands_under_seeds_subdir():
    ev = _eval_mod()
    assert ev.run_out_dir(Path("outputs/eval"), "habitat_base", 3) == \
        Path("outputs/eval/habitat_base/seeds/seed3")


def test_no_seed_means_greedy_decoding():
    ev = _eval_mod()
    assert ev.generation_kwargs(None, 0.7, 0.8, 20) == {"do_sample": False}


def test_seed_turns_on_sampling_with_given_params():
    ev = _eval_mod()
    assert ev.generation_kwargs(1, 0.7, 0.8, 20) == \
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def test_seed_with_zero_temperature_is_rejected():
    ev = _eval_mod()
    with pytest.raises(ValueError):
        ev.generation_kwargs(1, 0.0, 0.8, 20)
