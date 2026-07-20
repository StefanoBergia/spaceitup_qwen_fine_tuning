"""Tests for the multi-base-model plumbing (Qwen3.5-2B vs 0.8B re-run).

CPU-only: no model weights are loaded. The LoRA-target check is exercised against
a tiny stand-in module tree, and the eval-tree overlay against fixture metrics.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from rover_vlm.training import LORA_TARGETS_TEXT, unmatched_lora_targets

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    """Import a scripts/*.py module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- LoRA targets


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2)
        self.gate_proj = torch.nn.Linear(2, 2)


class _Tree(torch.nn.Module):
    """Stand-in for a decoder stack: nested names, so suffix matching is exercised."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(), _Block()])


def test_unmatched_targets_empty_when_all_present():
    assert unmatched_lora_targets(_Tree(), ["q_proj", "gate_proj"]) == []


def test_unmatched_targets_reports_missing():
    # the failure this guards: PEFT silently ignores a target that matches nothing
    assert unmatched_lora_targets(_Tree(), ["q_proj", "not_a_module"]) == ["not_a_module"]


def test_unmatched_targets_does_not_match_substrings():
    # "proj" must NOT match "q_proj" — PEFT matches on dot-delimited suffixes only
    assert unmatched_lora_targets(_Tree(), ["proj"]) == ["proj"]


# ------------------------------------------------------------- model-id default


@pytest.mark.parametrize("script", ["train", "evaluate"])
def test_model_id_defaults_to_2b(script, monkeypatch):
    """Existing commands must keep resolving to the 2B, unchanged."""
    module = load_script(script)
    argv = ["prog"] + (["--tag", "t"] if script == "evaluate" else [])
    monkeypatch.setattr(sys, "argv", argv)
    assert module.parse_args().model_id == "Qwen/Qwen3.5-2B"


# --------------------------------------------------------------- tree overlay


def write_tree(root: Path, tags_metrics: dict[str, dict]) -> Path:
    for tag, metrics in tags_metrics.items():
        d = root / tag
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"tag": tag, **metrics}))
    return root


@pytest.fixture
def trees(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"splits": {"train_full": 3660}}))
    primary = write_tree(tmp_path / "eval_a", {
        "habitat_base": {"model_id": "Qwen/Qwen3.5-0.8B", "parse_rate": 0.9, "mean_point_error_mean": 0.5},
        "habitat_train_500": {"model_id": "Qwen/Qwen3.5-0.8B", "parse_rate": 1.0, "mean_point_error_mean": 0.2},
        "habitat_train_full": {"model_id": "Qwen/Qwen3.5-0.8B", "parse_rate": 1.0, "mean_point_error_mean": 0.15},
    })
    comparison = write_tree(tmp_path / "eval_b", {
        "habitat_base": {"parse_rate": 0.95, "mean_point_error_mean": 0.44},
        "habitat_train_500": {"parse_rate": 1.0, "mean_point_error_mean": 0.18},
        "habitat_train_full": {"parse_rate": 1.0, "mean_point_error_mean": 0.12},
    })
    return primary, comparison, meta


def test_load_runs_reads_sizes_and_base(trees):
    ce = load_script("compare_evals")
    primary, _, meta = trees
    base, scaling = ce.load_runs(primary, meta)
    assert base["tag"] == "habitat_base"
    assert [size for size, _ in scaling] == [500, 3660]  # train_full resolved via meta


def test_model_label_prefers_recorded_model_id(trees):
    ce = load_script("compare_evals")
    primary, comparison, meta = trees
    base, scaling = ce.load_runs(primary, meta)
    assert ce.model_label(base, scaling, primary) == "0.8B"
    # a tree predating model_id recording falls back to its dir name, not a guess
    c_base, c_scaling = ce.load_runs(comparison, meta)
    assert ce.model_label(c_base, c_scaling, comparison) == "eval_b"


def test_single_group_table_has_no_base_model_column(trees):
    ce = load_script("compare_evals")
    primary, _, meta = trees
    base, scaling = ce.load_runs(primary, meta)
    table = ce.write_table([("0.8B", base, scaling)])
    assert "Base model" not in table
    assert table.splitlines()[0].startswith("| Model | Train size |")


def test_overlay_table_groups_both_models(trees):
    ce = load_script("compare_evals")
    primary, comparison, meta = trees
    groups = [
        ("0.8B", *ce.load_runs(primary, meta)),
        ("2B", *ce.load_runs(comparison, meta)),
    ]
    table = ce.write_table(groups)
    assert table.splitlines()[0].startswith("| Base model | Model | Train size |")
    body = table.splitlines()[2:]
    assert sum(row.startswith("| 0.8B |") for row in body) == 3
    assert sum(row.startswith("| 2B |") for row in body) == 3


def test_overlay_never_writes_to_comparison_tree(trees, monkeypatch):
    ce = load_script("compare_evals")
    primary, comparison, meta = trees
    before = {p: p.stat().st_mtime_ns for p in comparison.rglob("*") if p.is_file()}

    monkeypatch.setattr(sys, "argv", [
        "prog", "--eval-dir", str(primary), "--compare-dir", str(comparison),
        "--compare-label", "2B", "--meta", str(meta),
    ])
    ce.main()

    assert (primary / "comparison.md").exists()
    assert (primary / "scaling_curve.png").exists()
    assert not (comparison / "comparison.md").exists()
    assert not (comparison / "scaling_curve.png").exists()
    assert {p: p.stat().st_mtime_ns for p in comparison.rglob("*") if p.is_file()} == before
