"""Reasoning-trace training path: the merge (scripts/prepare_habitat_traces.py) and the
collator's <think>-block masking (rover_vlm.training.TrajectoryCollator).

The collator tests need the Qwen3.5-2B processor (tokenizer + chat template, CPU-only —
NOT the model weights). They skip cleanly where it is not cached, so the suite stays a
valid CPU gate on machines without the download.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from rover_vlm.training import TrajectoryCollator, TrajectoryDataset

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- the merge (scripts/prepare_habitat_traces.py) -----------------------------------


def _prep_mod():
    spec = importlib.util.spec_from_file_location(
        "prepare_habitat_traces",
        REPO_ROOT / "scripts" / "prepare_habitat_traces.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_habitat_traces"] = module
    spec.loader.exec_module(module)
    return module


def _prepared_rec(rid):
    return {"id": rid, "image": [f"{rid}.png"], "conversations": [
        {"from": "human", "value": "<image>\nPredict the path."},
        {"from": "gpt", "value": '{"path":[[0.5,0.9,1]],"goal":[0.5,0.6,0]}'}]}


def _kept_row(rid, trace, keep=True):
    return {"id": rid, "trace": trace, "keep": keep, "drop_reason": None if keep else "leak"}


def test_load_kept_traces_filters_keep_and_tags(tmp_path):
    pm = _prep_mod()
    f = tmp_path / "t.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in [
        _kept_row("a", "clear tiles ahead"),
        _kept_row("b", "dropped one", keep=False),          # not kept
        _kept_row("c", ""),                                 # empty
        _kept_row("d", "has a </think> tag inside"),        # tag-contaminated
    ]))
    traces, skipped_tags, skipped_empty = pm.load_kept_traces(f)
    assert set(traces) == {"a"}
    assert skipped_tags == 1 and skipped_empty == 1


def test_merge_drops_untraced_by_default(tmp_path, monkeypatch):
    pm = _prep_mod()
    prepared = tmp_path / "train.json"
    prepared.write_text(json.dumps([_prepared_rec("a"), _prepared_rec("b")]))
    traces = tmp_path / "tr.jsonl"
    traces.write_text(json.dumps(_kept_row("a", "clear tiles ahead")) + "\n")
    out = tmp_path / "out.json"

    monkeypatch.setattr(sys, "argv",
                        ["x", "--prepared", str(prepared), "--traces", str(traces),
                         "--out", str(out)])
    pm.main()
    recs = json.loads(out.read_text())
    assert [r["id"] for r in recs] == ["a"]              # b dropped (no trace)
    assert recs[0]["reasoning"] == "clear tiles ahead"


def test_merge_keep_untraced_includes_answer_only(tmp_path, monkeypatch):
    pm = _prep_mod()
    prepared = tmp_path / "train.json"
    prepared.write_text(json.dumps([_prepared_rec("a"), _prepared_rec("b")]))
    traces = tmp_path / "tr.jsonl"
    traces.write_text(json.dumps(_kept_row("a", "clear tiles ahead")) + "\n")
    out = tmp_path / "out.json"

    monkeypatch.setattr(sys, "argv",
                        ["x", "--prepared", str(prepared), "--traces", str(traces),
                         "--out", str(out), "--keep-untraced"])
    pm.main()
    recs = {r["id"]: r for r in json.loads(out.read_text())}
    assert set(recs) == {"a", "b"}
    assert recs["a"]["reasoning"] == "clear tiles ahead"
    assert "reasoning" not in recs["b"]                 # untraced stays answer-only


# --- the collator mask (needs the processor) -----------------------------------------


@pytest.fixture(scope="module")
def processor():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
    except Exception as e:  # not cached / offline
        pytest.skip(f"Qwen3.5-2B processor unavailable: {e}")


def _sample(reasoning=None):
    return {"id": "s", "image": Image.new("RGB", (56, 56), "gray"),
            "prompt": "Predict the path.",
            "answer": '{"path":[[0.5,0.9,1]],"goal":[0.5,0.6,0]}',
            "reasoning": reasoning}


def _trained_text(processor, batch, i):
    """Decode only the label positions that are NOT masked (-100) for sample i."""
    ids = batch["input_ids"][i]
    kept = [int(t) for t, lab in zip(ids, batch["labels"][i]) if int(lab) != -100]
    return processor.tokenizer.decode(kept)


def test_collator_trains_on_reasoning_when_present(processor):
    reasoning = "I roll forward over clear tiles toward the open doorway."
    coll = TrajectoryCollator(processor)
    batch = coll([_sample(reasoning=reasoning)])
    trained = _trained_text(processor, batch, 0)
    # the reasoning AND the answer are in the loss; the </think> boundary is too
    assert "clear tiles toward the open doorway" in trained
    assert '"path"' in trained and "</think>" in trained
    # the prompt text is masked out (not trained on)
    assert "Predict the path" not in trained


def test_collator_answer_only_excludes_reasoning(processor):
    coll = TrajectoryCollator(processor)
    batch = coll([_sample(reasoning=None)])
    trained = _trained_text(processor, batch, 0)
    assert '"path"' in trained                 # answer still trained
    assert "Predict the path" not in trained   # prompt masked
    # no reasoning was supplied, so none is in the loss beyond the empty think scaffold
    assert "clear tiles" not in trained


def test_collator_mixed_batch_aligns(processor):
    """A batch mixing traced and untraced samples must mask each at its own boundary
    (the alignment guard raises if a boundary slips)."""
    coll = TrajectoryCollator(processor)
    batch = coll([_sample(reasoning="turning left around the sofa"), _sample(reasoning=None)])
    assert "turning left around the sofa" in _trained_text(processor, batch, 0)
    assert "turning left around the sofa" not in _trained_text(processor, batch, 1)


def test_dataset_surfaces_reasoning_field(tmp_path):
    rec = _prepared_rec("a")
    rec["reasoning"] = "clear tiles ahead"
    plain = _prepared_rec("b")
    (tmp_path / "d.json").write_text(json.dumps([rec, plain]))
    # image files so __getitem__ can open them
    for rid in ("a", "b"):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{rid}.png")
    ds = TrajectoryDataset(tmp_path / "d.json", tmp_path)
    assert ds[0]["reasoning"] == "clear tiles ahead"
    assert ds[1]["reasoning"] is None
