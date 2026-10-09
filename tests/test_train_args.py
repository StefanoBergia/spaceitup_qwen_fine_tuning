import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _train():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_save_kwargs():
    t = _train()
    assert t.save_kwargs(SimpleNamespace(smoke=False, save_steps=0)) == {"save_strategy": "epoch"}
    assert t.save_kwargs(SimpleNamespace(smoke=False, save_steps=500)) == {
        "save_strategy": "steps", "save_steps": 500, "save_total_limit": 2}
    assert t.save_kwargs(SimpleNamespace(smoke=True, save_steps=500)) == {"save_strategy": "no"}


def test_resume_checkpoint(tmp_path):
    t = _train()
    assert t.resume_checkpoint(tmp_path, resume=True) is None          # nothing yet: start fresh
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-1000").mkdir()
    assert t.resume_checkpoint(tmp_path, resume=True).endswith("checkpoint-1000")
    assert t.resume_checkpoint(tmp_path, resume=False) is None
