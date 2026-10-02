"""Dry tests of slurm/cosmos3_serve_lib.sh and the label_descriptions.sbatch syntax.

The lib is sourced in a bash subprocess with a fake $VENV/bin/vllm (records its argv, then
sleeps) and a fake nvidia-smi; no server is started and no GPU is touched.
"""
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "slurm" / "cosmos3_serve_lib.sh"
COSMOS_FLAGS = ["--hf-overrides", "--mm-encoder-tp-mode"]
OTHER_FLAGS = ["--tensor-parallel-size", "--async-scheduling", "--allowed-local-media-path",
               "--media-io-kwargs", "--max-model-len", "--host", "--port"]


@pytest.fixture
def fake(tmp_path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    vllm = tmp_path / "venv" / "bin" / "vllm"
    vllm.write_text(f'#!/bin/bash\necho "$@" > {tmp_path}/args.txt\nexec sleep 60\n')
    vllm.chmod(0o755)
    (tmp_path / "bin").mkdir()
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 0\n")
    smi.chmod(0o755)
    return tmp_path


def run(fake, model, body):
    env = {**os.environ, "PATH": f"{fake}/bin:{os.environ['PATH']}", "VENV": str(fake / "venv"),
           "PORT": "18999", "ENDPOINT_FILE": str(fake / "endpoint.txt")}
    if model:
        env["MODEL"] = model
    return subprocess.run(["bash", "-c", f'source "{LIB}"; {body}'], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=30)


def start_args(fake, model):
    r = run(fake, model, "cosmos3_start_server >/dev/null; sleep 0.5; cosmos3_stop_server")
    assert r.returncode == 0, r.stderr
    return (fake / "args.txt").read_text()


def test_cosmos_model_gets_cosmos_flags(fake):
    args = start_args(fake, "nvidia/Cosmos3-Nano")
    assert args.startswith("serve nvidia/Cosmos3-Nano")
    for f in COSMOS_FLAGS + OTHER_FLAGS:
        assert f in args
    assert "Cosmos3ReasonerForConditionalGeneration" in args


def test_default_model_is_cosmos(fake):
    assert "--hf-overrides" in start_args(fake, None)


def test_gemma_model_drops_cosmos_flags_only(fake):
    args = start_args(fake, "google/gemma-3-12b-it")
    assert args.startswith("serve google/gemma-3-12b-it")
    for f in COSMOS_FLAGS:
        assert f not in args
    for f in OTHER_FLAGS:
        assert f in args


def test_model_is_read_at_start_time(fake):
    # one job switches MODEL between servers after the lib was sourced
    body = ('MODEL=google/gemma-3-12b-it; cosmos3_start_server >/dev/null; sleep 0.5; '
            'cosmos3_stop_server; cp "$VENV/../args.txt" "$VENV/../first.txt"; '
            'MODEL=nvidia/Cosmos3-Nano; cosmos3_start_server >/dev/null; sleep 0.5; cosmos3_stop_server')
    assert run(fake, None, body).returncode == 0
    assert "--hf-overrides" not in (fake / "first.txt").read_text()
    assert "--hf-overrides" in (fake / "args.txt").read_text()


def test_stop_server_kills_and_removes_endpoint(fake):
    body = ('cosmos3_start_server >/dev/null; echo x > "$ENDPOINT_FILE"; '
            'echo $SERVER_PID > "$VENV/../pid"; cosmos3_stop_server; '
            '[ ! -e "$ENDPOINT_FILE" ] && echo gone')
    r = run(fake, None, body)
    assert r.returncode == 0 and "gone" in r.stdout
    pid = int((fake / "pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_stop_server_without_server_is_noop(fake):
    assert run(fake, None, "cosmos3_stop_server").returncode == 0


@pytest.mark.parametrize("name", ["cosmos3_serve_lib.sh", "label_descriptions.sbatch"])
def test_bash_syntax(name):
    r = subprocess.run(["bash", "-n", str(ROOT / "slurm" / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _run_sbatch(fake, env_extra):
    """Run the real sbatch in a scratch copy with fake uv/vllm/curl; returns (rc, log lines)."""
    work = fake / "work"
    (work / "slurm").mkdir(parents=True)
    for n in ("cosmos3_serve_lib.sh", "label_descriptions.sbatch"):
        (work / "slurm" / n).write_text((ROOT / "slurm" / n).read_text())
    (work / "data").mkdir()
    for s in ("eval", "train"):
        for f in ("events", "objects"):
            (work / "data" / f"{f}_{s}.jsonl").write_text("")
    (work / "outputs" / "instr_labels_v3").mkdir(parents=True)
    (work / "outputs" / "instr_labels_v3" / "x_d1.jsonl").write_text("")  # so the final wc has a file
    log = fake / "calls.log"
    uv = fake / "bin" / "uv"
    uv.write_text(f'#!/bin/bash\necho "UV $*" >> {log}\n[ -n "$FAIL_GATE" ] && [[ "$*" == *--gate* ]] && exit 2\nexit 0\n')
    vllm = fake / "venv" / "bin" / "vllm"
    vllm.write_text(f'#!/bin/bash\necho "SERVE $2" >> {log}\nexec sleep 60\n')
    for n in ("curl",):
        (fake / "bin" / n).write_text("#!/bin/bash\nexit 0\n")
    for p in (uv, fake / "bin" / "curl"):
        p.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}/bin:{os.environ['PATH']}", "VENV": str(fake / "venv"),
           "PORT": "18998", "UV": str(uv), "DATA": "data", "SLURM_SUBMIT_DIR": str(work), **env_extra}
    env.pop("NAMER", None) if "NAMER" not in env_extra else None
    r = subprocess.run(["bash", str(work / "slurm" / "label_descriptions.sbatch")], env=env,
                       capture_output=True, text=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls


def _outs(calls):
    return [c.split("--out ")[1].split()[0].rsplit("/", 1)[1] for c in calls if c.startswith("UV") and "--out" in c]


def test_sbatch_default_namer_unchanged(fake):
    r, calls = _run_sbatch(fake, {})
    assert r.returncode == 0, r.stderr
    assert [c for c in calls if c.startswith("SERVE")] == ["SERVE nvidia/Cosmos3-Nano"]
    assert _outs(calls) == ["names_gate_d1.jsonl"] * 2 + ["desc_gate_d1.jsonl"] * 2
    assert sum("--gate" in c for c in calls) == 1 and "--gate" in calls[-1]


def test_sbatch_gemma_gate_switches_server(fake):
    r, calls = _run_sbatch(fake, {"NAMER": "gemma", "OVERWRITE": "1"})
    assert r.returncode == 0, r.stderr
    assert [c for c in calls if c.startswith("SERVE")] == ["SERVE google/gemma-3-12b-it", "SERVE nvidia/Cosmos3-Nano"]
    # names strictly before the switch, compose after
    kinds = [("S" if c.startswith("SERVE") else "n" if "--stage name" in c else "c") for c in calls]
    assert "".join(kinds) == "Snn" + "Scc"
    assert _outs(calls) == ["names_gate_gemma_d1.jsonl"] * 2 + ["desc_gate_gemma_d1.jsonl"] * 2
    assert "--overwrite" in calls[1] and "--overwrite" not in calls[2]


def test_sbatch_gemma_full_names_both_splits_then_one_switch(fake):
    r, calls = _run_sbatch(fake, {"NAMER": "gemma", "STAGE": "full"})
    assert r.returncode == 0, r.stderr
    kinds = "".join("S" if c.startswith("SERVE") else "n" if "--stage name" in c else "c" for c in calls)
    assert kinds == "S" + "nnnn" + "S" + "cccc"
    assert all("_gemma_d1" in o for o in _outs(calls))


def test_sbatch_failed_gate_stops_job_and_server(fake):
    r, calls = _run_sbatch(fake, {"NAMER": "gemma", "FAIL_GATE": "1"})
    assert r.returncode == 2
    assert not (fake / "work" / "outputs" / "cosmos3_endpoint.txt").exists()
    assert "gate labelled" not in r.stdout


def test_sbatch_rejects_unknown_namer(fake):
    r, calls = _run_sbatch(fake, {"NAMER": "bogus"})
    assert r.returncode == 1 and not calls
