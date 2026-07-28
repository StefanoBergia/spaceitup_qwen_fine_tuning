"""Pure-Python parts of the VLM-judge script (scripts/judge_traces.py): reasoning extraction,
verdict parsing, and aggregation. The GPU generation path is not exercised here."""

import importlib.util
import sys
from pathlib import Path


def _judge_mod():
    spec = importlib.util.spec_from_file_location(
        "judge_traces", Path(__file__).resolve().parent.parent / "scripts" / "judge_traces.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["judge_traces"] = module
    spec.loader.exec_module(module)
    return module


def test_reasoning_of_splits_on_think_close():
    jm = _judge_mod()
    assert jm.reasoning_of("clear floor</think>{}") == "clear floor"
    assert jm.reasoning_of("no tag here") == "no tag here"
    assert jm.reasoning_of("") == ""


def test_parse_verdict_valid_clamps_and_hallucinations():
    jm = _judge_mod()
    v = jm.parse_verdict('sure: {"faithful": 4, "occlusion": 2, "coherent": 5, '
                         '"hallucinations": ["sofa"]} done')
    assert v == {"faithful": 4, "occlusion": 2, "coherent": 5, "hallucinations": ["sofa"]}
    # out-of-range values are clamped to 1..5; floats rounded
    v2 = jm.parse_verdict('{"faithful": 9, "occlusion": 0, "coherent": 3.4, "hallucinations": []}')
    assert v2["faithful"] == 5 and v2["occlusion"] == 1 and v2["coherent"] == 3


def test_parse_verdict_rejects_bad():
    jm = _judge_mod()
    assert jm.parse_verdict("no json at all") is None
    assert jm.parse_verdict('{"faithful": 4}') is None            # missing keys
    assert jm.parse_verdict('{"faithful": "x", "occlusion": 2, "coherent": 3}') is None


def test_aggregate_scores_and_hallucination_rate():
    jm = _judge_mod()
    rows = [
        {"id": "a", "verdict": {"faithful": 4, "occlusion": 2, "coherent": 5, "hallucinations": ["x"]}},
        {"id": "b", "verdict": {"faithful": 2, "occlusion": 4, "coherent": 3, "hallucinations": []}},
        {"id": "c", "verdict": None},                              # unparsed
    ]
    a = jm.aggregate(rows)
    assert a["n_total"] == 3 and a["n_scored"] == 2
    assert abs(a["faithful_mean"] - 3.0) < 1e-9 and abs(a["occlusion_mean"] - 3.0) < 1e-9
    assert a["hallucination_rate"] == 0.5
