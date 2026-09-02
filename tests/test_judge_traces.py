"""Pure-Python parts of the ground-truth judge (scripts/judge_traces.py): reasoning extraction,
GT summary, verdict parsing, and aggregation. The GPU generation path is not exercised here."""

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


def test_gt_summary_direction_and_occlusion():
    jm = _judge_mod()
    # goal well to the left of the start, and hidden
    g = jm.gt_summary({"path": [[0.9, 1.0, 1], [0.5, 0.6, 0]], "goal": [0.3, 0.55, 0]})
    assert g["direction"] == "to the left" and g["goal_hidden"] is True
    assert g["n_occluded"] == 1 and g["n_path"] == 2
    assert "hidden" in g["goalstate"]
    # straight-ahead, visible goal
    g2 = jm.gt_summary({"path": [[0.5, 1.0, 1]], "goal": [0.5, 0.6, 1]})
    assert g2["direction"] == "roughly straight ahead" and g2["goal_hidden"] is False
    assert jm.gt_summary(None) is None and jm.gt_summary({}) is None


def test_parse_verdict_valid_and_bool_coercion():
    jm = _judge_mod()
    v = jm.parse_verdict('ok {"direction_correct": true, "occlusion_correct": false, '
                         '"contradicts_gt": false, "score": 4} end')
    assert v == {"direction_correct": True, "occlusion_correct": False,
                 "contradicts_gt": False, "score": 4}
    # string/int booleans coerce; score clamps
    v2 = jm.parse_verdict('{"direction_correct": "yes", "occlusion_correct": 0, '
                          '"contradicts_gt": "false", "score": 9}')
    assert v2["direction_correct"] is True and v2["occlusion_correct"] is False
    assert v2["contradicts_gt"] is False and v2["score"] == 5


def test_parse_verdict_rejects_bad():
    jm = _judge_mod()
    assert jm.parse_verdict("no json") is None
    assert jm.parse_verdict('{"direction_correct": true, "score": 3}') is None       # missing keys
    assert jm.parse_verdict('{"direction_correct": true, "occlusion_correct": true, '
                            '"contradicts_gt": false}') is None                       # no score


def test_judge_slug():
    jm = _judge_mod()
    assert jm.judge_slug("google/gemma-3-12b-it") == "gemma-3-12b-it"
    assert jm.judge_slug("nvidia/Cosmos-Reason2-8B") == "cosmos-reason2-8b"
    assert jm.judge_slug("bare-model") == "bare-model"
    assert jm.judge_slug("Org/Weird__Name!!") == "weird-name"
    assert jm.judge_slug("") == "judge" and jm.judge_slug("///") == "judge"


def test_default_out_dir_is_judge_namespaced():
    jm = _judge_mod()
    # main() builds: out_dir = args.out_dir or (eval_dir / tag / f"judge_{judge_slug(model_id)}")
    eval_dir, tag, model = Path("outputs/eval_habitat_v2_traced"), "habitat_train_full_traced", \
        "google/gemma-3-12b-it"
    out_dir = eval_dir / tag / f"judge_{jm.judge_slug(model)}"
    assert out_dir == eval_dir / tag / "judge_gemma-3-12b-it"
    # a second judge lands in a distinct dir, so results coexist rather than clobber
    cosmos = eval_dir / tag / f"judge_{jm.judge_slug('nvidia/Cosmos-Reason2-8B')}"
    assert cosmos != out_dir


def test_aggregate_accuracies_and_contradiction_rate():
    jm = _judge_mod()
    rows = [
        {"id": "a", "verdict": {"direction_correct": True, "occlusion_correct": True,
                                "contradicts_gt": False, "score": 5}},
        {"id": "b", "verdict": {"direction_correct": True, "occlusion_correct": False,
                                "contradicts_gt": True, "score": 2}},
        {"id": "c", "verdict": None},                                                # unparsed
    ]
    a = jm.aggregate(rows)
    assert a["n_total"] == 3 and a["n_scored"] == 2
    assert a["direction_accuracy"] == 1.0 and a["occlusion_accuracy"] == 0.5
    assert a["contradiction_rate"] == 0.5 and a["score_mean"] == 3.5
