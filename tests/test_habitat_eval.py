import pytest

from rover_vlm.eval import parse_path_answer, habitat_metrics, aggregate_habitat_metrics


def test_parse_clean_object():
    out = parse_path_answer('{"path":[[0.4,0.8,1],[0.4,0.5,0]],"goal":[0.5,0.6,0]}')
    assert out["path"] == [[0.4, 0.8, 1], [0.4, 0.5, 0]]
    assert out["goal"] == [0.5, 0.6, 0]


def test_parse_with_surrounding_text():
    out = parse_path_answer('Here: {"path": [[0.1,0.2,1],[0.3,0.4,1]], "goal": [0.5,0.6,1]} done')
    assert out is not None and len(out["path"]) == 2 and out["goal"][2] == 1


def test_parse_garbage_returns_none():
    assert parse_path_answer("no coordinates here") is None


def test_parse_ignores_distractor_braces_and_prose_triples():
    text = ('Reasoning: the map {rock: [0.9,0.9,1]} suggests going left. '
            '{"path":[[0.1,0.2,1],[0.3,0.4,1]],"goal":[0.5,0.6,1]}')
    assert parse_path_answer(text) == {"path": [[0.1, 0.2, 1], [0.3, 0.4, 1]], "goal": [0.5, 0.6, 1]}


def test_parse_regex_fallback_when_no_valid_object():
    # no JSON object with path/goal keys, but bracketed triples present -> fallback
    text = "coords (0.1,0.2,1) then (0.3,0.4,0) then (0.5,0.6,0)"
    assert parse_path_answer(text) == {"path": [[0.1, 0.2, 1], [0.3, 0.4, 0]], "goal": [0.5, 0.6, 0]}


def test_habitat_metrics_perfect_match():
    gt = {"path": [[0.5, 0.9, 1], [0.5, 0.5, 0]], "goal": [0.5, 0.3, 0]}
    m = habitat_metrics(gt, gt)
    assert m["mean_point_error"] < 1e-9
    assert m["path_visibility_acc"] == 1.0
    assert m["goal_point_error"] < 1e-9
    assert m["goal_visibility_correct"] == 1


def test_habitat_metrics_wrong_goal_visibility():
    gt = {"path": [[0.5, 0.9, 1], [0.5, 0.5, 1]], "goal": [0.5, 0.3, 1]}
    pred = {"path": [[0.5, 0.9, 1], [0.5, 0.5, 1]], "goal": [0.5, 0.3, 0]}
    m = habitat_metrics(pred, gt)
    assert m["goal_visibility_correct"] == 0


def test_aggregate_reports_rates():
    recs = [
        {"parsed": {"path": [[0.5, 0.9, 1]], "goal": [0.5, 0.3, 1]},
         "metrics": {"mean_point_error": 0.1, "frechet": 0.2, "path_visibility_acc": 1.0,
                     "goal_point_error": 0.05, "goal_visibility_correct": 1}},
        {"parsed": None, "metrics": None},
    ]
    agg = aggregate_habitat_metrics(recs)
    assert agg["parse_rate"] == 0.5
    assert agg["goal_visibility_accuracy"] == 1.0  # over parsed only


# --- real-image eval diagnostics ------------------------------------------------------

def test_signed_lateral_direction():
    from rover_vlm.eval import _signed_lateral
    right = [[0.5, 1.0, 1], [0.6, 0.8, 1], [0.7, 0.6, 1]]
    left = [[0.5, 1.0, 1], [0.4, 0.8, 1], [0.3, 0.6, 1]]
    straight = [[0.5, 1.0, 1], [0.5, 0.8, 1], [0.5, 0.6, 1]]
    assert _signed_lateral(right) > 0 and _signed_lateral(left) < 0
    assert _signed_lateral(straight) == pytest.approx(0.0)
    # mirrored paths must give opposite signs, not the same magnitude twice
    assert _signed_lateral(right) == pytest.approx(-_signed_lateral(left))


def test_spearman_and_constant_input():
    from rover_vlm.eval import spearman
    import math
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))  # no ranks to correlate


def _rec(pred_path, gt_path):
    return {"parsed": {"path": pred_path, "goal": pred_path[-1]},
            "gt": {"path": gt_path, "goal": gt_path[-1]}}


def test_shape_correlation_detects_image_blind_model():
    from rover_vlm.eval import shape_correlation
    import math
    bends = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    def path(b):
        return [[0.5 + b * i / 4, 1.0 - 0.1 * i, 1] for i in range(5)]

    tracking = [_rec(path(b), path(b)) for b in bends]
    assert shape_correlation(tracking) == pytest.approx(1.0)
    inverted = [_rec(path(-b), path(b)) for b in bends]
    assert shape_correlation(inverted) == pytest.approx(-1.0)
    canned = [_rec(path(0.0), path(b)) for b in bends]  # same answer every frame
    assert math.isnan(shape_correlation(canned))


def test_trivial_baselines_reward_a_degenerate_set():
    from rover_vlm.eval import trivial_baselines
    # every route runs straight up the middle -> an image-blind guess is nearly perfect
    straight_set = [{"path": [[0.5, 1.0 - 0.05 * i, 1] for i in range(10)], "goal": [0.5, 0.55, 1]}
                    for _ in range(20)]
    b = trivial_baselines(straight_set)
    assert b["straight"]["mean_point_error_median"] < 0.01
    assert b["constant"]["mean_point_error_median"] < 0.01
    # routes fanning left and right -> no single answer fits, so the floor is high
    varied = [{"path": [[0.5 + s * 0.04 * i, 1.0 - 0.05 * i, 1] for i in range(10)],
               "goal": [0.5 + s * 0.36, 0.55, 1]} for s in (-1, 1) for _ in range(10)]
    assert trivial_baselines(varied)["constant"]["mean_point_error_median"] > 0.1


def test_endpoint_spread_catches_a_pinned_terminus():
    from rover_vlm.eval import endpoint_spread
    # what Habitat training produces: the route bends, but always ends at x = 0.5
    def funnel(b):
        return [[0.5 + b * (4 - i) / 4, 1.0 - 0.1 * i, 1] for i in range(5)]

    def spread(b):
        return [[0.5 + b * i / 4, 1.0 - 0.1 * i, 1] for i in range(5)]

    bends = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    collapsed = endpoint_spread([_rec(funnel(b), spread(b)) for b in bends])
    assert collapsed["n"] == len(bends)
    assert collapsed["pred_end_x_sd"] == pytest.approx(0.0)
    assert collapsed["pred_end_x_mode_frac"] == pytest.approx(1.0)
    # the ground truth it is scored against is not pinned -- that contrast is the finding
    assert collapsed["gt_end_x_sd"] > 0.1
    assert collapsed["gt_end_x_mode_frac"] < 0.5

    healthy = endpoint_spread([_rec(spread(b), spread(b)) for b in bends])
    assert healthy["pred_end_x_sd"] == pytest.approx(healthy["gt_end_x_sd"])
    assert healthy["pred_end_x_mode_frac"] < 0.5


def test_endpoint_spread_survives_an_empty_run():
    from rover_vlm.eval import endpoint_spread
    import math
    # unparseable generations leave no `parsed`, so a run can reduce to nothing
    out = endpoint_spread([{"parsed": None, "gt": {"path": [[0.5, 1.0, 1]]}}])
    assert out["n"] == 0
    assert math.isnan(out["pred_end_x_sd"])


# --- round 3: goal handed in the prompt, answer is path-only ------------------------

def test_parse_accepts_path_only_answer():
    """Round-3 answers carry no "goal" key; it is the final waypoint."""
    out = parse_path_answer('{"path":[[0.4,0.99,1],[0.35,0.62,0]]}')
    assert out["path"] == [[0.4, 0.99, 1], [0.35, 0.62, 0]]
    assert out["goal"] == [0.35, 0.62, 0]


def test_parse_still_prefers_an_explicit_goal_key():
    """Legacy answers must keep parsing byte-identically, goal key winning over path[-1]."""
    out = parse_path_answer('{"path":[[0.4,0.99,1],[0.35,0.62,0]],"goal":[0.5,0.58,1]}')
    assert out["goal"] == [0.5, 0.58, 1]


def test_parse_rejects_empty_path_object():
    assert parse_path_answer('{"path":[]}') is None


def test_goal_copy_correct_flags_a_reproduced_endpoint():
    gt = {"path": [[0.4, 0.99, 1], [0.35, 0.62, 0]], "goal": [0.35, 0.62, 0]}
    copied = habitat_metrics({"path": [[0.5, 0.99, 1], [0.35, 0.62, 0]],
                              "goal": [0.35, 0.62, 0]}, gt)
    drifted = habitat_metrics({"path": [[0.5, 0.99, 1], [0.60, 0.62, 0]],
                               "goal": [0.60, 0.62, 0]}, gt)
    assert copied["goal_copy_correct"] == 1
    assert drifted["goal_copy_correct"] == 0


def test_to_goal_baseline_is_the_hardest_floor():
    """With the endpoint given, a blind straight-to-goal line is the bar to beat --
    it must be tighter than the set-level floors on curved routes."""
    from rover_vlm.eval import trivial_baselines
    gts = [{"path": [[0.5, 1.0, 1], [0.30, 0.75, 1], [0.20, 0.5, 0]], "goal": [0.20, 0.5, 0]},
           {"path": [[0.5, 1.0, 1], [0.70, 0.75, 1], [0.80, 0.5, 0]], "goal": [0.80, 0.5, 0]}]
    base = trivial_baselines(gts)
    assert set(base) == {"straight", "constant", "to_goal"}
    assert base["to_goal"]["mean_point_error_mean"] < base["straight"]["mean_point_error_mean"]
    assert base["to_goal"]["mean_point_error_mean"] < base["constant"]["mean_point_error_mean"]


def test_to_goal_is_not_zero_on_a_bent_route():
    """A route that bows away from the chord must cost the straight-line baseline
    something, otherwise the baseline could never be beaten meaningfully."""
    from rover_vlm.eval import trivial_baselines
    gts = [{"path": [[0.5, 1.0, 1], [0.9, 0.75, 1], [0.5, 0.5, 0]], "goal": [0.5, 0.5, 0]}]
    assert trivial_baselines(gts)["to_goal"]["mean_point_error_mean"] > 0.05
