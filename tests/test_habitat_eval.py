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
