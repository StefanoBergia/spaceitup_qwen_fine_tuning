import math

import pytest

from rover_vlm.consistency import (
    aggregate_consistency,
    per_image_consistency,
    sample_spread,
    spearman,
    start_edge,
)


def _pred(dx=0.0, goal_v=1):
    return {"path": [[0.5 + dx, 0.9, 1], [0.5 + dx, 0.6, 1], [0.5 + dx, 0.3, 1]],
            "goal": [0.5 + dx, 0.2, goal_v]}


def _entering_from(x0, y0):
    return {"path": [[x0, y0, 1], [0.5, 0.6, 1]], "goal": [0.5, 0.55, 0]}


def test_identical_draws_have_zero_spread_and_full_agreement():
    s = sample_spread([_pred(), _pred(), _pred()])
    assert s["n_draws"] == 3 and s["n_parsed"] == 3
    assert s["path_spread"] == 0.0
    assert s["goal_spread"] == 0.0
    assert s["goal_vis_agreement"] == 1.0
    assert s["start_edge_agreement"] == 1.0


def test_start_edge_classifies_borders_with_bottom_winning_corners():
    assert start_edge(_entering_from(0.0, 0.85)) == "left"
    assert start_edge(_entering_from(1.0, 0.81)) == "right"
    assert start_edge(_entering_from(0.5, 1.0)) == "bottom"
    assert start_edge(_entering_from(0.97, 1.0)) == "bottom"    # corner -> bottom
    assert start_edge(_entering_from(0.04, 0.7)) == "left"      # within tolerance
    assert start_edge(_entering_from(0.3, 0.7)) == "none"       # starts mid-image
    assert start_edge({"path": [], "goal": [0.5, 0.5, 1]}) == "none"


def test_route_flip_lowers_edge_agreement_but_jitter_does_not():
    flip = sample_spread([_entering_from(0.0, 0.85), _entering_from(1.0, 0.85),
                          _entering_from(0.0, 0.83)])
    assert flip["start_edge_agreement"] == pytest.approx(2 / 3)
    jitter = sample_spread([_entering_from(0.0, 0.85), _entering_from(0.02, 0.88)])
    assert jitter["start_edge_agreement"] == 1.0
    assert jitter["path_spread"] > 0.0


def test_shifted_draw_gives_known_spread():
    # two draws, one shifted by 0.1 in x: every resampled point is 0.1 apart
    s = sample_spread([_pred(0.0), _pred(0.1)])
    assert s["path_spread"] == pytest.approx(0.1)
    assert s["goal_spread"] == pytest.approx(0.1)


def test_pairwise_mean_over_three_draws():
    # draws at x, x+0.1, x+0.3 -> pairwise distances 0.1, 0.3, 0.2 -> mean 0.2
    s = sample_spread([_pred(0.0), _pred(0.1), _pred(0.3)])
    assert s["path_spread"] == pytest.approx(0.2)


def test_unparsed_draws_are_excluded_but_counted():
    s = sample_spread([_pred(), None, _pred()])
    assert s["n_draws"] == 3 and s["n_parsed"] == 2
    assert s["path_spread"] == 0.0


def test_fewer_than_two_parsed_draws_gives_no_spread():
    s = sample_spread([_pred(), None])
    assert s["n_parsed"] == 1
    assert s["path_spread"] is None and s["goal_spread"] is None
    assert s["goal_vis_agreement"] is None


def test_goal_visibility_agreement_is_majority_fraction():
    s = sample_spread([_pred(goal_v=1), _pred(goal_v=1), _pred(goal_v=0)])
    assert s["goal_vis_agreement"] == pytest.approx(2 / 3)


def test_goal_only_prediction_uses_goal_as_path():
    a = {"path": [], "goal": [0.5, 0.5, 1]}
    b = {"path": [], "goal": [0.5, 0.6, 1]}
    s = sample_spread([a, b])
    assert s["path_spread"] == pytest.approx(0.1)


def _rec(i, dx, err):
    return {"id": i, "parsed": _pred(dx), "metrics": {"mean_point_error": err}}


def test_per_image_merges_seeds_by_id_and_summarises_error():
    seed1 = {1: _rec(1, 0.0, 0.10), 2: _rec(2, 0.0, 0.50)}
    seed2 = {1: _rec(1, 0.1, 0.30), 2: {"id": 2, "parsed": None, "metrics": None}}
    greedy = {1: _rec(1, 0.0, 0.20), 2: _rec(2, 0.0, 0.40)}
    out = per_image_consistency([seed1, seed2], greedy)
    assert set(out) == {1, 2}
    assert out[1]["path_spread"] == pytest.approx(0.1)
    assert out[1]["err_mean"] == pytest.approx(0.2)
    assert out[1]["err_std"] == pytest.approx(0.1)
    assert out[1]["greedy_err"] == pytest.approx(0.2)
    # id 2: one seed unparsed -> single parsed draw, no spread, error stats over the one draw
    assert out[2]["n_parsed"] == 1 and out[2]["path_spread"] is None
    assert out[2]["err_mean"] == pytest.approx(0.5) and out[2]["err_std"] is None


def test_per_image_without_greedy_run():
    out = per_image_consistency([{1: _rec(1, 0.0, 0.1)}, {1: _rec(1, 0.2, 0.1)}])
    assert out[1]["greedy_err"] is None


def test_spearman_monotone_and_anticorrelated():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) == pytest.approx(1.0)
    assert spearman([1], [1]) is None
    assert spearman([1, 1, 1], [1, 2, 3]) is None   # constant input -> undefined


def test_aggregate_reports_medians_parse_rate_and_correlation():
    per_image = {
        1: {"n_draws": 2, "n_parsed": 2, "path_spread": 0.1, "goal_spread": 0.1,
            "goal_vis_agreement": 1.0, "start_edge_agreement": 1.0,
            "err_mean": 0.1, "err_std": 0.0, "greedy_err": 0.1},
        2: {"n_draws": 2, "n_parsed": 2, "path_spread": 0.3, "goal_spread": 0.2,
            "goal_vis_agreement": 0.5, "start_edge_agreement": 0.5,
            "err_mean": 0.3, "err_std": 0.1, "greedy_err": 0.3},
        3: {"n_draws": 2, "n_parsed": 1, "path_spread": None, "goal_spread": None,
            "goal_vis_agreement": None, "start_edge_agreement": None,
            "err_mean": 0.9, "err_std": None, "greedy_err": 0.9},
    }
    agg = aggregate_consistency(per_image)
    assert agg["num_images"] == 3 and agg["num_with_spread"] == 2
    assert agg["parse_rate"] == pytest.approx(5 / 6)
    assert agg["path_spread_median"] == pytest.approx(0.2)
    assert agg["path_spread_mean"] == pytest.approx(0.2)
    assert agg["goal_spread_median"] == pytest.approx(0.15)
    assert agg["goal_vis_agreement_mean"] == pytest.approx(0.75)
    assert agg["start_edge_agreement_mean"] == pytest.approx(0.75)
    assert agg["start_edge_flip_rate"] == pytest.approx(0.5)   # image 2 flipped, image 1 did not
    assert agg["err_std_mean"] == pytest.approx(0.05)
    assert agg["spread_vs_greedy_err_spearman"] == pytest.approx(1.0)


def test_aggregate_handles_no_spread_at_all():
    agg = aggregate_consistency({1: {"n_draws": 1, "n_parsed": 0, "path_spread": None,
                                     "goal_spread": None, "goal_vis_agreement": None,
                                     "err_mean": None, "err_std": None, "greedy_err": None}})
    assert agg["num_with_spread"] == 0 and agg["path_spread_median"] is None
    assert agg["spread_vs_greedy_err_spearman"] is None
    assert agg["start_edge_flip_rate"] is None
    assert math.isclose(agg["parse_rate"], 0.0)
