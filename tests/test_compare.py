from rover_vlm.compare import mcnemar_exact, paired_bootstrap, significance_label


def _recs(flags, key="accepted_correct"):
    return {str(i): {"metrics": {key: v}} for i, v in enumerate(flags)}


# --- McNemar ------------------------------------------------------------------------


def test_mcnemar_counts_only_discordant_pairs():
    # both right on 0,1 / both wrong on 2 -> uninformative; 3 and 4 are the disagreements
    a = _recs([1, 1, 0, 1, 0])
    b = _recs([1, 1, 0, 0, 1])
    a_only, b_only, _ = mcnemar_exact(a, b, "accepted_correct")
    assert (a_only, b_only) == (1, 1)


def test_mcnemar_identical_models_is_not_significant():
    a = _recs([1, 0, 1, 0])
    _, _, p = mcnemar_exact(a, dict(a), "accepted_correct")
    assert p == 1.0


def test_mcnemar_lopsided_disagreement_is_significant():
    # a right on 12 that b misses, b right on none that a misses
    a = _recs([1] * 12 + [1] * 8)
    b = _recs([0] * 12 + [1] * 8)
    a_only, b_only, p = mcnemar_exact(a, b, "accepted_correct")
    assert (a_only, b_only) == (12, 0)
    assert p < 0.001


def test_mcnemar_small_lopsided_is_not_significant():
    # 3-vs-0 is lopsided but far too small to conclude anything: p = 2/2^3 = 0.25
    a, b = _recs([1, 1, 1]), _recs([0, 0, 0])
    _, _, p = mcnemar_exact(a, b, "accepted_correct")
    assert p == 0.25


def test_mcnemar_treats_missing_metrics_as_wrong():
    a = {"x": {"metrics": {"accepted_correct": 1}}}
    b = {"x": {"metrics": None}}          # unparseable prediction
    a_only, b_only, _ = mcnemar_exact(a, b, "accepted_correct")
    assert (a_only, b_only) == (1, 0)


def test_mcnemar_uses_only_shared_ids():
    a = _recs([1, 1, 1])
    b = {"0": {"metrics": {"accepted_correct": 0}}}   # only sample 0 in common
    a_only, b_only, _ = mcnemar_exact(a, b, "accepted_correct")
    assert (a_only, b_only) == (1, 0)


# --- paired bootstrap ---------------------------------------------------------------


def _err(vals):
    return {str(i): {"metrics": {"mean_point_error": v}} for i, v in enumerate(vals)}


def test_bootstrap_reports_paired_difference():
    a = _err([0.2] * 40)
    b = _err([0.1] * 40)
    r = paired_bootstrap(a, b, "mean_point_error", iters=400)
    assert r["n"] == 40
    assert abs(r["diff"] - 0.1) < 1e-9
    # a constant difference has no spread, so the interval collapses onto it
    assert abs(r["lo"] - 0.1) < 1e-9 and abs(r["hi"] - 0.1) < 1e-9


def test_bootstrap_interval_covers_zero_for_noisy_tie():
    a = _err([0.1, 0.2] * 30)
    b = _err([0.2, 0.1] * 30)
    r = paired_bootstrap(a, b, "mean_point_error", iters=600)
    assert r["lo"] < 0 < r["hi"], "a genuine tie must not look significant"


def test_bootstrap_skips_samples_either_model_failed_to_parse():
    a = _err([0.1, 0.2])
    b = _err([0.3, 0.4])
    b["1"]["metrics"] = None              # b failed on sample 1
    r = paired_bootstrap(a, b, "mean_point_error", iters=200)
    assert r["n"] == 1, "pairs must be dropped when either side has no metrics"


def test_bootstrap_is_deterministic():
    a, b = _err([0.1, 0.5, 0.2, 0.9]), _err([0.2, 0.1, 0.4, 0.3])
    r1 = paired_bootstrap(a, b, "mean_point_error", iters=300)
    r2 = paired_bootstrap(a, b, "mean_point_error", iters=300)
    assert r1 == r2


def test_bootstrap_returns_none_without_overlap():
    assert paired_bootstrap(_err([0.1]), {}, "mean_point_error") is None


def test_significance_label():
    assert significance_label(0.01) == "significant"
    assert significance_label(0.2) == "not significant"
